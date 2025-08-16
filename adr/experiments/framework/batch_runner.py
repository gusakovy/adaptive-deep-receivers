import os
import itertools
from copy import deepcopy
from tqdm import tqdm
import argparse
from adr.experiments.utils import load_config, generate_config_hash
from adr.experiments.framework.single_runner import run_experiment, save_results, clean_config


def expand_config_lists(config: dict[str, any]) -> list[dict[str, any]]:
    """Expand configuration with list parameters into all possible combinations.

    Args:
        config (dict[str, any]): Configuration dictionary where some values can be lists.

    Returns:
        list[dict[str, any]]: List of configuration dictionaries with all possible parameter combinations.
    """
    list_params = {}
    def find_lists(obj, path=""):
        """Recursively find all list parameters in the config."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, list):
                    list_params[current_path] = value
                elif isinstance(value, dict):
                    find_lists(value, current_path)
    find_lists(config)

    if not list_params:
        return [clean_config(config)]  # No lists found, return original config

    # Generate all combinations
    param_names = list(list_params.keys())
    param_values = list(list_params.values())
    configs = []
    seen_hashes = set()

    for combination in itertools.product(*param_values):
        new_config = deepcopy(config)

        for param_path, value in zip(param_names, combination):
            parts = param_path.split('.')
            current = new_config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
 
        cleaned_config = clean_config(new_config)
        config_hash = generate_config_hash(cleaned_config)
        if config_hash not in seen_hashes:
            seen_hashes.add(config_hash)
            configs.append(cleaned_config)

    return configs

def run_batch_experiments(
    config_path: str,
    output_dir: str = 'adr/experiments/results', 
    verbose: bool = True
    ) -> dict[str, any]:
    """Run batch experiments for all parameter combinations in config.

    Args:
        config_path (str): Path to batch configuration file.
        output_dir (str, optional): Base directory for saving results. Defaults to 'adr/experiments/results'.
        verbose (bool, optional): Whether to show progress bars and detailed output. Defaults to True.

    Returns:
        dict[str, any]: Dictionary with batch run summary and results.
    """
    base_config = load_config(config_path)
    configs = expand_config_lists(base_config)

    if verbose:
        print(f"Found {len(configs)} parameter combinations to run")
        if len(configs) > 1:
            print("Parameter combinations:")
            list_params = {}
            def find_lists(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if isinstance(value, list):
                            list_params[current_path] = value
                        elif isinstance(value, dict):
                            find_lists(value, current_path)
            find_lists(base_config)
            for param, values in list_params.items():
                print(f"  {param}: {values}")

    # Track results
    batch_results = {
        'total_experiments': len(configs),
        'completed_experiments': 0,
        'failed_experiments': 0,
        'failure_messages': [],
        'experiment_hashes': [],
        'experiment_results': {},
        'config_path': config_path,
        'output_dir': output_dir
    }

    # Run experiments
    progress_bar = tqdm(configs, desc="Running experiments") if verbose else configs

    for i, config in enumerate(progress_bar):
        try:
            config_hash = generate_config_hash(config)
            experiment_dir = os.path.join(output_dir, config_hash)
            if os.path.exists(experiment_dir):
                batch_results['experiment_hashes'].append(config_hash)
                batch_results['completed_experiments'] += 1
                continue

            results, model = run_experiment(config)
            save_results(results, model, config, output_dir)
            batch_results['experiment_hashes'].append(config_hash)
            batch_results['experiment_results'][config_hash] = {
                'final_sync_ber': results['final_sync_ber'],
                'config': config
            }
            batch_results['completed_experiments'] += 1

        except Exception as e:
            batch_results['failed_experiments'] += 1
            batch_results['failure_messages'].append(f"Error running experiment {config_hash}: {e}\n")
            # raise e # For debugging
            continue

    batch_results_clean = deepcopy(batch_results)
    for hash_key in batch_results_clean['experiment_results']:
        if 'config' in batch_results_clean['experiment_results'][hash_key]:
            del batch_results_clean['experiment_results'][hash_key]['config']

    if verbose:
        print(f"\nBatch run complete!")
        print(f"Total experiments: {batch_results['total_experiments']}")
        print(f"Completed: {batch_results['completed_experiments']}")
        print(f"Failed: {batch_results['failed_experiments']}")
        if batch_results['failed_experiments'] > 0:
            print(f"Failure messages: {batch_results['failure_messages']}")

    return batch_results

def main():
    """Main function for batch experiment runner."""
    parser = argparse.ArgumentParser(description='Run batch experiments from config with list parameters')
    parser.add_argument('--config_path', type=str, 
                        help='Path to batch experiment config JSON file',
                        default='adr/experiments/framework/batch_config.json')
    parser.add_argument('--output_dir', type=str, 
                        help='Base output directory for results',
                        default='adr/experiments/results')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Config file not found: {args.config_path}")
        return

    # Run batch experiments
    results = run_batch_experiments(
        config_path=args.config_path,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\nBatch experiments complete with {results['completed_experiments']} successful runs")

if __name__ == "__main__":
    main()

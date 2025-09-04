import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from adr.experiments.utils import load_config, generate_config_hash
from adr.experiments.framework.single_runner import run_experiment, save_results, load_experiment_by_hash


def find_array_param(config: dict[str, any]) -> tuple[str, list]:
    """Find the first array parameter in the config and its values."""
    def find_arrays(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, list):
                    # Skip the seed parameter as it's handled separately
                    if current_path == "experiment.seed":
                        continue
                    return current_path, value
                elif isinstance(value, dict):
                    result = find_arrays(value, current_path)
                    if result is not None:
                        return result
        return None

    result = find_arrays(config)
    if result is None:
        raise ValueError("No array parameter found in config")
    return result

def get_seed_values(config: dict[str, any]) -> list:
    """Get seed values from config, always returning a list."""
    if 'experiment' in config and 'seed' in config['experiment']:
        seed_value = config['experiment']['seed']
        if isinstance(seed_value, list):
            return seed_value
        else:
            return [seed_value]
    return []

def is_y_param_array(experiment_data: dict[str, any], y_param: str) -> bool:
    """Check if the y_param is an array in the experiment results."""
    if y_param not in experiment_data['results']:
        return False
    value = experiment_data['results'][y_param]
    return isinstance(value, list) and len(value) > 0

def load_compared_configs(compared_configs_dir: str) -> dict[str, dict[str, any]]:
    """Load best configurations for each algorithm from a directory."""
    compared_configs = {}
    config_files = sorted([f for f in os.listdir(compared_configs_dir) if f.endswith('.json')])

    # Load the configs
    for config_file in config_files:
        config_name = os.path.splitext(config_file)[0]
        config_path = os.path.join(compared_configs_dir, config_file)
        with open(config_path, 'r') as f:
            config = json.load(f)
        compared_configs[config_name] = config

    return compared_configs

def get_plot_params(config: dict[str, any]) -> dict[str, any]:
    """Extract plot parameters (color, marker, linestyle) from config if available."""
    plot_params = {}
    if 'plot' in config:
        plot_config = config['plot']
        for param in ['color', 'marker', 'linestyle', 'markersize', 'linewidth', 'alpha', 'capsize', 'capthick']:
            if param in plot_config:
                plot_params[param] = plot_config[param]
    return plot_params

def run_experiments_for_param(
    base_config: dict[str, any],
    compared_configs: dict[str, dict[str, any]],
    array_param: str,
    array_values: list,
    output_dir: str,
    y_param: str = 'avg_track_ber',
    no_run: bool = False
    ) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, any]]]:
    """Run experiments for each algorithm and array value combination."""
    results = {}
    plot_params = {}
    seed_values = get_seed_values(base_config)
    has_multiple_seeds = len(seed_values) > 1

    for config_name in compared_configs.keys():
        results[config_name] = {'x_values': [], 'y_values': [], 'y_errors': []}
        plot_params[config_name] = get_plot_params(compared_configs[config_name])
        compared_configs[config_name].pop('plot', None)

        for array_param_value in tqdm(array_values, desc=f"Running {config_name} for different {array_param.split('.')[-1].replace('_', ' ').title()} values"):
            # Create config for this combination
            config = base_config.copy()
            for key, value in compared_configs[config_name].items():
                config[key] = value

            # Set the array parameter value
            parts = array_param.split('.')
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = array_param_value

            # Run experiments for each seed
            seed_results = []
            for seed_value in tqdm(seed_values, desc="Running seeds", leave=False, disable=not has_multiple_seeds):
                config['experiment']['seed'] = seed_value
                config_hash = generate_config_hash(config)

                try:
                    experiment_data = load_experiment_by_hash(config_hash, output_dir)
                except FileNotFoundError:
                    experiment_data = None

                try:
                    if experiment_data is None and not no_run:
                        exp_results, model = run_experiment(config)
                        save_results(exp_results, model, config, output_dir)
                        experiment_data = load_experiment_by_hash(config_hash, output_dir)

                    if experiment_data is None:
                        print(f"Warning: No results found for {config_name} at {array_param}={array_param_value}, seed={seed_value}")
                        continue

                    seed_results.append(experiment_data['results'][y_param])

                except Exception as e:
                    print(f"Warning: Failed to run/load experiment for {config_name} at {array_param}={array_param_value}, seed={seed_value}: {e}")
                    # raise e # For debugging

            if seed_results:
                seed_results = np.array(seed_results)
                mean_result = np.mean(seed_results)

                results[config_name]['x_values'].append(array_param_value)
                results[config_name]['y_values'].append(mean_result)

                # Calculate standard error for multiple seeds
                if has_multiple_seeds:
                    std_result = np.std(seed_results, axis=0)
                    results[config_name]['y_errors'].append(std_result)
                else:
                    results[config_name]['y_errors'].append(np.nan)
            else:
                print(f"Warning: No valid results found for {config_name} at {array_param}={array_param_value} across all seeds")

    return results, plot_params

def create_plot(
    results: dict[str, dict[str, list]],
    array_param: str,
    y_param: str = 'avg_track_ber',
    save_dir: str = None,
    log_scale: bool = True,
    plot_params: dict[str, dict[str, any]] = None,
    no_title: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    save_name: str = None,
    no_error_bars: bool = False
    ) -> plt.Figure:
    """Create a plot comparing all algorithms against the array parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(results))

    for (algorithm, data), color in zip(results.items(), colors):
        x_values = np.array(data['x_values'])
        y_values = np.array(data['y_values'])

        if 'y_errors' in data:
            y_errors = np.array(data['y_errors'])
        else:
            y_errors = None

        # Sort by x-values for proper line plotting
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        y_sorted = y_values[sort_idx]
        y_errors_sorted = y_errors[sort_idx] if y_errors is not None else None
        if no_error_bars:
            y_errors_sorted = None

        # Plot line with markers and error bars
        ax.errorbar(
            x=x_sorted,
            y=y_sorted,
            yerr=y_errors_sorted,
            color=plot_params[algorithm].get('color', color),
            marker=plot_params[algorithm].get('marker', 'o'),
            linestyle=plot_params[algorithm].get('linestyle', '-'),
            label=algorithm,
            linewidth=plot_params[algorithm].get('linewidth', 2),
            markersize=plot_params[algorithm].get('markersize', 6),
            alpha=plot_params[algorithm].get('alpha', 0.8),
            capsize=plot_params[algorithm].get('capsize', 5),
            capthick=plot_params[algorithm].get('capthick', 2)
            )

    # Formatting
    param_name = array_param.split('.')[-1].replace('_', ' ').title()
    y_param_name = y_param.replace('_', ' ').title()
    ax.set_xlabel(param_name if xlabel is None else xlabel, fontsize=20)
    ax.set_ylabel(y_param_name if ylabel is None else ylabel, fontsize=20)
    if log_scale:
        ax.set_yscale('log')
    if not no_title:
        ax.set_title(f'{y_param_name} vs. {param_name}')
    ax.grid(which="both")
    # remove the errorbars from the legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='best', fontsize=15)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_param_name = array_param.replace('.', '_').replace('/', '_')
        save_path = os.path.join(save_dir, save_name if save_name else f"{y_param}_vs_{safe_param_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def run_experiments_for_time_series(
    base_config: dict[str, any],
    compared_configs: dict[str, dict[str, any]],
    output_dir: str,
    y_param: str = 'ber',
    no_run: bool = False
    ) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, any]]]:
    """Run experiments for time series plotting where x-axis is time index."""
    results = {}
    plot_params = {}
    seed_values = get_seed_values(base_config)
    has_multiple_seeds = len(seed_values) > 1

    for config_name in tqdm(compared_configs.keys(), desc="Running compared configs"):
        results[config_name] = {'x_values': [], 'y_values': [], 'y_errors': []}
        plot_params[config_name] = get_plot_params(compared_configs[config_name])
        compared_configs[config_name].pop('plot', None)

        # Create config for this combination
        config = base_config.copy()
        for key, value in compared_configs[config_name].items():
            config[key] = value

        # Run experiments for each seed
        seed_results = []
        for seed_value in tqdm(seed_values, desc="Running seeds", leave=False, disable=not has_multiple_seeds):
            config['experiment']['seed'] = seed_value
            config_hash = generate_config_hash(config)

            try:
                experiment_data = load_experiment_by_hash(config_hash, output_dir)
            except FileNotFoundError:
                experiment_data = None

            try:
                if experiment_data is None and not no_run:
                    exp_results, model = run_experiment(config)
                    save_results(exp_results, model, config, output_dir)
                    experiment_data = load_experiment_by_hash(config_hash, output_dir)

                if experiment_data is None:
                    print(f"Warning: No results found for {config_name}, seed={seed_value}")
                    continue

                if is_y_param_array(experiment_data, y_param):
                    y_array = experiment_data['results'][y_param]
                    seed_results.append(y_array)
                else:
                    print(f"Warning: {y_param} is not an array for {config_name}")

            except Exception as e:
                print(f"Warning: Failed to run/load experiment for {config_name}: {e}")
                # raise e # For debugging

        if seed_results:
            seed_results = np.array(seed_results)
            mean_results = np.mean(seed_results, axis=0)
            time_indices = list(range(len(mean_results)))

            results[config_name]['x_values'] = time_indices
            results[config_name]['y_values'] = mean_results.tolist()

            # Calculate upper and lower error bars for multiple seeds
            if has_multiple_seeds:
                upper_errors = np.array([np.mean(results[results > mean] - mean) if np.any(results > mean) else 0 
                                       for results, mean in zip(seed_results.T, mean_results)])
                lower_errors = np.array([np.mean(mean - results[results < mean]) if np.any(results < mean) else 0 
                                       for results, mean in zip(seed_results.T, mean_results)])
                lower_errors[mean_results-lower_errors <= 1e-10] = 0
                results[config_name]['y_errors'] = [lower_errors, upper_errors]
            else:
                results[config_name]['y_errors'] = None
        else:
            print(f"Warning: No valid results found for {config_name} across all seeds")

    return results, plot_params

def create_time_series_plot(
    results: dict[str, dict[str, list]],
    y_param: str = 'ber',
    save_dir: str = None,
    log_scale: bool = True,
    plot_params: dict[str, dict[str, any]] = {},
    no_title: bool = False,
    xlabel: str = None,
    ylabel: str = None,
    save_name: str = None,
    no_error_bars: bool = False
    ) -> plt.Figure:
    """Create a time series plot where x-axis is time index."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(results))

    for (algorithm, data), color in zip(results.items(), colors):
        if not data['y_values']:
            continue

        x_values = np.array(data['x_values'])
        y_values = np.array(data['y_values'])
        if 'y_errors' in data and data['y_errors'] is not None and not no_error_bars:
            y_errors = np.array(data['y_errors'])
        else:
            y_errors = None

        # Plot line with markers
        ax.plot(
            x_values,
            y_values,
            color=plot_params[algorithm].get('color', color),
            marker=plot_params[algorithm].get('marker', 'o'),
            markevery=10,
            linestyle=plot_params[algorithm].get('linestyle', '-'),
            label=algorithm,
            linewidth=plot_params[algorithm].get('linewidth', 2),
            markersize=plot_params[algorithm].get('markersize', 6),
            alpha=plot_params[algorithm].get('alpha', 0.8),
            )

        # Add shaded error region if errors exist
        if y_errors is not None:
            ax.fill_between(
                x_values,
                y_values - y_errors[0],
                y_values + y_errors[1],
                color=plot_params[algorithm].get('color', color),
                alpha=0.2
                )

    # Formatting
    y_param_name = y_param.replace('_', ' ').title()
    ax.set_xlabel('Channel Snapshot' if xlabel is None else xlabel, fontsize=20)
    ax.set_ylabel(y_param_name if ylabel is None else ylabel, fontsize=20)
    if log_scale:
        ax.set_yscale('log')
    if not no_title:
        ax.set_title(f'{y_param_name} vs. Time')
    ax.grid(which="both")
    ax.legend(loc='best', fontsize=15)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name if save_name else f"{y_param}_time_series.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def main():
    """Main function for plotting best configurations."""
    parser = argparse.ArgumentParser(description='Plot results using best configurations')
    parser.add_argument('--compared_configs_dir', type=str,
                        help='Path to directory containing compared config files for each algorithm',
                        default='adr/experiments/best_algo_configs')
    parser.add_argument('--base_config', type=str,
                        help='Path to base configuration file',
                        default='adr/experiments/framework/scenario_config.json')
    parser.add_argument('--output_dir', type=str,
                        help='Directory for experiment results',
                        default='adr/experiments/results')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save plots',
                        default='adr/experiments/plots')
    parser.add_argument('--y_param', type=str,
                        help='Parameter to plot on y-axis (from experiment results)',
                        default='avg_track_ber')
    parser.add_argument('--no_run', action='store_true',
                        help='Skip running missing experiments')
    parser.add_argument('--no_log_scale', action='store_true',
                        help='Disable log scale for y-axis')
    parser.add_argument('--time_series', action='store_true',
                        help='Plot time series data where x-axis is time index (no array param search)')
    parser.add_argument('--no_error_bars', action='store_true',
                        help='Disable error bars for the plot')
    parser.add_argument('--no_title', action='store_true',
                        help='Disable title for the plot')
    parser.add_argument('--xlabel', type=str,
                        help='Label for x-axis',
                        default=None)
    parser.add_argument('--ylabel', type=str,
                        help='Label for y-axis',
                        default=None)
    parser.add_argument('--save_name', type=str,
                        help='Name of the plot',
                        default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.base_config):
        print(f"Config file not found: {args.base_config}")
        return

    if not os.path.exists(args.compared_configs_dir):
        print(f"Compared configs directory not found: {args.compared_configs_dir}")
        return

    # Load compared configs
    compared_configs = load_compared_configs(args.compared_configs_dir)
    print(f"Loaded compared configs for algorithms: {list(compared_configs.keys())}")

    # Load base config
    base_config = load_config(args.base_config)

    # Check for multiple seeds in the base config
    seed_values = get_seed_values(base_config)
    if len(seed_values) > 1:
        print(f"Found multiple seeds: {seed_values}")
        print("Will run experiments for each seed and average results with error bars.")

    if args.time_series: # Run experiments for time series
        results, plot_params = run_experiments_for_time_series(
            base_config=base_config,
            compared_configs=compared_configs,
            output_dir=args.output_dir,
            y_param=args.y_param,
            no_run=args.no_run
        )

        if not results:
            print("No results found!")
            return

        # Create time series plot
        create_time_series_plot(
            results=results,
            y_param=args.y_param,
            save_dir=args.save_dir,
            log_scale=not args.no_log_scale,
            plot_params=plot_params,
            no_title=args.no_title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            save_name=args.save_name,
            no_error_bars=args.no_error_bars
        )

    else: # Regular mode: find array parameter in config
        array_param, array_values = find_array_param(base_config)
        print(f"Found array parameter: {array_param}")
        print(f"Values: {array_values}")

        results, plot_params = run_experiments_for_param(
            base_config=base_config,
            compared_configs=compared_configs,
            array_param=array_param,
            array_values=array_values,
            output_dir=args.output_dir,
            y_param=args.y_param,
            no_run=args.no_run
        )

        if not results:
            print("No results found!")
            return

        create_plot(
            results=results,
            array_param=array_param,
            y_param=args.y_param,
            save_dir=args.save_dir,
            log_scale=not args.no_log_scale,
            plot_params=plot_params,
            no_title=args.no_title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            save_name=args.save_name,
            no_error_bars=args.no_error_bars
        )

    if not args.save_dir:
        plt.show()

if __name__ == "__main__":
    main()

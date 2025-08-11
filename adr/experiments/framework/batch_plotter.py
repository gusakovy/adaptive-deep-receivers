from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from adr.experiments.utils import load_config, generate_config_hash
from adr.experiments.framework.batch_runner import run_batch_experiments, expand_config_lists
from adr.experiments.framework.single_runner import load_experiment_by_hash


def load_batch_results(
    config_path: str,
    output_dir: str = 'adr/experiments/results',
    ensure_complete: bool = True
    ) -> dict[str, any]:
    """Load results for all experiments in a batch configuration.

    Args:
        config_path (str): Path to batch configuration file.
        output_dir (str, optional): Directory containing experiment results. Defaults to 'adr/experiments/results'.
        ensure_complete (bool, optional): Whether to run missing experiments first. Defaults to True.

    Returns:
        Dict[str, Any]: Dictionary mapping config hashes to experiment data.
    """
    if ensure_complete:
        print("Ensuring all experiments are complete...")
        run_batch_experiments(config_path, output_dir, verbose=True)

    # Load the base configuration and expand it
    base_config = load_config(config_path)
    configs = expand_config_lists(base_config)

    # Load results for each configuration
    batch_results = {}
    for config in configs:
        config_hash = generate_config_hash(config)
        try:
            experiment_data = load_experiment_by_hash(config_hash, output_dir)
            experiment_data['config'] = config
            batch_results[config_hash] = experiment_data
        except Exception as e:
            print(f"Warning: Could not load experiment {config_hash}: {e}")
    return batch_results

def extract_config_value(config: dict[str, any], param_path: str) -> any:
    """Extract a configuration value given a dot-separated path."""
    parts = param_path.split('.')
    current = config
    for part in parts:
        current = current[part]
    return current

def param_exists_in_config(param: str, config: dict[str, any]) -> bool:
    """Check if a dot-notation parameter exists in the config."""
    try:
        parts = param.split('.')
        current = config
        for part in parts:
            current = current[part]
        return True
    except (KeyError, TypeError):
        return False

def group_results_by_params(
    batch_results: dict[str, any],
    x_param: str,
    y_metric: str,
    group_params: list[str] = None
    ) -> dict[str, dict[str, list]]:
    """Group results by parameter combinations for plotting.

    Args:
        batch_results (dict[str, any]): Dictionary of experiment results.
        x_param (str): Parameter to use for x-axis.
        y_metric (str): Metric to plot on y-axis.
        group_params (list[str], optional): Parameters to group by (create separate lines). Defaults to None.

    Returns:
        dict[str, dict[str, list]]: Nested dictionary: {group_key: {x_values: [...], y_values: [...], configs: [...]}}.
    """
    if group_params is None:
        group_params = []

    # Filter out parameters that don't exist in any config
    valid_group_params = []
    for param in group_params:
        if any(param_exists_in_config(param, experiment_data['config']) for experiment_data in batch_results.values()):
            valid_group_params.append(param)

    grouped_data = defaultdict(lambda: {'x_values': [], 'y_values': [], 'configs': []})

    for config_hash, experiment_data in batch_results.items():
        config = experiment_data['config']
        results = experiment_data['results']

        # Skip if x_param doesn't exist in this config
        try:
            x_value = extract_config_value(config, x_param)
        except KeyError:
            continue

        # Create group key from group parameters
        if valid_group_params:
            group_values = []
            for param in valid_group_params:
                try:
                    value = extract_config_value(config, param)
                    # Use parameter name without full path for cleaner labels
                    param_name = param.split('.')[-1]
                    group_values.append(f"{param_name}={value}")
                except KeyError:
                    continue
            if group_values:  # Only create group if we have valid parameters
                group_key = ", ".join(group_values)
            else:
                group_key = "all"
        else:
            group_key = "all"

        # Store data
        grouped_data[group_key]['x_values'].append(x_value)
        grouped_data[group_key]['y_values'].append(results[y_metric])
        grouped_data[group_key]['configs'].append(config)

    return dict(grouped_data)

def create_plot(
    grouped_data: dict[str, dict[str, list]], 
    x_param: str,
    y_metric: str = 'avg_track_ber',
    title: str = None,
    save_dir: str = None,
    log_scale: bool = False
    ) -> plt.Figure:
    """Create a plot from grouped experiment data.

    Args:
        grouped_data (dict[str, dict[str, list]]): Grouped data from group_results_by_params.
        x_param (str): Name of x-axis parameter.
        y_metric (str, optional): Name of y-axis metric. Defaults to 'avg_track_ber'.
        title (str, optional): Plot title. Defaults to None.
        save_dir (str, optional): Directory to save plot (filename auto-generated). Defaults to None.
        log_scale (bool, optional): Whether to use log scale for y-axis. Defaults to False.
        
    Returns:
        plt.Figure: Matplotlib figure object.
    """
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a large color palette that can handle many combinations
    n_groups = len(grouped_data)
    if n_groups <= 10:
        # Use seaborn's default palette for small numbers
        colors = sns.color_palette("tab10", n_groups)
    elif n_groups <= 20:
        # Use a larger qualitative palette
        colors = sns.color_palette("tab20", n_groups)
    else:
        # For very large numbers, use a continuous colormap
        colors = sns.color_palette("husl", n_groups)

    # Define different line styles to help distinguish lines further
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']

    # Plot each group as a separate line
    for i, (group_key, data) in enumerate(grouped_data.items()):
        x_values = np.array(data['x_values'])
        y_values = np.array(data['y_values'])

        # Sort by x-values for proper line plotting
        sort_idx = np.argsort(x_values)
        x_sorted = x_values[sort_idx]
        y_sorted = y_values[sort_idx]

        # Use different colors, line styles, and markers for better distinction
        color = colors[i % len(colors)]
        linestyle = line_styles[(i//len(colors)) % len(line_styles)]
        marker = markers[i % len(markers)]

        # Plot line with markers
        ax.plot(x_sorted, y_sorted, 
               color=color,
               linestyle=linestyle,
               marker=marker,
               label=group_key, 
               linewidth=2, 
               markersize=6,
               alpha=0.8)

    # Formatting
    ax.set_xlabel(x_param.replace('.', ' ').replace('_', ' ').title())
    ax.set_ylabel(y_metric.replace('_', ' ').title())

    if log_scale:
        ax.set_yscale('log')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_param.replace(".", " ").replace("_", " ").title()}')

    # Legend - handle many entries gracefully
    if len(grouped_data) > 1:
        if len(grouped_data) <= 30:
            # Normal legend for reasonable number of entries
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Compact legend for many entries
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize='small', ncol=2)
        plt.tight_layout()

    # Grid
    ax.grid(True, alpha=0.3)

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_param_name = x_param.replace('.', '_').replace('/', '_')
        save_path = os.path.join(save_dir, f"{y_metric}_vs_{safe_param_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig

def plot_batch_results(config_path: str,
                      x_param: str,
                      y_metric: str = 'avg_track_ber',
                      group_params: list[str] = None,
                      output_dir: str = 'adr/experiments/results',
                      save_dir: str = None,
                      title: str = None,
                      log_scale: bool = False,
                      ensure_complete: bool = True,
                      batch_results: dict[str, any] = None) -> plt.Figure:
    """Complete pipeline to plot batch experiment results.

    Args:
        config_path (str): Path to batch configuration file.
        x_param (str): Parameter to plot on x-axis (dot-separated path).
        y_metric (str): Metric to plot on y-axis.
        group_params (list[str], optional): Parameters to group by (create separate lines). Defaults to None.
        output_dir (str): Directory containing experiment results.
        save_dir (str, optional): Directory to save plot (filename auto-generated). Defaults to None.
        title (str, optional): Custom plot title. Defaults to None.
        log_scale (bool, optional): Whether to use log scale for y-axis. Defaults to False.
        ensure_complete (bool, optional): Whether to run missing experiments first. Defaults to True.
        batch_results (dict[str, any], optional): Pre-loaded batch results to use instead of loading from disk. Defaults to None.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    # Load all batch results if not provided
    if batch_results is None:
        batch_results = load_batch_results(config_path, output_dir, ensure_complete)

    if not batch_results:
        raise ValueError("No experiment results found!")

    print(f"Loaded {len(batch_results)} experiment results")

    # Group results for plotting
    grouped_data = group_results_by_params(batch_results, x_param, y_metric, group_params)

    print(f"Found {len(grouped_data)} parameter groups")
    for group_key, data in grouped_data.items():
        print(f"  {group_key}: {len(data['x_values'])} experiments")

    # Create plot
    fig = create_plot(grouped_data, x_param, y_metric, title, save_dir, log_scale)

    return fig

def auto_detect_array_params(config_path: str) -> list[str]:
    """Automatically detect which parameters are arrays in the config."""
    base_config = load_config(config_path)

    array_params = []

    def find_arrays(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, list):
                    array_params.append(current_path)
                elif isinstance(value, dict):
                    find_arrays(value, current_path)

    find_arrays(base_config)
    return array_params

def create_plot_grid(config_path: str,
                    output_dir: str = 'adr/experiments/results',
                    y_metric: str = 'avg_track_ber',
                    save_dir: str = None,
                    ensure_complete: bool = True) -> list[plt.Figure]:
    """Create a grid of plots for all array parameters in the config.

    Args:
        config_path (str): Path to batch configuration file.
        output_dir (str, optional): Directory containing experiment results. Defaults to 'adr/experiments/results'.
        y_metric (str, optional): Metric to plot on y-axis. Defaults to 'avg_track_ber'.
        save_dir (str, optional): Directory to save plots. Defaults to None.
        ensure_complete (bool, optional): Whether to run missing experiments first. Defaults to True.

    Returns:
        list[plt.Figure]: List of matplotlib figure objects.
    """
    # Detect array parameters
    array_params = auto_detect_array_params(config_path)

    if not array_params:
        print("No array parameters found in config")
        return []

    print(f"Found array parameters: {array_params}")

    # Load results once for all plots
    batch_results = load_batch_results(config_path, output_dir, ensure_complete)

    figures = []

    for x_param in array_params:
        # Check if any configs have this parameter
        has_param = any(param_exists_in_config(x_param, experiment_data['config']) 
                       for experiment_data in batch_results.values())
        if not has_param:
            print(f"\nSkipping plot for {x_param} - parameter not found in any configs")
            continue

        # Use ALL other array parameters as grouping variables for this specific plot
        group_params = [p for p in array_params if p != x_param]

        print(f"\nCreating plot for {x_param}")
        if group_params:
            print(f"  Grouping by: {group_params}")

        try:
            fig = plot_batch_results(
                config_path=config_path,
                x_param=x_param,
                y_metric=y_metric,
                group_params=group_params if len(group_params) > 0 else None,
                output_dir=output_dir,
                save_dir=save_dir,
                ensure_complete=ensure_complete,
                log_scale=True,
                batch_results=batch_results
            )
            figures.append(fig)
        except Exception as e:
            print(f"Failed to create plot for {x_param}: {e}")

    return figures

def get_all_other_array_params(config_path: str, exclude_param: str = None) -> list[str]:
    """Get all array parameters except the excluded one."""
    array_params = auto_detect_array_params(config_path)
    if exclude_param:
        return [p for p in array_params if p != exclude_param]
    return array_params

def main():
    """Main function for the plotting script."""
    parser = argparse.ArgumentParser(description='Plot batch experiment results')
    parser.add_argument('--config_path', type=str,
                        help='Path to batch configuration file',
                        default='adr/experiments/framework/batch_config.json')
    parser.add_argument('--x_param', type=str,
                        help='Parameter to plot on x-axis (dot-separated path)')
    parser.add_argument('--y_metric', type=str, default='avg_track_ber',
                        help='Metric to plot on y-axis')
    parser.add_argument('--group_params', type=str, nargs='*',
                        help='Parameters to group by (create separate lines)')
    parser.add_argument('--output_dir', type=str, default='adr/experiments/results',
                        help='Directory containing experiment results')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save plots (filename auto-generated)',
                        default='adr/experiments/plots')
    parser.add_argument('--title', type=str,
                        help='Custom plot title')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use log scale for y-axis')
    parser.add_argument('--auto_grid', action='store_true',
                        help='Automatically create plots for all array parameters')
    parser.add_argument('--no_run', action='store_true',
                        help='Skip running missing experiments')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.config_path):
        print(f"Config file not found: {args.config_path}")
        return

    if not args.x_param and not args.auto_grid:
        args.auto_grid = True

    if args.x_param:
        # Create single plot with all parameter combinations
        # If no group_params specified, use all other array parameters
        if args.group_params is None:
            group_params = get_all_other_array_params(args.config_path, args.x_param)
            if group_params:
                print(f"Auto-detected grouping parameters: {group_params}")
        else:
            group_params = args.group_params

        fig = plot_batch_results(
            config_path=args.config_path,
            x_param=args.x_param,
            y_metric=args.y_metric,
            group_params=group_params,
            output_dir=args.output_dir,
            save_dir=args.save_dir,
            title=args.title,
            log_scale=args.log_scale,
            ensure_complete=not args.no_run
        )

        if not args.save_dir:
            fig.show()

    elif args.auto_grid:
        # Create separate plots for each array parameter
        figures = create_plot_grid(
            config_path=args.config_path,
            output_dir=args.output_dir,
            y_metric=args.y_metric,
            save_dir=args.save_dir,
            ensure_complete=not args.no_run
        )
        print(f"Created {len(figures)} plots")

    else:
        print("Either --x_param or --auto_grid must be specified")
        print("Available array parameters:")
        array_params = auto_detect_array_params(args.config_path)
        for param in array_params:
            print(f"  {param}")

if __name__ == "__main__":
    main()

import os 
import random
import argparse
import yaml
import json
import datetime
import numpy as np
from pathlib import Path
from accelerate.utils import reduce
from accelerate import Accelerator
from typing import Any, Dict, List, get_origin, get_args, Union

import torch
from safetensors.torch import load_file

from models import CAMMST
from config.schemas import (
    Config, 
    GeneralConfig, 
    DataConfig, 
    TrainingConfig, 
    WandBConfig
)

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(_worker_id: int) -> None:
    """
    Worker initialization function for DataLoader to ensure reproducible results.
    
    Args:
        _worker_id (int): Worker ID (unused but required by DataLoader)
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_weight_decay(model: torch.nn.Module, weight_decay: float = 1e-5, skip_list: tuple = ()) -> List[Dict[str, Any]]:
    """
    Add weight decay to parameters that don't have it.
    Set weight decay to 0 for bias and 1D weight tensors.
    Following timm's implementation.
    
    Args:
        model (torch.nn.Module): Model to add weight decay to
        weight_decay (float): Weight decay value
        skip_list (tuple): List of parameter names to skip weight decay
    
    Returns:
        List[Dict[str, Any]]: List of parameter groups with weight decay
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class MetricTracker:
    """
    Distributed metrics tracker for Accelerate.
    
    Uses running averages instead of storing all values to minimize memory usage.
    Computes accurate global averages in distributed settings.
    """
    def __init__(self) -> None:
        """Initialize metric tracker."""
        self.running_metrics: Dict[str, float] = {}
        self.running_counts: Dict[str, int] = {}
        self.total_samples: int = 0
    
    def update(self, metrics_dict: Dict[str, torch.Tensor], batch_size: int = 1) -> None:
        """
        Update metrics using running averages for memory efficiency.
        
        Args:
            metrics_dict: Dictionary of metrics (key: metric name, value: tensor)
            batch_size: Number of samples in the current batch
            
        Raises:
            TypeError: If metrics_dict is not a dictionary
            ValueError: If batch_size <= 0
        """
        if not isinstance(metrics_dict, dict):
            raise TypeError(f"metrics_dict must be a dictionary, got {type(metrics_dict)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            
        self.total_samples += batch_size
        
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                current_value = value.item()
            else:
                current_value = float(value)
            
            if key not in self.running_metrics:
                self.running_metrics[key] = current_value
                self.running_counts[key] = batch_size
            else:
                # weighted running average
                total_count = self.running_counts[key] + batch_size
                self.running_metrics[key] = (
                    self.running_metrics[key] * self.running_counts[key] + 
                    current_value * batch_size
                ) / total_count
                self.running_counts[key] = total_count
    
    def compute_epoch_averages(self, accelerator) -> Dict[str, float]:
        """
        Compute accurate global averages across all processes.
        
        Args:
            accelerator: Accelerator instance for distributed operations
            
        Returns:
            Dictionary of globally averaged metrics
        """
        if self.total_samples == 0 or not self.running_metrics:
            return {}
            
        averages = {}
        
        if accelerator.use_distributed:
            # optimize: gather all metrics at once instead of separate calls
            metric_keys = list(self.running_metrics.keys())
            local_data = []
            
            for key in metric_keys:
                local_avg = self.running_metrics[key]
                local_count = self.running_counts[key]
                local_sum = local_avg * local_count
                local_data.extend([local_sum, local_count])
            
            if not local_data:
                return {}
                
            # single gather call for all metrics
            try:
                gathered_data = accelerator.gather_for_metrics(
                    torch.tensor(local_data, device=accelerator.device, dtype=torch.float32)
                )
                
                # fix: gather_for_metrics concatenates all process data in first dimension
                # gathered_data shape: [num_processes * len(local_data)]
                # we need to reshape to [num_processes, len(local_data)]
                num_metrics = len(metric_keys)
                data_per_process = num_metrics * 2  # each metric has [sum, count]
                num_processes = gathered_data.shape[0] // data_per_process
                
                # reshape to [num_processes, data_per_process]
                gathered_data = gathered_data.view(num_processes, data_per_process)
                
                for i, key in enumerate(metric_keys):
                    sum_idx = i * 2
                    count_idx = i * 2 + 1
                    
                    total_sum = gathered_data[:, sum_idx].sum()
                    total_count = gathered_data[:, count_idx].sum()
                    global_avg = total_sum / total_count if total_count > 0 else 0.0
                    averages[key] = global_avg.item()
                    
            except Exception as e:
                # fallback to local averages if gathering fails
                if accelerator.is_main_process:
                    accelerator.print(f"Warning: Failed to gather metrics, using local averages: {e}")
                averages = self.running_metrics.copy()
        else:
            # single process - use local averages
            averages = self.running_metrics.copy()
        
        return averages
    
    def reset(self) -> None:
        """Reset tracker for next epoch."""
        self.running_metrics.clear()
        self.running_counts.clear()
        self.total_samples = 0
    
    def get_current_averages(self) -> Dict[str, float]:
        """Get current local averages without distributed operations."""
        return self.running_metrics.copy()


class DistributedEarlyStopping:
    """
    Early stopping for distributed training based on the simple EarlyStopping class.
    Since validation scores are globally averaged, all processes make the same decision.
    """
    def __init__(self, patience: int = 10, delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True) -> None:
        """
        Initialize distributed early stopping.

        Args:
            patience (int): Number of epochs to wait before stopping
            delta (float): Minimum change threshold to qualify as improvement
            mode (str): 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
            verbose (bool): Whether to print early stopping messages
            
        Raises:
            ValueError: If mode is not 'min' or 'max'
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
            
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        # initialize best score properly for the given mode
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.mode = mode
        self.delta = delta
        
    def __call__(self, score: float, accelerator) -> bool:
        """
        Check if early stopping should be triggered with distributed synchronization.
        
        Args:
            score (float): Current validation score (should be globally averaged)
            accelerator: Accelerator instance for distributed operations
            
        Returns:
            bool: True if early stopping should be triggered (synchronized across all processes)
        """
        # only main process updates counters to avoid race conditions
        if accelerator.is_main_process:
            if self.mode == 'min':
                if score < (self.best_score - self.delta):
                    self.counter = 0
                    self.best_score = score
                    if self.verbose:
                        accelerator.print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}\n')
                else:
                    self.counter += 1
                    if self.verbose:
                        msg = (
                            f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, '
                            f'Best: {self.best_score:.5f}, Current: {score:.5f}, '
                            f'Delta: {np.abs(self.best_score - score):.5f}\n'
                        )
                        accelerator.print(msg)

            elif self.mode == 'max':
                if score > (self.best_score + self.delta):
                    self.counter = 0
                    self.best_score = score
                    if self.verbose:
                        accelerator.print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}\n')
                else:
                    self.counter += 1
                    if self.verbose:
                        msg = (
                            f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, '
                            f'Best: {self.best_score:.5f}, Current: {score:.5f}, '
                            f'Delta: {np.abs(self.best_score - score):.5f}\n'
                        )
                        accelerator.print(msg)

            # check patience only on main
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    accelerator.print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')

        # broadcast decision from main to all ranks
        stop_tensor = torch.tensor(1 if (accelerator.is_main_process and self.early_stop) else 0, device=accelerator.device)
        stop_tensor = reduce(stop_tensor, reduction="max")
        global_stop = stop_tensor.item() == 1
        # ensure all ranks share same flag locally
        self.early_stop = global_stop
        return global_stop


# --- Config Loading Functions ---

def parse_key_value_pairs(set_args: List[List[str]] | None) -> Dict[str, Any]:
    """
    Parse --set key=value arguments into a nested dictionary.
    
    Args:
        set_args: List of key=value strings from --set argument
        
    Returns:
        dict: Parsed key-value pairs as nested dictionary
        
    Example:
        ```python
        # --set general.seed=42 training.lr=0.001
        args = [['general.seed=42'], ['training.lr=0.001']]
        result = parse_key_value_pairs(args)
        # {'general': {'seed': 42}, 'training': {'lr': 0.001}}
        ```
        
    Raises:
        ValueError: If key=value format is invalid
    """
    overrides: Dict[str, Any] = {}
    
    if not set_args:
        return overrides
    
    for arg_group in set_args:
        for kv_pair in arg_group:
            if '=' not in kv_pair:
                raise ValueError(f"Invalid --set format: '{kv_pair}'. Expected KEY=VALUE")
                
            key_path, value = kv_pair.split('=', 1)
            
            # nested key path (e.g., "general.seed" -> ["general", "seed"])
            keys = key_path.split('.')
            
            if not all(keys):
                raise ValueError(f"Invalid key path: '{key_path}'. Empty key component")
            
            # type inference for value
            parsed_value: Any
            try:
                if value.lower() in ('true', 'false'):
                    parsed_value = value.lower() == 'true'
                elif value.lower() == 'none':
                    parsed_value = None
                elif '.' in value and value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                    parsed_value = float(value)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    parsed_value = int(value)
                else:
                    parsed_value = value
            except (ValueError, AttributeError):
                parsed_value = value
            
            # build nested dictionary
            current = overrides
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = parsed_value
    
    return overrides


def validate_config_keys(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """
    Validate that all keys in overrides exist in the config.
    
    Args:
        config: Base configuration dictionary
        overrides: Dictionary containing overrides to validate
        
    Raises:
        ValueError: If any key in overrides doesn't exist in config
    """
    def check_nested_keys(base_dict: Dict[str, Any], override_dict: Dict[str, Any], path: str = "") -> None:
        """Recursively check if all override keys exist in base config."""
        for key, value in override_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in base_dict:
                available_keys = list(base_dict.keys())
                raise ValueError(
                    f"Config key '{current_path}' does not exist. "
                    f"Available keys at this level: {available_keys}"
                )
            
            if isinstance(value, dict) and isinstance(base_dict[key], dict):
                check_nested_keys(base_dict[key], value, current_path)
    
    check_nested_keys(config, overrides)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply CLI argument overrides to the loaded config.
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command line arguments (must have 'set' attribute)
        
    Returns:
        dict: Updated configuration with CLI overrides
        
    Raises:
        ValueError: If any --set key doesn't exist in config
    """
    # parse --set arguments
    set_overrides = parse_key_value_pairs(getattr(args, 'set', None))
    
    # validate that all override keys exist in config
    if set_overrides:
        validate_config_keys(config, set_overrides)
    
    # apply overrides using deep update
    def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config, set_overrides)
    
    return config


def load_config(config_path: str, args: argparse.Namespace | None = None) -> Config:
    """
    Load and validate config from YAML file with Pydantic.
    
    Supports CLI overrides via --set KEY=VALUE.
    
    Args:
        config_path: Path to YAML configuration file
        args: Optional parsed command line arguments for overrides
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If config validation fails
    """
    # check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # load yaml file
    with open(config_path, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # apply CLI overrides if args provided
    if args is not None:
        config_dict = apply_cli_overrides(config_dict, args)
    
    # validate with pydantic
    return Config(**config_dict)


def load_model(
    dataset: str,
    fold: int,
    model_path: str | Path | None = None,
    device: str = "auto",
    config: Config | None = None,
) -> tuple:
    """
    Load CAMMST model from output_dir structure (dataset/fold_X/best_model/).

    Args:
        dataset: Dataset name (e.g., 'her2st', 'skin', 'stnet')
        fold: Fold number (0-7)
        model_path: Output directory path containing trained models
        device: Device to load model on ('auto', 'cpu', 'cuda')

    Returns:
        Tuple of (loaded CAMMST model, device string)

    Raises:
        ValueError: If fold number is invalid
        FileNotFoundError: If model or config files are not found
    """
    if not (0 <= fold <= 7):
        raise ValueError(f"Fold number must be between 0 and 7, got {fold}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path is None:
        model_path = Path("./output_dir") / dataset
    else:
        model_path = Path(model_path)

    fold_dir = model_path / f"fold_{fold}"
    checkpoint_dir = fold_dir / "best_model"

    if config is None:
        config_path = fold_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_config(str(config_path))

    model = CAMMST(**config.model.model_dump())

    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path), device=device)
        model.load_state_dict(state_dict)
    else:
        pytorch_path = checkpoint_dir / "pytorch_model.bin"
        if pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"No model file found in {checkpoint_dir}. "
                "Expected model.safetensors or pytorch_model.bin"
            )

    model.to(device)
    model.eval()

    return model, device


def print_kfold_summary(
    fold_results: List[Dict[str, Any]],
    n_folds: int,
    total_time_str: str,
    base_output_path: Path,
    accelerator: Accelerator
) -> None:
    """
    Print and save k-fold cross-validation summary.

    Computes average and standard deviation metrics for best models across all folds,
    prints results to terminal, and saves summary to JSON file.

    Args:
        fold_results: List of dictionaries containing fold results with keys:
            - 'fold': fold number
            - 'best_loss', 'best_mse', 'best_mae', 'best_pcc': best model metrics
            - 'best_all_masked_loss', 'best_all_masked_mse', 'best_all_masked_mae', 'best_all_masked_pcc': all masked metrics
            - 'training_time': training time in seconds
        n_folds: Number of folds in cross-validation
        total_time_str: Total training time as formatted string
        base_output_path: Path to output directory for saving summary JSON
        accelerator: Accelerator object for distributed printing
    """
    # extract metrics from fold results
    all_best_losses = [r['best_loss'] for r in fold_results]
    all_best_mses = [r['best_mse'] for r in fold_results]
    all_best_maes = [r['best_mae'] for r in fold_results]
    all_best_pccs = [r['best_pcc'] for r in fold_results]
    all_best_all_masked_losses = [r['best_all_masked_loss'] for r in fold_results]
    all_best_all_masked_mses = [r['best_all_masked_mse'] for r in fold_results]
    all_best_all_masked_maes = [r['best_all_masked_mae'] for r in fold_results]
    all_best_all_masked_pccs = [r['best_all_masked_pcc'] for r in fold_results]

    # calculate average performance (using best model's metrics)
    avg_best_loss = sum(all_best_losses) / len(all_best_losses)
    avg_best_mse = sum(all_best_mses) / len(all_best_mses)
    avg_best_mae = sum(all_best_maes) / len(all_best_maes)
    avg_best_pcc = sum(all_best_pccs) / len(all_best_pccs)
    avg_best_all_masked_loss = sum(all_best_all_masked_losses) / len(all_best_all_masked_losses)
    avg_best_all_masked_mse = sum(all_best_all_masked_mses) / len(all_best_all_masked_mses)
    avg_best_all_masked_mae = sum(all_best_all_masked_maes) / len(all_best_all_masked_maes)
    avg_best_all_masked_pcc = sum(all_best_all_masked_pccs) / len(all_best_all_masked_pccs)

    # calculate standard deviations
    std_best_loss = (sum((x - avg_best_loss) ** 2 for x in all_best_losses) / len(all_best_losses)) ** 0.5
    std_best_mse = (sum((x - avg_best_mse) ** 2 for x in all_best_mses) / len(all_best_mses)) ** 0.5
    std_best_mae = (sum((x - avg_best_mae) ** 2 for x in all_best_maes) / len(all_best_maes)) ** 0.5
    std_best_pcc = (sum((x - avg_best_pcc) ** 2 for x in all_best_pccs) / len(all_best_pccs)) ** 0.5
    std_best_all_masked_loss = (sum((x - avg_best_all_masked_loss) ** 2 for x in all_best_all_masked_losses) / len(all_best_all_masked_losses)) ** 0.5
    std_best_all_masked_mse = (sum((x - avg_best_all_masked_mse) ** 2 for x in all_best_all_masked_mses) / len(all_best_all_masked_mses)) ** 0.5
    std_best_all_masked_mae = (sum((x - avg_best_all_masked_mae) ** 2 for x in all_best_all_masked_maes) / len(all_best_all_masked_maes)) ** 0.5
    std_best_all_masked_pcc = (sum((x - avg_best_all_masked_pcc) ** 2 for x in all_best_all_masked_pccs) / len(all_best_all_masked_pccs)) ** 0.5

    # print fold results to terminal
    accelerator.print(f'\n{"="*60}')
    accelerator.print(f'K-FOLD CROSS-VALIDATION SUMMARY ({n_folds} folds)')
    accelerator.print(f'{"="*60}')

    accelerator.print(f'Total training time: {total_time_str}')
    accelerator.print(f'Individual fold results:')

    for result in fold_results:
        fold_time = str(datetime.timedelta(seconds=int(result['training_time'])))
        accelerator.print(f'  Fold {result["fold"]}: Best Loss = {result["best_loss"]:.5f}, '
              f'Best MSE = {result["best_mse"]:.5f}, MAE = {result["best_mae"]:.5f}, '
              f'PCC = {result["best_pcc"]:.4f}, Time = {fold_time}')
        accelerator.print(f'    All Masked: Loss = {result["best_all_masked_loss"]:.5f}, '
              f'MSE = {result["best_all_masked_mse"]:.5f}, MAE = {result["best_all_masked_mae"]:.5f}, '
              f'PCC = {result["best_all_masked_pcc"]:.4f}')

    accelerator.print(f'\nAverage Results (Best Models):')
    accelerator.print(f'  Best Loss: {avg_best_loss:.5f} ± {std_best_loss:.5f}')
    accelerator.print(f'  Best MSE: {avg_best_mse:.5f} ± {std_best_mse:.5f}')
    accelerator.print(f'  Best MAE: {avg_best_mae:.5f} ± {std_best_mae:.5f}')
    accelerator.print(f'  Best PCC: {avg_best_pcc:.4f} ± {std_best_pcc:.4f}')
    accelerator.print(f'\nAverage Results (All Masked - Best Models):')
    accelerator.print(f'  All Masked Loss: {avg_best_all_masked_loss:.5f} ± {std_best_all_masked_loss:.5f}')
    accelerator.print(f'  All Masked MSE: {avg_best_all_masked_mse:.5f} ± {std_best_all_masked_mse:.5f}')
    accelerator.print(f'  All Masked MAE: {avg_best_all_masked_mae:.5f} ± {std_best_all_masked_mae:.5f}')
    accelerator.print(f'  All Masked PCC: {avg_best_all_masked_pcc:.4f} ± {std_best_all_masked_pcc:.4f}')

    # save summary to JSON file
    summary_file = base_output_path / 'kfold_summary.json'
    try:
        summary_data = {
            'n_folds': n_folds,
            'total_time': total_time_str,
            'average_best_loss': avg_best_loss,
            'average_best_mse': avg_best_mse,
            'average_best_mae': avg_best_mae,
            'average_best_pcc': avg_best_pcc,
            'average_best_all_masked_loss': avg_best_all_masked_loss,
            'average_best_all_masked_mse': avg_best_all_masked_mse,
            'average_best_all_masked_mae': avg_best_all_masked_mae,
            'average_best_all_masked_pcc': avg_best_all_masked_pcc,
            'std_best_loss': std_best_loss,
            'std_best_mse': std_best_mse,
            'std_best_mae': std_best_mae,
            'std_best_pcc': std_best_pcc,
            'std_best_all_masked_loss': std_best_all_masked_loss,
            'std_best_all_masked_mse': std_best_all_masked_mse,
            'std_best_all_masked_mae': std_best_all_masked_mae,
            'std_best_all_masked_pcc': std_best_all_masked_pcc,
            'fold_results': fold_results,
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        accelerator.print(f'Summary saved to: {summary_file}')
    except Exception as e:
        accelerator.print(f'Warning: Failed to save summary: {e}')


def print_config_help(config: Config) -> None:
    """
    Print formatted help for configuration parameters.
    
    Displays all config sections with their fields, types, defaults,
    and current values.
    
    Args:
        config: Configuration object to display help for
    """
    section_models = {
        "general": GeneralConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "wandb": WandBConfig,
    }
    
    print("=" * 60)
    print(" Configuration Parameters Help")
    print("=" * 60)
    
    config_dict = config.model_dump()
    
    for section_name, model_class in section_models.items():
        section_config = config_dict.get(section_name)
        
        if section_config is None:
            continue
            
        if not isinstance(section_config, dict):
            continue
        
        # pydantic v2: use model_fields
        model_fields = getattr(model_class, 'model_fields', {})
        
        # filter fields with non-None values
        fields_to_show = [
            (name, info) for name, info in model_fields.items()
            if name in section_config and section_config.get(name) is not None
        ]
        
        if not fields_to_show:
            continue
        
        print(f"\n[{section_name.upper()}]")
        print("-" * 40)
        
        for field_name, field_info in fields_to_show:
            description = field_info.description or "No description"
            default_val = field_info.default
            
            # get type annotation
            if hasattr(field_info, 'annotation'):
                annotation = field_info.annotation
                origin = get_origin(annotation)
                
                if origin is Union:
                    args = get_args(annotation)
                    if len(args) == 2 and type(None) in args:
                        inner_type = args[0] if args[1] is type(None) else args[1]
                        field_type = f"Optional[{getattr(inner_type, '__name__', str(inner_type))}]"
                    else:
                        field_type = str(annotation)
                else:
                    field_type = getattr(annotation, '__name__', str(annotation))
            else:
                field_type = "Unknown"
            
            current_val = section_config.get(field_name, "Not set")
            
            print(f"  {field_name} ({field_type})")
            print(f"    Description: {description}")
            print(f"    Default: {default_val}")
            print(f"    Current: {current_val}")
            print()
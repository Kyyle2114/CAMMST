import argparse
import time
import datetime
import json
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch
import numpy as np
from tqdm import tqdm

from config.schemas import Config
from utils import misc, data
from engines.engine_train import compute_metrics


def get_args_parser() -> argparse.ArgumentParser:
    """
    Create and return argument parser for inference configuration.

    All configuration is handled through the YAML config file.
    Override config values with --set KEY=VALUE

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(add_help=False)

    # --- Config file path ---
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/default.yaml',
        help='path to YAML configuration file'
    )

    # --- Show help for config ---
    parser.add_argument(
        '--help_config',
        action='store_true',
        help='show detailed help for configuration parameters'
    )

    # --- Override config values ---
    parser.add_argument(
        '--set',
        action='append',
        nargs='+',
        metavar='KEY=VALUE',
        help='override any config value (e.g., --set general.seed=42 --set data.dataset_name=her2st)'
    )

    # --- Inference specific arguments ---
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_dir',
        help='directory containing trained models (default: output_dir)'
    )

    parser.add_argument(
        '--inference_output_dir',
        type=str,
        default='inference_results',
        help='directory to save inference results (default: inference_results)'
    )

    parser.add_argument(
        '--visible_ratio',
        type=float,
        default=0.1,
        help='fraction of spots to keep visible during inference (default: 0.1)'
    )

    return parser


def run_inference_for_fold(
    dataset_name: str,
    fold: int,
    config: Config,
    output_dir: str,
    inference_output_dir: str,
    visible_ratio: float
) -> Dict[str, Any]:
    """
    Run inference for a specific fold and dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'her2st', 'skin', 'stnet')
        fold: Fold number (0-7)
        config: Configuration object
        output_dir: Directory containing trained models
        inference_output_dir: Directory to save inference results
        visible_ratio: Fraction of spots to keep visible

    Returns:
        Dictionary containing fold inference results
    """
    print(f'\n========== Running inference for {dataset_name} Fold {fold} ==========\n')

    fold_start_time = time.time()

    # Load trained model for this fold
    try:
        model, device = misc.load_model(
            dataset=dataset_name,
            fold=fold,
            model_path=output_dir,
            config=config,
        )
        print(f"Loaded model for fold {fold} on device: {device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model for fold {fold}: {e}")

    # Create inference output directory for this fold
    fold_inference_dir = Path(inference_output_dir) / dataset_name / f'fold_{fold}'
    fold_inference_dir.mkdir(parents=True, exist_ok=True)

    # Split slides for this fold (we need validation slides for inference)
    try:
        train_slides, val_slides = data.split_slides(config, fold=fold)
        print(f"Validation slides for fold {fold}: {len(val_slides)} slides")
    except Exception as e:
        raise RuntimeError(f"Failed to split slides for fold {fold}: {e}")

    # Create data loader for validation slides
    try:
        data_config = {
            "data": {
                "dataset_path": config.data.dataset_path,
                "feature_path": config.data.feature_path,
                "dataset_name": dataset_name,
            }
        }

        # Create a simple data loader that loads all validation data
        val_dataloader = data.create_feature_dataloaders(
            train_slides=train_slides,
            val_slides=val_slides,
            config=data_config,
        )["val"]

    except Exception as e:
        raise RuntimeError(f"Failed to create data loader for fold {fold}: {e}")

    # Collect inference results for all slides in this fold
    fold_results = []

    # Process each batch in the validation set
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Fold {fold} Inference")):
            try:
                # Move batch to device
                features = batch['features'].to(device)
                gt_expressions = batch['gt_expressions'].to(device)
                coords = batch['coords'].to(device)
                slide_name = batch['slide_name']

                inference_results = model.slide_inference(
                    features=features,
                    gt_expressions=gt_expressions,
                    coords=coords,
                    visible_ratio=visible_ratio
                )

                # Extract predictions and targets from slide_inference results
                pred_expressions = inference_results['pred_expressions']  # (B*N, G) - all predictions
                pred_expressions_with_gt = inference_results['pred_expressions_with_gt']  # (B*N, G) - GT for visible, pred for masked
                gt_all = inference_results['gt_all']  # (B*N, G) - ground truth for all
                pred_expressions_masked = inference_results['pred_expressions_masked']  # (B*M, G) - predictions for masked spots
                gt_masked = inference_results['gt_masked']  # (B*M, G) - ground truth for masked spots

                # Store results for this batch
                batch_result = {
                    'batch_idx': batch_idx,
                    'slide_name': slide_name,
                    'num_spots': features.shape[1],
                    'num_genes': gt_expressions.shape[2],
                    'pred_expressions': pred_expressions.cpu().numpy(),
                    'pred_expressions_with_gt': pred_expressions_with_gt.cpu().numpy(),
                    'gt_expressions': gt_all.cpu().numpy(),
                    'pred_expressions_masked': pred_expressions_masked.cpu().numpy(),
                    'gt_masked': gt_masked.cpu().numpy(),
                    'visible_ratio': visible_ratio,
                    'inference_metadata': {
                        'p_x_shape': inference_results['p_x'].shape,
                        'vis_idx_shape': inference_results['vis_idx'].shape,
                        'mask_shape': inference_results['mask'].shape,
                    }
                }

                fold_results.append(batch_result)

            except Exception as e:
                print(f"Warning: Failed to process batch {batch_idx} in fold {fold}: {e}")
                continue

    # Calculate metrics for both prediction types
    if fold_results:
        # Collect all predictions and targets for metrics calculation
        all_pred_expressions = []
        all_pred_expressions_with_gt = []
        all_targets = []
        
        all_pred_expressions_masked = []
        all_targets_masked = []

        for result in fold_results:
            all_pred_expressions.append(torch.from_numpy(result['pred_expressions']))
            all_pred_expressions_with_gt.append(torch.from_numpy(result['pred_expressions_with_gt']))
            all_targets.append(torch.from_numpy(result['gt_expressions']))
            all_pred_expressions_masked.append(torch.from_numpy(result['pred_expressions_masked']))
            all_targets_masked.append(torch.from_numpy(result['gt_masked']))
            
        # Calculate metrics for all predictions (pred_expressions vs gt)
        metrics_all_pred = compute_metrics(all_pred_expressions, all_targets)

        # Calculate metrics for predictions with GT (pred_expressions_with_gt vs gt)
        # Note: pred_expressions_with_gt contains GT values for visible spots and predictions for masked spots
        metrics_with_gt = compute_metrics(all_pred_expressions_with_gt, all_targets)
        
        metrics_masked = compute_metrics(all_pred_expressions_masked, all_targets_masked)

        fold_metrics = {
            'metrics_all_pred': {
                'mse': metrics_all_pred['mse'],
                'mae': metrics_all_pred['mae'],
                'avg_gene_correlation': metrics_all_pred['avg_gene_correlation']
            },
            'metrics_with_gt': {
                'mse': metrics_with_gt['mse'],
                'mae': metrics_with_gt['mae'],
                'avg_gene_correlation': metrics_with_gt['avg_gene_correlation']
            },
            'metrics_masked': {
                'mse': metrics_masked['mse'],
                'mae': metrics_masked['mae'],
                'avg_gene_correlation': metrics_masked['avg_gene_correlation']
            },
            'total_spots': sum(len(result['pred_expressions']) for result in fold_results),
            'num_genes': fold_results[0]['num_genes'] if fold_results else 0,
            'num_batches': len(fold_results)
        }
    else:
        fold_metrics = {
            'metrics_all_pred': {
                'mse': 0.0,
                'mae': 0.0,
                'avg_gene_correlation': 0.0
            },
            'metrics_with_gt': {
                'mse': 0.0,
                'mae': 0.0,
                'avg_gene_correlation': 0.0
            },
            'metrics_masked': {
                'mse': 0.0,
                'mae': 0.0,
                'avg_gene_correlation': 0.0
            },
            'total_spots': 0,
            'num_genes': 0,
            'num_batches': 0
        }

    # Save fold results
    results_file = fold_inference_dir / 'inference_results.json'
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'fold': fold,
                'dataset': dataset_name,
                'metrics': fold_metrics,
                'inference_time': time.time() - fold_start_time
            }, f, indent=2)

        print(f"Saved inference results for fold {fold} to {results_file}")

    except Exception as e:
        print(f"Warning: Failed to save results for fold {fold}: {e}")

    inference_time = time.time() - fold_start_time
    print(f"Fold {fold} inference completed in {inference_time:.2f} seconds")

    return {
        'fold': fold,
        'dataset': dataset_name,
        'metrics': fold_metrics,
        'num_batches': len(fold_results),
        'inference_time': inference_time
    }


def main(args: argparse.Namespace) -> None:
    """
    Main function for model inference.

    Args:
        args: Parsed command line arguments
    """
    # Load configuration
    try:
        config: Config = misc.load_config(args.config, args)

        # Print config help and exit
        if args.help_config:
            misc.print_config_help(config)
            exit(0)

    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Set up output directories
    output_dir = Path(args.output_dir)
    inference_output_dir = Path(args.inference_output_dir)
    inference_output_dir.mkdir(parents=True, exist_ok=True)

    visible_ratio = args.visible_ratio

    print("=== CAMMST Inference ===")
    print(f"Config: {args.config}")
    print(f"Output dir: {output_dir}")
    print(f"Inference output dir: {inference_output_dir}")
    print(f"Visible ratio: {visible_ratio}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Get dataset name from config
    dataset_name = config.data.dataset_path.split('/')[-1]
    n_folds = config.data.folds

    print(f"Dataset: {dataset_name}")
    print(f"Number of folds: {n_folds}")

    # Overall timing
    overall_start_time = time.time()

    # Run inference for each fold
    fold_results = []

    for fold in range(n_folds):
        try:
            fold_result = run_inference_for_fold(
                dataset_name=dataset_name,
                fold=fold,
                config=config,
                output_dir=str(output_dir),
                inference_output_dir=str(inference_output_dir),
                visible_ratio=visible_ratio
            )
            fold_results.append(fold_result)

        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            continue

    # Calculate overall statistics
    total_time = time.time() - overall_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if fold_results:
        # Aggregate metrics across folds for both prediction types
        avg_mse_all_pred = np.mean([r['metrics']['metrics_all_pred']['mse'] for r in fold_results])
        avg_mae_all_pred = np.mean([r['metrics']['metrics_all_pred']['mae'] for r in fold_results])
        avg_pcc_all_pred = np.mean([r['metrics']['metrics_all_pred']['avg_gene_correlation'] for r in fold_results])

        avg_mse_with_gt = np.mean([r['metrics']['metrics_with_gt']['mse'] for r in fold_results])
        avg_mae_with_gt = np.mean([r['metrics']['metrics_with_gt']['mae'] for r in fold_results])
        avg_pcc_with_gt = np.mean([r['metrics']['metrics_with_gt']['avg_gene_correlation'] for r in fold_results])
        
        avg_mse_masked = np.mean([r['metrics']['metrics_masked']['mse'] for r in fold_results])
        avg_mae_masked = np.mean([r['metrics']['metrics_masked']['mae'] for r in fold_results])
        avg_pcc_masked = np.mean([r['metrics']['metrics_masked']['avg_gene_correlation'] for r in fold_results])

        summary = {
            'dataset': dataset_name,
            'total_folds': len(fold_results),
            'visible_ratio': visible_ratio,
            'average_metrics_all_pred': {
                'mse': float(avg_mse_all_pred),
                'mae': float(avg_mae_all_pred),
                'avg_gene_correlation': float(avg_pcc_all_pred)
            },
            'average_metrics_with_gt': {
                'mse': float(avg_mse_with_gt),
                'mae': float(avg_mae_with_gt),
                'avg_gene_correlation': float(avg_pcc_with_gt)
            },
            'average_metrics_masked': {
                'mse': float(avg_mse_masked),
                'mae': float(avg_mae_masked),
                'avg_gene_correlation': float(avg_pcc_masked)
            },
            'fold_results': fold_results,
            'total_inference_time': total_time_str
        }

        # Save overall summary
        summary_file = inference_output_dir / dataset_name / 'inference_summary.json'
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n=== Inference Summary ===")
        print(f"Dataset: {dataset_name}")
        print(f"Folds completed: {len(fold_results)}/{n_folds}")
        print(f"Visible ratio: {visible_ratio}")
        print("\n--- All Predictions (pred_expressions) ---")
        print(f"MSE: {avg_mse_all_pred:.4f}")
        print(f"MAE: {avg_mae_all_pred:.4f}")
        print(f"PCC: {avg_pcc_all_pred:.4f}")
        print("\n--- Predictions with GT (pred_expressions_with_gt) ---")
        print(f"MSE: {avg_mse_with_gt:.4f}")
        print(f"MAE: {avg_mae_with_gt:.4f}")
        print(f"PCC: {avg_pcc_with_gt:.4f}")
        print("\n--- Predictions for masked spots (pred_expressions_masked) ---")
        print(f"MSE: {avg_mse_masked:.4f}")
        print(f"MAE: {avg_mae_masked:.4f}")
        print(f"PCC: {avg_pcc_masked:.4f}")
        print(f"\nTotal time: {total_time_str}")
        print(f"Results saved to: {inference_output_dir / dataset_name}")

    else:
        print("No folds completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAMMST-Inference', parents=[get_args_parser()])
    args = parser.parse_args()

    try:
        main(args)
        print('\n=== Inference Complete ===\n')

    except Exception as e:
        print(f'\n=== Inference Failed ===\n')
        print(f'Error: {e}\n')
        exit(1)
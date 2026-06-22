import math
import argparse
import torchinfo
import time
import datetime
import json
import yaml
from pathlib import Path
from accelerate import Accelerator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import torch

from config.schemas import Config
from utils import misc, lr_sched, data
from engines import engine_train
from models import CAMMST


def get_args_parser() -> argparse.ArgumentParser:
    """
    Create and return argument parser for training configuration.
    
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
        help='override any config value (e.g., --set general.seed=42 --set training.lr=0.001)'
    )
    
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Main function for model training.

    Args:
        args: Parsed command line arguments containing:
            - config: Path to YAML configuration file
            - set: Optional config overrides in KEY=VALUE format
            - help_config: Show config help and exit
    """
    # --- Load configuration ---
    try:
        config: Config = misc.load_config(args.config, args)
        
        # print config help and exit
        if args.help_config:
            misc.print_config_help(config)
            exit(0)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")
    
    # --- Seed & Output setup ---
    misc.seed_everything(config.general.seed)

    base_output_path = Path(config.general.output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # get number of folds for cross-validation
    n_folds = config.data.folds

    # overall timing
    overall_start_time = time.time()

    # initialize results storage for k-fold cross-validation
    fold_results = []

    # perform k-fold cross-validation
    for fold in range(n_folds):
        fold_start_time = time.time()

        # create fold-specific output directory
        fold_output_path = base_output_path / f'fold_{fold}'
        fold_output_path.mkdir(parents=True, exist_ok=True)

        # --- Accelerate setting ---
        try:
            accelerator = Accelerator(
                gradient_accumulation_steps=1,
                mixed_precision=config.training.mixed_precision,
                log_with='wandb',
                project_dir=str(fold_output_path)
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Accelerator for fold {fold}: {e}\n")

        accelerator.print(f'\n========== Fold {fold + 1}/{n_folds} ==========\n')
        accelerator.print(f'config: {args.config}')

        # wandb initialization through accelerator (fold-specific)
        try:
            if config.wandb.project_name is not None:
                run_name = f"{config.wandb.run_name}_fold_{fold}"
                accelerator.init_trackers(
                    project_name=config.wandb.project_name,
                    config=config.model_dump(),
                    init_kwargs={"wandb": {"name": run_name}}
                )
            else:
                # no wandb logging
                accelerator.print("Continuing without WandB logging...")

        except Exception as e:
            accelerator.print(f"Warning: Failed to initialize WandB tracking for fold {fold}: {e}")
            accelerator.print("Continuing without WandB logging...")

        # save config as YAML file for this fold
        config_file = fold_output_path / 'config.yaml'
        try:
            with open(config_file, mode="w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, indent=4, sort_keys=False)

        except IOError as e:
            accelerator.print(f"Warning: Failed to save config for fold {fold}: {e}\n")

        # --- Dataset & Dataloader ---
        try:
            # split slides for current fold
            train_slides, val_slides = data.split_slides(config, fold=fold)

            # create data loaders for UNI features and gene expressions
            data_config = {
                "data": {
                    "dataset_path": config.data.dataset_path,
                    "feature_path": config.data.feature_path,
                    "dataset_name": config.data.dataset_path.split('/')[-1],  # e.g., "stnet"
                }
            }

            dataloaders = data.create_feature_dataloaders(
                train_slides=train_slides,
                val_slides=val_slides,
                config=data_config,
                num_workers=config.data.num_workers
            )

            train_loader = dataloaders["train"]
            val_loader = dataloaders["val"]

            # get dataset sizes for logging
            train_size = len(train_slides)
            val_size = len(val_slides)
            accelerator.print(f'Train slides: {train_size}, Val slides: {val_size}')

        except Exception as e:
            raise RuntimeError(f"Failed to create datasets for fold {fold}: {e}\n")

        # --- Model config ---
        try:
            model = CAMMST(**config.model.model_dump())
            n_parameters = sum(p.numel() for p in model.parameters())

        except Exception as e:
            raise RuntimeError(f"Failed to create model for fold {fold}: {e}\n")

        # print model info (only for first fold)
        if fold == 0 and accelerator.is_main_process:
            accelerator.print()
            accelerator.print('=== MODEL INFO ===')
            torchinfo.summary(model)
            accelerator.print()

        # --- Training config ---
        eff_batch_size = config.data.batch_size
        abs_lr = config.training.lr     

        # following timm: set wd as 0 for bias and norm layers
        param_groups = misc.add_weight_decay(model, config.training.weight_decay)

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=0.0  # small lr for warm-up
        )

        # calculate total number of steps for the scheduler
        num_update_steps_per_epoch = math.ceil(len(train_loader) / config.training.grad_accum_steps)
        num_training_steps = config.training.epoch * num_update_steps_per_epoch
        num_warmup_steps = config.training.warmup_epochs * num_update_steps_per_epoch

        try:
            # use per-step lr scheduler
            scheduler = lr_sched.CosineAnnealingWarmUpRestarts(
                optimizer,
                T_0=num_training_steps,
                T_mult=1,
                eta_max=abs_lr,
                T_up=num_warmup_steps,
                gamma=1.0
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create learning rate scheduler for fold {fold}: {e}\n")

        # --- Prepare everything with accelerator ---
        try:
            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )

        except Exception as e:
            raise RuntimeError(f"Failed to prepare components with Accelerator for fold {fold}: {e}\n")

        # --- WandB logging for this fold ---
        try:
            # log additional configurations to wandb through accelerator
            accelerator.log({
                f'config/effective_batch_size': eff_batch_size,
                f'config/mixed_precision': config.training.mixed_precision,
                f'config/num_parameters': n_parameters
            }, step=0)

        except Exception as e:
            accelerator.print(f"Warning: Failed to log to WandB for fold {fold}: {e}\n")

        # initialize training variables for this fold
        min_loss = float('inf')
        best_epoch = 0
        best_eval_stats = {'loss': float('inf')}  # Best model's evaluation metrics
        best_all_mask_eval_stats = {'loss': float('inf')}  # Best model's all masked evaluation metrics
        eval_stats = {'loss': float('inf')}

        # early stopping: lower is better ('min' mode)
        es = misc.DistributedEarlyStopping(
            patience=config.training.patience,
            delta=0.0,
            mode='min',
            verbose=True
        )

        metrics_tracker = misc.MetricTracker()
        all_mask_eval_stats = None  # store all masked evaluation statistics (no gene information)

        # training loop
        for epoch in range(config.training.epoch):
            train_stats = engine_train.train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                epoch=epoch,
                clip_grad=config.training.clip_grad,
                metrics_tracker=metrics_tracker,
                config=config
            )

            # model evaluation (validation set)
            eval_stats = engine_train.evaluate(
                model=model,
                data_loader=val_loader,
                accelerator=accelerator,
                metrics_tracker=metrics_tracker,
                config=config
            )

            # update best validation metrics
            val_loss = eval_stats['loss']
            accelerator.print(f"[INFO] Validation loss: {val_loss:.5f}")
            accelerator.print(f"[INFO] Best validation loss so far: {min_loss:.5f}")

            # save best model based on validation loss
            if val_loss < min_loss:
                accelerator.print(f'[INFO] Validation loss improved from {min_loss:.5f} to {val_loss:.5f}. Saving best model.')
                min_loss = val_loss
                best_epoch = epoch
                best_eval_stats = eval_stats.copy()  # save best model's evaluation metrics

                try:
                    save_dir = fold_output_path / "best_model"
                    accelerator.save_model(model, save_dir)
                    accelerator.print(f"[INFO] Best model for fold {fold} saved to {save_dir}")

                except Exception as e:
                    accelerator.print(f"Warning: Failed to save best model for fold {fold}: {e}")
                
                # model evaluation (validation set) with all spots masked
                all_mask_eval_stats = engine_train.evaluate_with_all_mask(
                    model=model,
                    data_loader=val_loader,
                    accelerator=accelerator,
                    metrics_tracker=metrics_tracker,
                    config=config
                )
                
                best_all_mask_eval_stats = all_mask_eval_stats.copy()  # save best model's all masked evaluation metrics

            # stats logging to file
            if config.general.output_dir:
                log_stats = {
                    **{f'train_{k}': f'{v:.6f}' for k, v in train_stats.items()},
                    **{f'test_{k}': f'{v:.6f}' for k, v in eval_stats.items()},
                    **{f'test_all_masked_{k}': f'{v:.6f}' for k, v in (all_mask_eval_stats.items() if all_mask_eval_stats is not None else {})},
                    'epoch': epoch,
                    'fold': fold,
                    'n_parameters': n_parameters
                }

                try:
                    log_file_path = fold_output_path / "log.txt"
                    with open(log_file_path, mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

                except IOError as e:
                    accelerator.print(f"Warning: Failed to write log file for fold {fold}: {e}\n")

            # wandb logging
            try:
                accelerator.log(
                    {
                        f'train/loss': train_stats['loss'],
                        f'train/recon_loss': train_stats.get('recon_loss', 0.0),
                        f'train/pcc_loss': train_stats.get('pcc_loss', 0.0),
                        f'train/sample_loss': train_stats.get('sample_loss', 0.0),
                        f'train/contrast_loss': train_stats.get('contrast_loss', 0.0),
                        f'train/learning_rate': train_stats['lr'],
                        f'eval/loss': eval_stats['loss'],
                        f'eval/recon_loss': eval_stats.get('recon_loss', 0.0),
                        f'eval/pcc_loss': eval_stats.get('pcc_loss', 0.0),
                        f'eval/sample_loss': eval_stats.get('sample_loss', 0.0),
                        f'eval/contrast_loss': eval_stats.get('contrast_loss', 0.0),
                        f'eval/mse': eval_stats.get('mse', 0.0),
                        f'eval/mae': eval_stats.get('mae', 0.0),
                        f'eval/pcc': eval_stats.get('avg_gene_correlation', 0.0),
                        f'epoch': epoch
                    }, step=epoch
                )

                if all_mask_eval_stats is not None:
                    accelerator.log(
                        {
                            f'eval_all_masked/loss': all_mask_eval_stats['loss'],
                            f'eval_all_masked/recon_loss': all_mask_eval_stats.get('recon_loss', 0.0),
                            f'eval_all_masked/pcc_loss': all_mask_eval_stats.get('pcc_loss', 0.0),
                            f'eval_all_masked/mse': all_mask_eval_stats.get('mse', 0.0),
                            f'eval_all_masked/mae': all_mask_eval_stats.get('mae', 0.0),
                            f'eval_all_masked/pcc': all_mask_eval_stats.get('avg_gene_correlation', 0.0),
                        }, step=epoch
                    )
            except Exception as e:
                accelerator.print(f"Warning: Failed to log to WandB for fold {fold}: {e}\n")

            # check early stopping
            should_stop = es(eval_stats['loss'], accelerator)

            if should_stop:
                accelerator.print(f"[INFO] Early stopping triggered at epoch {epoch}\n")
                break
                
            # reset all masked evaluation statistics
            all_mask_eval_stats = None

        # store fold results (using best model's performance)
        fold_result = {
            'fold': fold,
            'best_loss': min_loss,
            'best_epoch': best_epoch,
            'best_mse': best_eval_stats.get("mse", 0.0),
            'best_mae': best_eval_stats.get("mae", 0.0),
            'best_pcc': best_eval_stats.get("avg_gene_correlation", 0.0),
            'best_all_masked_loss': best_all_mask_eval_stats.get("loss", float('inf')),
            'best_all_masked_mse': best_all_mask_eval_stats.get("mse", 0.0),
            'best_all_masked_mae': best_all_mask_eval_stats.get("mae", 0.0),
            'best_all_masked_pcc': best_all_mask_eval_stats.get("avg_gene_correlation", 0.0),
            'final_loss': eval_stats["loss"],
            'training_time': time.time() - fold_start_time
        }
        fold_results.append(fold_result)

        # save fold result to JSON file for this fold
        fold_result_file = fold_output_path / 'fold_result.json'
        try:
            with open(fold_result_file, 'w') as f:
                json.dump(fold_result, f, indent=2)
        except Exception as e:
            accelerator.print(f'Warning: Failed to save fold {fold} result: {e}')

        accelerator.print(f'[INFO] Fold {fold+1} / {n_folds} completed\n')
        
        # end training for this fold
        if accelerator.is_main_process:
            accelerator.end_training()

    # --- K-fold Cross-validation Summary ---
    total_time = time.time() - overall_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # print and save k-fold cross-validation summary
    misc.print_kfold_summary(
        fold_results=fold_results,
        n_folds=n_folds,
        total_time_str=total_time_str,
        base_output_path=base_output_path,
        accelerator=accelerator
    )
    
    # end training
    if accelerator.is_main_process:
        accelerator.end_training()
    

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser('CAMMST-Training', parents=[get_args_parser()])
    args = parser.parse_args() 
    
    try:
        main(args)
        print('\n=== Training Complete ===\n')

    except Exception as e:
        print(f'\n=== Training Failed ===\n')
        print(f'Error: {e}\n')
        exit(1)
import sys
import math
from typing import Iterable, Dict, Optional, List
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler

import torch
import torch.nn.functional as F
import numpy as np

from utils.misc import MetricTracker


def compute_metrics(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics (MSE, MAE, PCC) for gene expression prediction.

    Args:
        predictions: List of prediction tensors, each of shape (M, G) where M is masked spots, G is genes
        targets: List of target tensors, each of shape (M, G) where M is masked spots, G is genes

        Returns:
        Dictionary containing computed metrics:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - avg_gene_correlation: Average Pearson correlation coefficient across genes
    """
    if not predictions:
        return {
            'mse': 0.0,
            'mae': 0.0,
            'avg_gene_correlation': 0.0
        }

    # concatenate all predictions and targets
    preds = torch.cat(predictions, dim=0)  # (Total_masked_spots, G)
    gts = torch.cat(targets, dim=0)       # (Total_masked_spots, G)

    # basic metrics
    mse = F.mse_loss(preds, gts).item()
    mae = F.l1_loss(preds, gts).item()

    # convert to numpy for correlation calculation
    preds_np = preds.cpu().numpy()
    gts_np = gts.cpu().numpy()

    # gene-wise correlation (correlation for each gene across all spots)
    num_samples, num_genes = preds_np.shape

    # vectorized NaN/inf handling
    preds_clean = np.nan_to_num(preds_np, nan=0.0, posinf=0.0, neginf=0.0)
    gts_clean = np.nan_to_num(gts_np, nan=0.0, posinf=0.0, neginf=0.0)

    # center the data for all genes at once
    # preds_clean: (N, G), preds_mean: (1, G) -> preds_centered: (N, G)
    preds_mean = preds_clean.mean(axis=0, keepdims=True)  # (1, G)
    gts_mean = gts_clean.mean(axis=0, keepdims=True)      # (1, G)

    preds_centered = preds_clean - preds_mean  # (N, G)
    gts_centered = gts_clean - gts_mean        # (N, G)

    # compute covariance and variances for all genes simultaneously
    covariance = np.sum(preds_centered * gts_centered, axis=0)  # (G,)
    pred_var = np.sum(preds_centered ** 2, axis=0)              # (G,)
    gt_var = np.sum(gts_centered ** 2, axis=0)                  # (G,)

    # compute correlations: handle division by zero
    denominator = np.sqrt(pred_var * gt_var)
    correlations = np.divide(
        covariance, denominator,
        out=np.zeros_like(covariance),
        where=denominator != 0.0
    )

    # average correlation across all genes
    avg_corr = float(np.mean(correlations))

    return {
        'mse': mse,
        'mae': mae,
        'avg_gene_correlation': avg_corr
    }


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: AcceleratedOptimizer,
    scheduler: AcceleratedScheduler,
    accelerator: Accelerator,
    epoch: int,
    clip_grad: float | None = None,
    metrics_tracker: Optional[MetricTracker] = None,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Train the CAMMST model for one epoch using Accelerate.

    Args:
        model: CAMMST model to train (must have compute_loss method)
        data_loader: PyTorch DataLoader yielding batches with 'features', 'gt_expressions', 'coords'
        optimizer: An `AcceleratedOptimizer` instance from `accelerator.prepare()`.
        scheduler: An `AcceleratedScheduler` instance from `accelerator.prepare()`.
        accelerator: Accelerator object for distributed training
        epoch: Current epoch number
        clip_grad: Gradient clipping norm (None for no clipping)
        metrics_tracker: Optional reusable MetricTracker instance
        config: Optional config object for loss weights
    
    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average total training loss across all processes
        - recon_loss: Average reconstruction loss
        - sample_loss: Average sampling loss
        - contrast_loss: Average contrastive loss
        - lr: Current learning rate

    Raises:
        RuntimeError: If loss becomes infinite or NaN
        ValueError: If invalid gradient clipping value is provided
        TypeError: If arguments have wrong types
    """
    try:
        model.train()
        
        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()
        
        progress_bar = tqdm(
            data_loader, 
            disable=not accelerator.is_main_process or not sys.stdout.isatty(),
            desc=f"Epoch {epoch}",
            dynamic_ncols=True
        )
            
        for batch in progress_bar:
            # input format: features, gt_expressions, coords, bio_salience_score
            features = batch['features']
            gt_expressions = batch['gt_expressions']
            coords = batch['coords']
            bio_salience_score = batch.get('local_bio_salience_score')  
            batch_size = features.size(0)

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # forward pass
                outputs = model(features, gt_expressions, coords, bio_salience_score=bio_salience_score)

                # loss computation
                losses = model.compute_loss(outputs, **config.loss.model_dump())
                loss = losses['total_loss']

                # update metrics for all loss components
                metrics_tracker.update({
                    'loss': loss,
                    'recon_loss': losses['recon_loss'],
                    'pcc_loss': losses['pcc_loss'],
                    'sample_loss': losses['sample_loss'],
                    'contrast_loss': losses['contrast_loss']
                }, batch_size=batch_size)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    error_msg = f"Loss is {loss_value}, stopping training"
                    accelerator.print(error_msg)
                    raise RuntimeError(error_msg)

                # backward pass with accelerator
                accelerator.backward(loss)
                
                # gradient clipping
                if clip_grad is not None:
                    if clip_grad <= 0:
                        raise ValueError(f"clip_grad must be positive, got {clip_grad}")
                    accelerator.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                scheduler.step()
                
            # progress bar shows current local averages
            lr = optimizer.param_groups[0]["lr"]
            current_averages = metrics_tracker.get_current_averages()
            progress_bar.set_postfix({
                'loss': f"{current_averages.get('loss', 0.0):.4f}",
                'recon': f"{current_averages.get('recon_loss', 0.0):.4f}",
                'pcc': f"{current_averages.get('pcc_loss', 0.0):.4f}",
                'sample': f"{current_averages.get('sample_loss', 0.0):.4f}",
                'contrast': f"{current_averages.get('contrast_loss', 0.0):.4f}",
                'lr': f"{lr:.6f}"
            })
        
        # clean up progress bar
        progress_bar.close()

        # compute global averages only once at epoch end
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)
        avg_stats['lr'] = optimizer.param_groups[0]["lr"]

        loss = avg_stats.get('loss', 0.0)
        recon_loss = avg_stats.get('recon_loss', 0.0)
        pcc_loss = avg_stats.get('pcc_loss', 0.0)
        sample_loss = avg_stats.get('sample_loss', 0.0)
        contrast_loss = avg_stats.get('contrast_loss', 0.0)
        lr = avg_stats['lr']
        accelerator.print(
            f"Training Epoch {epoch} - "
            f"Loss: {loss:.4f}, "
            f"Recon: {recon_loss:.4f}, "
            f"PCC: {pcc_loss:.4f}, "
            f"Sample: {sample_loss:.4f}, "
            f"Contrast: {contrast_loss:.4f}, "
            f"Learning Rate: {lr:.6f} \n"
        )
        
        return avg_stats
        
    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()
        raise RuntimeError(f"Training failed: {e}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    accelerator: Accelerator,
    metrics_tracker: Optional[MetricTracker] = None,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluate the CAMMST model using Accelerate.

    Args:
        model: CAMMST model to evaluate (must have compute_loss method)
        data_loader: PyTorch DataLoader yielding batches with 'features', 'gt_expressions', 'coords'
        accelerator: Accelerator object for distributed training
        metrics_tracker: Optional reusable MetricTracker instance
        config: Optional config object for loss weights
    
    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average total validation loss across all processes
        - recon_loss: Average reconstruction loss
        - pcc_loss: Average PCC loss
        - sample_loss: Average sampling loss
        - contrast_loss: Average contrastive loss
        - mse: Mean Squared Error for gene expression prediction
        - mae: Mean Absolute Error for gene expression prediction
        - avg_gene_correlation: Average gene-wise Pearson correlation

    Raises:
        RuntimeError: If evaluation encounters unexpected errors
        TypeError: If arguments have wrong types
    """
    try:
        model.eval()

        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()

        # lists to collect predictions and targets for metrics calculation
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        progress_bar = tqdm(
            data_loader,
            disable=not accelerator.is_main_process or not sys.stdout.isatty(),
            desc="Evaluating",
            dynamic_ncols=True
        )

        for batch in progress_bar:
            # input format: features, gt_expressions, coords, bio_salience_score
            features = batch['features']
            gt_expressions = batch['gt_expressions']
            coords = batch['coords']
            bio_salience_score = batch.get('local_bio_salience_score')  
            batch_size = features.size(0)

            # forward pass (no gradients needed for evaluation)
            with torch.no_grad():
                # set visible ratio to 0.1 for evaluation
                outputs = model(features, gt_expressions, coords, visible_ratio=0.1, bio_salience_score=bio_salience_score)
                losses = model.compute_loss(outputs, **config.loss.model_dump())

                # collect predictions and targets for metrics calculation
                pred_expressions = outputs['pred_expressions']  # (B, M, G) - masked predictions
                gt_masked = outputs['gt_masked']             # (B, M, G) - masked ground truth

                all_predictions.append(pred_expressions.detach().cpu())
                all_targets.append(gt_masked.detach().cpu())

            # update metrics tracker with all loss components
            metrics_tracker.update({
                'loss': losses['total_loss'],
                'recon_loss': losses['recon_loss'],
                'pcc_loss': losses['pcc_loss'],
                'sample_loss': losses['sample_loss'],
                'contrast_loss': losses['contrast_loss'],
            }, batch_size=batch_size)

            # show current running averages
            current_averages = metrics_tracker.get_current_averages()
            progress_bar.set_postfix({
                'loss': f"{current_averages.get('loss', 0.0):.4f}",
                'recon': f"{current_averages.get('recon_loss', 0.0):.4f}",
                'pcc': f"{current_averages.get('pcc_loss', 0.0):.4f}",
                'sample': f"{current_averages.get('sample_loss', 0.0):.4f}",
                'contrast': f"{current_averages.get('contrast_loss', 0.0):.4f}",
            })

        # clean up progress bar
        progress_bar.close()

        # compute global averages using MetricTracker
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)

        # compute evaluation metrics (MSE, MAE, PCC)
        eval_metrics = compute_metrics(all_predictions, all_targets)

        # combine loss metrics and evaluation metrics
        result_metrics = {**avg_stats, **eval_metrics}

        loss = avg_stats.get("loss", 0.0)
        recon_loss = avg_stats.get("recon_loss", 0.0)
        pcc_loss = avg_stats.get("pcc_loss", 0.0)
        sample_loss = avg_stats.get("sample_loss", 0.0)
        contrast_loss = avg_stats.get("contrast_loss", 0.0)
        mse = eval_metrics.get("mse", 0.0)
        mae = eval_metrics.get("mae", 0.0)
        corr = eval_metrics.get("avg_gene_correlation", 0.0)
    
        accelerator.print(
            f"Evaluation - "
            f"Loss: {loss:.4f}, "
            f"Recon: {recon_loss:.4f}, "
            f"PCC: {pcc_loss:.4f}, "
            f"Sample: {sample_loss:.4f}, "
            f"Contrast: {contrast_loss:.4f}, "
            f"MSE: {mse:.4f}, "
            f"MAE: {mae:.4f}, "
            f"Gene-PCC: {corr:.4f} \n"
        )

        return result_metrics

    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()

        error_msg = f"Evaluation failed: {e}"
        accelerator.print(error_msg)
        raise RuntimeError(error_msg)


@torch.no_grad()
def evaluate_with_all_mask(
    model: torch.nn.Module,
    data_loader: Iterable,
    accelerator: Accelerator,
    metrics_tracker: Optional[MetricTracker] = None,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Evaluate the CAMMST model using infer_with_all_mask (all spots masked).

    This function uses infer_with_all_mask method which treats all spots as masked,
    without using any visible gene information.

    Args:
        model: CAMMST model to evaluate (must have infer_with_all_mask method)
        data_loader: PyTorch DataLoader yielding batches with 'features', 'gt_expressions', 'coords'
        accelerator: Accelerator object for distributed training
        metrics_tracker: Optional reusable MetricTracker instance
        config: Optional config object for loss weights

    Returns:
        Dictionary containing the global average for each metric:
        - loss: Average total validation loss across all processes
        - recon_loss: Average reconstruction loss
        - pcc_loss: Average PCC loss
        - mse: Mean Squared Error for gene expression prediction
        - mae: Mean Absolute Error for gene expression prediction
        - avg_gene_correlation: Average gene-wise Pearson correlation

    Raises:
        RuntimeError: If evaluation encounters unexpected errors
        TypeError: If arguments have wrong types
    """
    try:
        model.eval()

        if metrics_tracker is None:
            metrics_tracker = MetricTracker()
        else:
            metrics_tracker.reset()

        # lists to collect predictions and targets for metrics calculation
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        progress_bar = tqdm(
            data_loader,
            disable=not accelerator.is_main_process or not sys.stdout.isatty(),
            desc="Evaluating (all masked)",
            dynamic_ncols=True
        )

        for batch in progress_bar:
            # input format: features, gt_expressions, coords
            features = batch['features']
            gt_expressions = batch['gt_expressions']
            coords = batch['coords']
            batch_size = features.size(0)

            # inference with all spots masked (no gradients needed)
            with torch.no_grad():
                outputs = model.infer_with_all_mask(features, gt_expressions, coords)
                losses = model.compute_loss(
                    outputs,
                    recon_weight=config.loss.get('recon_weight', 1.0),
                    pcc_weight=config.loss.get('pcc_weight', 0.5),
                    sample_weight=0.0,  # No sampling loss when all masked
                    contrast_weight=0.0,  # No contrastive loss when all masked
                    contrast_temp=config.loss.get('contrast_temp', 1.0),
                    contrastive_type=config.loss.get('contrastive_type', 'soft'),
                )

                # collect predictions and targets for metrics calculation
                # note: pred_expressions and gt_masked are for ALL spots (B, N, G)
                pred_expressions = outputs['pred_expressions']  # (B, N, G) - all spots predictions
                gt_masked = outputs['gt_masked']             # (B, N, G) - all spots ground truth

                all_predictions.append(pred_expressions.detach().cpu())
                all_targets.append(gt_masked.detach().cpu())

            # update metrics tracker with loss components (excluding sample and contrast)
            metrics_tracker.update({
                'loss': losses['total_loss'],
                'recon_loss': losses['recon_loss'],
                'pcc_loss': losses['pcc_loss'],
            }, batch_size=batch_size)

            # show current running averages
            current_averages = metrics_tracker.get_current_averages()
            progress_bar.set_postfix({
                'loss': f"{current_averages.get('loss', 0.0):.4f}",
                'recon': f"{current_averages.get('recon_loss', 0.0):.4f}",
                'pcc': f"{current_averages.get('pcc_loss', 0.0):.4f}",
            })

        # clean up progress bar
        progress_bar.close()

        # compute global averages using MetricTracker
        avg_stats = metrics_tracker.compute_epoch_averages(accelerator)

        # compute evaluation metrics (MSE, MAE, PCC)
        eval_metrics = compute_metrics(all_predictions, all_targets)

        # combine loss metrics and evaluation metrics
        result_metrics = {**avg_stats, **eval_metrics}

        loss = avg_stats.get("loss", 0.0)
        recon_loss = avg_stats.get("recon_loss", 0.0)
        pcc_loss = avg_stats.get("pcc_loss", 0.0)
        mse = eval_metrics.get("mse", 0.0)
        mae = eval_metrics.get("mae", 0.0)
        corr = eval_metrics.get("avg_gene_correlation", 0.0)
    
        accelerator.print(
            f"Evaluation (All Masked) - "
            f"Loss: {loss:.4f}, "
            f"Recon: {recon_loss:.4f}, "
            f"PCC: {pcc_loss:.4f}, "
            f"MSE: {mse:.4f}, "
            f"MAE: {mae:.4f}, "
            f"Gene-PCC: {corr:.4f} \n"
        )

        return result_metrics

    except Exception as e:
        # clean up resources on error
        if 'progress_bar' in locals() and accelerator.is_main_process:
            progress_bar.close()

        error_msg = f"Evaluation (all masked) failed: {e}"
        accelerator.print(error_msg)
        raise RuntimeError(error_msg)

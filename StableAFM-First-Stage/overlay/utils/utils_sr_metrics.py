import torch
import torch.nn.functional as F


def local_neighbor_contrast(image):
    """Return each pixel minus the mean of its eight immediate neighbors."""
    squeeze_batch = image.ndim == 3
    if squeeze_batch:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError('Expected a CHW or NCHW tensor.')

    padded = F.pad(image, (1, 1, 1, 1), mode='reflect')
    neighborhood_sum = F.avg_pool2d(padded, kernel_size=3, stride=1) * 9.0
    contrast = image - (neighborhood_sum - image) / 8.0
    return contrast.squeeze(0) if squeeze_batch else contrast


def phase_impulse_metrics(estimate, target, scale):
    """Measure local-contrast error and phase bias at known sample locations."""
    estimate_contrast = local_neighbor_contrast(estimate)
    target_contrast = local_neighbor_contrast(target)
    sample_estimate = estimate_contrast[..., 0::scale, 0::scale]
    sample_target = target_contrast[..., 0::scale, 0::scale]

    phase_mask = torch.zeros_like(estimate_contrast, dtype=torch.bool)
    phase_mask[..., 0::scale, 0::scale] = True
    other_estimate = estimate_contrast.masked_select(~phase_mask)
    other_target = target_contrast.masked_select(~phase_mask)

    estimate_ratio = (
        sample_estimate.abs().mean() / other_estimate.abs().mean().clamp_min(1e-12)
    )
    target_ratio = (
        sample_target.abs().mean() / other_target.abs().mean().clamp_min(1e-12)
    )
    return {
        'phase_contrast_mae': F.l1_loss(sample_estimate, sample_target).item(),
        'phase_ratio': estimate_ratio.item(),
        'target_phase_ratio': target_ratio.item(),
        'phase_ratio_error': torch.abs(estimate_ratio - target_ratio).item(),
    }


def hard_project_decimation(estimate, low_quality, scale):
    projected = estimate.clone()
    projected[..., 0::scale, 0::scale] = low_quality
    return projected


def decimation_consistency_mae(estimate, low_quality, scale):
    sampled = estimate[..., 0::scale, 0::scale]
    return torch.mean(torch.abs(sampled - low_quality)).item()


def residual_boundary_excess(estimate, target, periods=(4, 8, 16, 32)):
    """Measure residual-gradient discontinuity at nested periodic boundaries.

    For period p, boundaries at coordinates 0 mod p are compared with the
    half-period controls. For p > 4 this isolates extra p-period structure
    while controlling for the lower-period boundaries shared by both groups.
    """
    residual = estimate - target
    dx = torch.abs(residual[..., 1:] - residual[..., :-1])
    dy = torch.abs(residual[..., 1:, :] - residual[..., :-1, :])
    x_coords = torch.arange(1, residual.shape[-1], device=residual.device)
    y_coords = torch.arange(1, residual.shape[-2], device=residual.device)
    metrics = {}

    for period in periods:
        x_boundary = x_coords.remainder(period).eq(0)
        x_control = x_coords.remainder(period).eq(period // 2)
        y_boundary = y_coords.remainder(period).eq(0)
        y_control = y_coords.remainder(period).eq(period // 2)

        boundary_sum = dx[..., x_boundary].sum() + dy[..., y_boundary, :].sum()
        boundary_count = dx[..., x_boundary].numel() + dy[..., y_boundary, :].numel()
        control_sum = dx[..., x_control].sum() + dy[..., y_control, :].sum()
        control_count = dx[..., x_control].numel() + dy[..., y_control, :].numel()

        boundary_mean = boundary_sum / max(boundary_count, 1)
        control_mean = control_sum / max(control_count, 1)
        if boundary_mean.item() < 1e-12 and control_mean.item() < 1e-12:
            metrics[period] = 0.0
        else:
            metrics[period] = (boundary_mean / control_mean.clamp_min(1e-12) - 1.0).item()

    return metrics


def calculate_decimation_artifact_metrics(estimate, target, low_quality, scale):
    metrics = {
        'dc_mae': decimation_consistency_mae(estimate, low_quality, scale),
        'grid_excess': residual_boundary_excess(estimate, target),
    }
    metrics.update(phase_impulse_metrics(estimate, target, scale))
    return metrics

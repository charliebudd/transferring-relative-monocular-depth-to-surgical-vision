import torch
from typing import Tuple, Sequence

def __apply_mask(depths, masks) -> Sequence[torch.Tensor]:
    if masks == None:
        return [depth for depth in depths]
    else:
        return [depth[mask] for depth, mask in zip(depths, masks)]

def normalise_depths(depths:torch.Tensor, masks:torch.Tensor) -> torch.Tensor:
    masked_depths = __apply_mask(depths, masks)
    means = torch.stack([md.mean() for md in masked_depths])
    stds = torch.stack([md.std() for md in masked_depths])
    return (depths - means[:, None, None, None]) / stds[:, None, None, None]

def fit_shifts_and_scales(source_depths:torch.Tensor, target_depths:torch.Tensor, masks:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_source_depths = __apply_mask(source_depths, masks)
    masked_target_depths = __apply_mask(target_depths, masks)
    shifts, scales = [], []
    for masked_source_depth, masked_target_depth in zip(masked_source_depths, masked_target_depths):
        A = torch.vander(masked_source_depth, 2)
        B = masked_target_depth.unsqueeze(-1)
        X = torch.pinverse(A, rcond=1e-4) @ B
        scale, shift = X[0], X[1]
        shifts.append(shift)
        scales.append(scale)
    return torch.cat(shifts), torch.cat(scales)

def fit_depths(source_depths:torch.Tensor, target_depths:torch.Tensor, masks:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        shifts, scales = fit_shifts_and_scales(source_depths, target_depths, masks)
    fitted_depths = (shifts[:, None, None, None] + scales[:, None, None, None] * source_depths)
    return fitted_depths
    
def ssimae(predicted_depths:torch.Tensor, target_depths:torch.Tensor, masks:torch.Tensor, normalise:bool=True) -> torch.Tensor:
    if normalise:
        target_depths = normalise_depths(target_depths, masks)
    with torch.no_grad():
        shifts, scales = fit_shifts_and_scales(predicted_depths, target_depths, masks)
    error_maps = (shifts[:, None, None, None] + scales[:, None, None, None] * predicted_depths) - target_depths
    ssimaes = torch.stack([masked_error_map.abs().mean() for masked_error_map in __apply_mask(error_maps, masks)])
    return ssimaes

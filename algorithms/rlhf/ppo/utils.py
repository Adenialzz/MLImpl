import torch
import torch.nn.functional as F
from typing import Dict, Optional, Iterable

def logical_or_without_broadcasting(x, y):
    assert x.shape[:-1] == y.shape[:-1]
    input_length = x.shape[1]
    output_length = y.shape[1]

    padding_amount = output_length - input_length

    assert padding_amount >= 0

    padded_x = F.pad(x, (0, padding_amount))

    return torch.logical_or(padded_x, y)

def gather_dict(d: Dict[str, torch.Tensor], device: torch.device, keys: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Move the tensors at the given keys to device."""
    if keys is None:
        keys = d.keys()

    for key in keys:
        d[key] = d[key].to(device)

    return d




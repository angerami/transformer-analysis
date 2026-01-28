import torch
import platform


def get_device(device=None):
    """
    Get torch device with automatic fallback.
    On macOS: defaults to cpu (mps has stability issues)
    Otherwise: cuda -> cpu

    Args:
        device: Manual override ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)

    if platform.system() == "Darwin":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

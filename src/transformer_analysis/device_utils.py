import torch


def get_device(device=None):
    """
    Pick the best available device.

    Priority: explicit override -> CUDA (Colab / lab box) -> MPS (Apple Silicon)
    -> CPU. Forward-only eval on MPS is solid in torch >= 2.

    Args:
        device: Manual override ('cuda', 'mps', 'cpu', or None for auto)

    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

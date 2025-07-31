import torch

def gradient_guided_mask(w_t: torch.Tensor,
                         w_prev: torch.Tensor,
                         gradient: torch.Tensor,
                         ratio: float = 0.1) -> list:
    """
    Picks top‑ratio*N indices by |(w_prev - w_t) * gradient|.
    """
    delta  = w_prev - w_t
    scores = delta * gradient
    k      = int(ratio * scores.numel())
    _, idx = torch.topk(scores.abs(), k)
    return idx.tolist()

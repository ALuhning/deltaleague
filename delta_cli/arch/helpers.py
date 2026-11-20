"""Optional helpers for participant bundles (importable inside the sandbox)."""
import torch


def gather_logprobs(hidden, weight, targets, bias=None, chunk=1024):
    """log_softmax(hidden @ weight.T + bias)[targets] in float32, chunked over positions.

    hidden: [B, T, H]; weight: [V, H]; targets: [B, T] -> [B, T]. Avoids
    materializing [B, T, V] logits for large vocabularies.
    """
    flat, ys = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    out = torch.empty(ys.shape, dtype=torch.float32, device=hidden.device)
    for i in range(0, flat.shape[0], chunk):
        logits = (flat[i:i + chunk] @ weight.T).float()
        if bias is not None:
            logits += bias.float()
        out[i:i + chunk] = logits.log_softmax(-1).gather(-1, ys[i:i + chunk, None])[:, 0]
    return out.view(targets.shape)


def last_logits(hidden, weight, bias=None):
    """Float32 logits for the final position: [B, T, H] -> [B, V]."""
    logits = (hidden[:, -1] @ weight.T).float()
    return logits + bias.float() if bias is not None else logits

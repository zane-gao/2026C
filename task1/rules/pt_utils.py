from __future__ import annotations

try:
    import pytensor.tensor as pt  # type: ignore
except Exception as exc:
    raise RuntimeError("pytensor is required for pt_utils") from exc


def sigmoid(x):
    # pytensor API compatibility: avoid pt.nnet.sigmoid (may be missing)
    return 1.0 / (1.0 + pt.exp(-x))


def masked_softmax(logits, mask, axis=-1):
    mask = pt.cast(mask, "float32")
    masked_logits = pt.switch(mask > 0, logits, -1e9)
    m = pt.max(masked_logits, axis=axis, keepdims=True)
    ex = pt.exp(masked_logits - m) * mask
    denom = pt.sum(ex, axis=axis, keepdims=True) + 1e-12
    return ex / denom


def soft_rank(x, tau: float):
    x = pt.as_tensor_variable(x)
    x_i = x.dimshuffle(0, "x")
    x_j = x.dimshuffle("x", 0)
    diff = (x_j - x_i) / tau
    s = sigmoid(diff)
    # subtract diagonal sigmoid(0)=0.5 to avoid counting self
    n = x.shape[0]
    return 1.0 + pt.sum(s, axis=1) - 0.5

def compute_confidence(n: int, lambda_param: float = 5.0) -> float:
    """Buffer saturation confidence C(n) = n / (n + lambda)."""
    return n / (n + lambda_param)

def compute_jaccard_dissimilarity(top_k_base: list, top_k_short: list) -> float:
    """Measures Jaccard dissimilarity between Baseline ALS and Short-Term model top-K sets."""
    set_base = set(top_k_base)
    set_short = set(top_k_short)
    intersection = len(set_base.intersection(set_short))
    union = len(set_base.union(set_short))
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)

def compute_hybrid_weights(n: int, top_k_base: list, top_k_short: list, lambda_param: float = 5.0):
    """Calculates w1 and w2 dynamic blending weights using a configurable lambda parameter."""
    confidence = compute_confidence(n, lambda_param=lambda_param)
    dissimilarity = compute_jaccard_dissimilarity(top_k_base, top_k_short)
    
    w2 = dissimilarity * confidence
    w1 = 1.0 - w2
    return w1, w2, dissimilarity
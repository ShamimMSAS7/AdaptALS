import numpy as np

class ShortTermModel:
    def __init__(self, item_factors, decay_gamma=0.8, max_buffer=20):
        self.item_factors = item_factors
        self.gamma = decay_gamma
        self.max_buffer = max_buffer

    def compute_centroid(self, session_buffer, decay_gamma=None):
        """Computes weighted latent centroid vector x_tilde from active session interactions."""
        if not session_buffer:
            return None

        gamma = decay_gamma if decay_gamma is not None else self.gamma
        recent_items = session_buffer[-self.max_buffer:]
        n = len(recent_items)
        
        weights = [gamma ** (n - 1 - m) * r for m, (item_id, r) in enumerate(recent_items)]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return None

        dim = len(next(iter(self.item_factors.values())))
        x_tilde = np.zeros(dim, dtype=np.float32)

        for m, (item_id, r) in enumerate(recent_items):
            if item_id in self.item_factors:
                x_tilde += weights[m] * self.item_factors[item_id]

        return x_tilde / weight_sum

    def predict_user_scores(self, x_tilde, candidate_item_ids):
        if x_tilde is None:
            return {}
        return {
            item_id: float(np.dot(x_tilde, self.item_factors[item_id]))
            for item_id in candidate_item_ids if item_id in self.item_factors
        }
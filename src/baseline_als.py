import numpy as np
from pyspark.ml.recommendation import ALS

class BaselineALS:
    def __init__(self, rank=32, max_iter=15, reg_param=0.05, alpha=20.0):
        self.als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            implicitPrefs=True,
            alpha=alpha,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            seed=42
        )
        self.model = None
        self.item_factors = {}
        self.user_factors = {}

    def fit(self, train_df):
        """Fits PySpark Implicit ALS and dumps latent matrices into fast local dicts."""
        self.model = self.als.fit(train_df)
        
        items_list = self.model.itemFactors.collect()
        self.item_factors = {row['id']: np.array(row['features'], dtype=np.float32) for row in items_list}
        
        users_list = self.model.userFactors.collect()
        self.user_factors = {row['id']: np.array(row['features'], dtype=np.float32) for row in users_list}

    def predict_user_scores(self, user_id, candidate_item_ids):
        if user_id not in self.user_factors:
            return {}
        u_vec = self.user_factors[user_id]
        scores = {}
        for item_id in candidate_item_ids:
            if item_id in self.item_factors:
                scores[item_id] = float(np.dot(u_vec, self.item_factors[item_id]))
        return scores
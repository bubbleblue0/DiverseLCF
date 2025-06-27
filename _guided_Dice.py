import numpy as np
import tensorflow as tf

class DiceRandomInitCF:
    def __init__(
            self,
            feature_min,
            feature_max,
            *,
            probability=0.8,
            tolerance=0.3,
            max_iter=200,
            min_iter=30,
            optimizer=None,
            pred_margin_weight=1.0,
            prox_weight=0.5,
            diversity_weight=1.0,
            categorical_penalty=0.0,
            learning_rate=0.05,
            n_counterfactuals=1,
            diversity_mode="dpp",  # dpp or avg_dist
            random_state=42,
            stopping_threshold=0.5,
    ):

        self.feature_min = tf.convert_to_tensor(feature_min, dtype=tf.float32)
        self.feature_max = tf.convert_to_tensor(feature_max, dtype=tf.float32)

        if len(self.feature_min.shape) == 1:
            self.feature_min = tf.expand_dims(self.feature_min, axis=0)
        if len(self.feature_min.shape) == 2:
            self.feature_min = tf.expand_dims(self.feature_min, axis=-1)
        if len(self.feature_max.shape) == 1:
            self.feature_max = tf.expand_dims(self.feature_max, axis=0)
        if len(self.feature_max.shape) == 2:
            self.feature_max = tf.expand_dims(self.feature_max, axis=-1)

        self.probability_ = tf.constant([probability])
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.pred_margin_weight = pred_margin_weight
        self.prox_weight = prox_weight
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_counterfactuals = n_counterfactuals
        self.diversity_mode = diversity_mode
        self.categorical_penalty = categorical_penalty
        self.stopping_threshold = stopping_threshold

        # 🔧 Automatically disable diversity loss if n_counterfactuals = 1
        if n_counterfactuals < 2:
            self.diversity_weight = 0.0
        else:
            self.diversity_weight = diversity_weight

        self.optimizer_ = (
            tf.optimizers.Adam(learning_rate=self.learning_rate)
            if optimizer is None
            else optimizer
        )

    def fit(self, model):
        self.model_ = model
        return self

    def pred_margin_loss(self, prediction, target_label):
        pred_target = prediction[:, target_label]
        return tf.reduce_mean(tf.maximum(0.0, 1.0 - pred_target))

    def weighted_mae(self, original_sample, cf_sample):
        return tf.reduce_mean(tf.abs(original_sample - cf_sample))

    def compute_proximity_loss(self, x, cf_samples):
        losses = [self.weighted_mae(x, cf) for cf in cf_samples]
        return tf.reduce_mean(losses)

    def compute_diversity_loss(self, cf_samples):
        if self.n_counterfactuals == 1:
            return 0.0
        pairwise_dists = []
        for i in range(self.n_counterfactuals):
            for j in range(i + 1, self.n_counterfactuals):
                pairwise_dists.append(tf.reduce_mean(tf.abs(cf_samples[i] - cf_samples[j])))

        if self.diversity_mode == "dpp":
            # DPP-style inverse distances
            matrix = []
            for d in pairwise_dists:
                matrix.append(1.0 / (1.0 + d))
            det = tf.reduce_prod(matrix)
            return -tf.math.log(det + 1e-5)
        elif self.diversity_mode == "avg_dist":
            return -tf.reduce_mean(pairwise_dists)
        else:
            raise ValueError("Unknown diversity mode.")

    def compute_loss(self, x, cf_samples, preds, target_label):
        # pred_loss = tf.reduce_mean([self.pred_margin_loss(pred[:, target_label], target_label) for pred in preds])
        pred_loss = tf.reduce_mean([self.pred_margin_loss(pred, target_label) for pred in preds])

        proximity_loss = self.compute_proximity_loss(x, cf_samples)
        diversity_loss = self.compute_diversity_loss(cf_samples)

        total_loss = (
            self.pred_margin_weight * pred_loss
            + self.prox_weight * proximity_loss
            + self.diversity_weight * diversity_loss
        )

        # === LOGGING ===
        # tf.print("[compute_loss] pred_loss:", pred_loss,
        #          ", proximity_loss:", proximity_loss,
        #          ", diversity_loss:", diversity_loss,
        #          ", total_loss:", total_loss)

        return total_loss

    def clip_to_bounds(self, cf_samples):
        clipped = []
        for idx, cf in enumerate(cf_samples):
            tiled_min = tf.tile(self.feature_min, [tf.shape(cf)[0], 1, 1])
            tiled_max = tf.tile(self.feature_max, [tf.shape(cf)[0], 1, 1])
            clipped_cf = tf.clip_by_value(cf, tiled_min, tiled_max)
            clipped.append(tf.Variable(clipped_cf))  # <- Re-wrap to Variable
        return clipped

    def transform(self, x, target_labels):
        # print(x.shape) # (28, 288, 1)
        result_samples = np.empty((x.shape[0], self.n_counterfactuals, x.shape[1], x.shape[2]))
        losses = np.empty(x.shape[0])

        for i in range(x.shape[0]):
            if i % 25 == 0:
                print(f"{i + 1} samples transformed.")

            cfs, loss = self._transform_sample(x[i:i + 1], target_labels[i])

            result_samples[i] = cfs
            losses[i] = loss

        print(f"All {i + 1} samples transformed.")
        return result_samples, losses

    def _transform_sample(self, x, target_label):
        best_cfs = None
        best_loss = float("inf")

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        n_init_attempts = 3  # Like in DiverseLatentCF

        best_cfs_all = []
        best_losses_all = []

        for init_attempt in range(n_init_attempts):
            # 1. Initialize
            cf_samples = [
                tf.Variable(
                    x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1),
                    dtype=tf.float32
                )
                for _ in range(self.n_counterfactuals)
            ]

            early_stop_flags = [False] * self.n_counterfactuals
            best_cfs_this = [None] * self.n_counterfactuals
            best_losses_this = [float('inf')] * self.n_counterfactuals

            for it in range(self.max_iter):
                with tf.GradientTape() as tape:
                    preds = [self.model_(cf) for cf in cf_samples]
                    loss = self.compute_loss(x, cf_samples, preds, target_label)

                grads = tape.gradient(loss, cf_samples)
                # print("grads", grads)
                self.optimizer_.apply_gradients(zip(grads, cf_samples))

                cf_samples = self.clip_to_bounds(cf_samples)

                # Check early stopping for each CF
                for idx, pred in enumerate(preds):
                    if early_stop_flags[idx]:
                        continue
                    pred_val = tf.reduce_mean(pred[:, target_label])
                    if self.probability_ - pred_val <= self.tolerance_:
                        early_stop_flags[idx] = True

                    if loss.numpy() < best_losses_this[idx]:
                        best_losses_this[idx] = loss.numpy()
                        best_cfs_this[idx] = cf_samples[idx].numpy().squeeze(axis=0)

                if all(early_stop_flags) and it >= self.min_iter:
                    break

            best_cfs_all.append(best_cfs_this)
            best_losses_all.append(best_losses_this)

        # 2. After all initialization attempts, pick best counterfactuals
        final_cfs = []
        for i in range(self.n_counterfactuals):
            best_valid_cf = None
            best_valid_loss = float('inf')
            best_any_cf = None
            best_any_loss = float('inf')

            for j in range(n_init_attempts):
                candidate_cf = best_cfs_all[j][i]
                candidate_loss = best_losses_all[j][i]
                if candidate_cf is None:
                    continue

                pred = self.model_(candidate_cf[np.newaxis, ...])
                pred_val = pred[0, target_label]
                if pred_val >= self.probability_:
                    if candidate_loss < best_valid_loss:
                        best_valid_loss = candidate_loss
                        best_valid_cf = candidate_cf
                if candidate_loss < best_any_loss:
                    best_any_loss = candidate_loss
                    best_any_cf = candidate_cf

            if best_valid_cf is not None:
                final_cfs.append(best_valid_cf)
            else:
                final_cfs.append(best_any_cf)

        return np.stack(final_cfs), float(np.mean([np.min(l) for l in best_losses_all]))

    # def _transform_sample(self, x, target_label):
    #     best_cfs = None
    #     best_loss = float("inf")
    #
    #     tf.random.set_seed(self.random_state)
    #     np.random.seed(self.random_state)
    #
    #     # Initialize CF samples near x (no squeeze)
    #     cf_samples = [
    #         tf.Variable(
    #             x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1),
    #             dtype=tf.float32
    #         )
    #         for _ in range(self.n_counterfactuals)
    #     ]
    #
    #     for it in range(self.max_iter):
    #         with tf.GradientTape() as tape:
    #             preds = [self.model_(cf) for cf in cf_samples]
    #             loss = self.compute_loss(x, cf_samples, preds, target_label)
    #
    #         grads = tape.gradient(loss, cf_samples)
    #
    #         # Filter out None grads
    #         grads_and_vars = [(g, v) for g, v in zip(grads, cf_samples) if g is not None]
    #
    #         if grads_and_vars:  # Only apply if non-empty
    #             self.optimizer_.apply_gradients(grads_and_vars)
    #
    #         cf_samples = self.clip_to_bounds(cf_samples)
    #
    #         # Early stopping check
    #         valid = True
    #         for pred in preds:
    #             pred_val = tf.reduce_mean(pred)  # scalar
    #             if target_label == 1 and pred_val < self.stopping_threshold:
    #                 valid = False
    #             if target_label == 0 and pred_val > (1 - self.stopping_threshold):
    #                 valid = False
    #
    #         if it >= self.min_iter and valid:
    #             break
    #
    #         if loss.numpy() < best_loss:
    #             best_loss = loss.numpy()
    #             best_cfs = [cf.numpy() for cf in cf_samples]  # no squeeze yet
    #
    #     if best_cfs is None:
    #         best_cfs = [cf.numpy() for cf in cf_samples]
    #
    #     # At the final step: squeeze batch axis
    #     best_cfs = [np.squeeze(cf, axis=0) for cf in best_cfs]
    #
    #     return np.stack(best_cfs), best_loss


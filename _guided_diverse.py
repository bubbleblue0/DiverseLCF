import warnings
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from wildboar.explain import IntervalImportance
from LIMESegment.Utils.explanations import LIMESegment


class DiverseLatentCF:
    def __init__(
        self,
        probability=0.8,
        *,
        tolerance=0.3,
        max_iter=200,
        optimizer=None,
        autoencoder=None,
        pred_margin_weight=1.0,
        prox_weight=0.5,
        learning_rate=0.05,
        diversity_weight=0.3,
        diversity_mode="pairwise",
        n_counterfactuals=1,
        random_state=42,
    ):
        self.probability_ = tf.constant([probability])
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        self.autoencoder = autoencoder
        self.pred_margin_weight = pred_margin_weight
        self.prox_weight = prox_weight
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_counterfactuals = n_counterfactuals
        self.diversity_mode = diversity_mode
        self.n_init_attempts = 3  # Number of initializations per CF
        self.optimizer_ = (
            tf.optimizers.Adam(learning_rate=self.learning_rate) if optimizer is None else optimizer
        )

        # 🔧 Disable diversity loss when only 1 CF
        if n_counterfactuals < 2:
            self.diversity_weight = 0.0
        else:
            self.diversity_weight = diversity_weight


    def fit(self, model):
        if self.autoencoder:
            encoder_inputs = self.autoencoder.input
            z_mean_layer = self.autoencoder.get_layer("z_mean").output
            z_log_var_layer = self.autoencoder.get_layer("z_log_var").output
            z_layer = self.autoencoder.get_layer("z").output
            self.encoder_ = keras.Model(inputs=encoder_inputs, outputs=[z_mean_layer, z_log_var_layer, z_layer])

            latent_dim = z_layer.shape[1]
            decoder_input = keras.Input(shape=(latent_dim,))
            x = decoder_input
            for layer in self.autoencoder.get_layer("decoder").layers[1:]:
                x = layer(x)
            self.decoder_ = keras.Model(inputs=decoder_input, outputs=x)
        else:
            self.decoder_ = None
            self.encoder_ = None
        self.model_ = model
        return self

    def pred_margin_loss(self, prediction):
        """Binary cross-entropy loss for validity term. Assumes prediction is a probability."""
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(self.probability_, prediction)

    # def pred_margin_loss(self, prediction, target_label):
    #     pred_target = prediction[:, target_label]
    #     return tf.reduce_mean(tf.maximum(0.0, 1.0 - pred_target))


    def compute_diversity_loss(self, z_list):
        k = len(z_list)
        if k < 2:
            return tf.constant(0.0, dtype=tf.float32)  # No diversity to compute

        # Compute pairwise L1 distances
        pairwise_dists = []
        for i in range(k):
            for j in range(i + 1, k):
                dist = tf.reduce_mean(tf.abs(z_list[i] - z_list[j]))
                pairwise_dists.append(dist)

        if self.diversity_mode == "pairwise":
            return -tf.reduce_mean(pairwise_dists)

        elif self.diversity_mode == "dpp":
            sim_products = [1.0 / (1.0 + d) for d in pairwise_dists]
            product = tf.reduce_prod(sim_products)
            return -tf.math.log(product + 1e-5)

        else:
            raise ValueError("Unsupported diversity mode: choose 'pairwise' or 'dpp'")

    def weighted_mae(self, original_sample, cf_sample):
        return tf.reduce_sum(tf.abs(original_sample - cf_sample))

    def compute_loss(self, x, decoded_list, preds, z_list, target_label, early_stop_flags, verbose=True):
        """Compute total loss (cf_loss + diversity loss) across all active CFs"""
        active_losses = []
        cf_losses = []

        margin_losses = []
        step_losses = []

        for idx, (decoded, pred) in enumerate(zip(decoded_list, preds)):
            if early_stop_flags[idx]:
                cf_losses.append(0.0)
                margin_losses.append(0.0)
                step_losses.append(0.0)
                continue

            # pred_margin_loss = self.pred_margin_loss(pred, target_label)
            pred_margin_loss = self.pred_margin_loss(pred[:, target_label])
            step_loss = self.weighted_mae(tf.cast(x, tf.float32), tf.cast(decoded, tf.float32))

            loss = self.pred_margin_weight * pred_margin_loss + self.prox_weight * step_loss

            cf_losses.append(loss)
            active_losses.append(loss)
            margin_losses.append(pred_margin_loss)
            step_losses.append(step_loss)

        cf_loss_avg = tf.reduce_mean(active_losses) if active_losses else 0.0
        diversity_loss = self.compute_diversity_loss(z_list)
        print("Diversity loss: {}".format(diversity_loss)) #Diversity loss: [22.958317       nan       nan]
        total_joint_loss = cf_loss_avg + self.diversity_weight * diversity_loss

        if verbose:
            print(f"  Prediction margin loss avg: {tf.reduce_mean(margin_losses):.4f}")
            print(f"  Input proximity loss avg   : {tf.reduce_mean(step_losses):.4f}")
            print(f"  Diversity loss             : {diversity_loss:.4f}")
            print(f"  Total joint loss           : {total_joint_loss:.4f}")

        return total_joint_loss, cf_losses, diversity_loss

    def transform(self, x, target_labels):
        result_samples = np.empty((x.shape[0], self.n_counterfactuals, x.shape[1], x.shape[2]))
        losses = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            if i % 25 == 0:
                print(f"{i + 1} samples been transformed.")
            cfs, loss = self._transform_sample(
                x[np.newaxis, i], target_labels[i]
            )
            result_samples[i] = cfs
            losses[i] = loss
        print(f"{i + 1} samples been transformed, in total.")

        return result_samples, losses


    def _transform_sample(self, x, target_label):
        z_mean, z_log_var, z = self.encoder_(x)

        best_decoded_all = []
        best_loss_all = []

        for init_attempt in range(self.n_init_attempts):
            z_list = [
                tf.Variable(z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(shape=z_mean.shape, seed=self.random_state))
                for _ in range(self.n_counterfactuals)
            ]


            # # z_list = [
            # #     tf.Variable(
            # #         z_mean + tf.random.normal(shape=z_mean.shape, mean=0.0, stddev=0.05, seed=self.random_state))
            # #     for _ in range(self.n_counterfactuals)
            # # ]
            # z_list = [
            #     tf.Variable(z_mean) for _ in range(self.n_counterfactuals)
            # ]

            best_decoded = [None] * self.n_counterfactuals
            best_loss = [float('inf')] * self.n_counterfactuals
            early_stop_flags = [False] * self.n_counterfactuals

            for it in range(self.max_iter):
                with tf.GradientTape() as tape:
                    decoded_list = [self.decoder_(zi) for zi in z_list]
                    preds = [self.model_(di) for di in decoded_list]
                    total_joint_loss, cf_losses, _ = self.compute_loss(
                        x, decoded_list, preds, z_list, target_label, early_stop_flags, verbose=True
                    )

                # Apply gradients jointly to active z_i
                active_z_list = [z_list[i] for i in range(self.n_counterfactuals) if not early_stop_flags[i]]
                grads = tape.gradient(total_joint_loss, active_z_list)
                self.optimizer_.apply_gradients(zip(grads, active_z_list))

                for idx in range(self.n_counterfactuals):
                    if early_stop_flags[idx]:
                        continue
                    pred = preds[idx]
                    if self.probability_ - pred[:, target_label] <= self.tolerance_:
                        early_stop_flags[idx] = True

                    if cf_losses[idx] < best_loss[idx]:
                        best_loss[idx] = cf_losses[idx].numpy()
                        best_decoded[idx] = decoded_list[idx].numpy().squeeze(axis=0)

                if all(early_stop_flags):
                    break

            best_decoded_all.append(best_decoded)
            best_loss_all.append(best_loss)

        # Choose best across all attempts per CF slot
        final_decoded = []
        for i in range(self.n_counterfactuals):
            best_valid = None
            best_valid_loss = float("inf")
            best_any = None
            best_any_loss = float("inf")

            for j in range(self.n_init_attempts):
                decoded = best_decoded_all[j][i]
                loss = best_loss_all[j][i]
                if decoded is not None:
                    pred = self.model_(decoded[np.newaxis, ...])
                    if pred[0, target_label] >= self.probability_:
                        if loss < best_valid_loss:
                            best_valid_loss = loss
                            best_valid = decoded
                    if loss < best_any_loss:
                        best_any_loss = loss
                        best_any = decoded

            final_decoded.append(best_valid if best_valid is not None else best_any)

        return np.stack(final_decoded), float(np.mean([np.min(l) for l in best_loss_all]))


def get_local_weights(input_sample, classifier_model, random_state=None, pred_label=None):
    n_timesteps, n_dims = input_sample.shape
    desired_label = int(1 - pred_label) if pred_label is not None else 1
    seg_imp, seg_idx = LIMESegment(
        input_sample,
        classifier_model,
        model_type=desired_label,
        cp=10,
        window_size=10,
        random_state=random_state,
    )

    if desired_label == 1:
        masking_threshold = np.percentile(seg_imp, 25)
        masking_idx = np.where(seg_imp <= masking_threshold)
    else:
        masking_threshold = np.percentile(seg_imp, 75)
        masking_idx = np.where(seg_imp >= masking_threshold)

    weighted_steps = np.ones(n_timesteps)
    for start_idx in masking_idx[0]:
        weighted_steps[seg_idx[start_idx] : seg_idx[start_idx + 1]] = 0

    weighted_steps = weighted_steps.reshape(1, n_timesteps, n_dims)
    return weighted_steps

def get_global_weights(input_samples, input_labels, classifier_model, random_state=None):
    n_samples, n_timesteps, n_dims = input_samples.shape

    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            p = self.model.predict(X.reshape(n_samples, n_timesteps, 1))
            return np.argmax(p, axis=1)

    clf = ModelWrapper(classifier_model)
    i = IntervalImportance(scoring="accuracy", n_interval=10, random_state=random_state)
    i.fit(clf, input_samples.reshape(input_samples.shape[0], -1), input_labels)

    masking_threshold = np.percentile(i.importances_.mean, 75)
    masking_idx = np.where(i.importances_.mean >= masking_threshold)

    weighted_steps = np.ones(n_timesteps)
    seg_idx = i.intervals_
    for start_idx in masking_idx[0]:
        weighted_steps[seg_idx[start_idx][0] : seg_idx[start_idx][1]] = 0

    weighted_steps = weighted_steps.reshape(1, n_timesteps, 1)
    return weighted_steps

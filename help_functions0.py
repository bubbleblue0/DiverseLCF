import os
import csv
import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
from scipy.spatial import distance
from tslearn.utils import to_sklearn_dataset
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from _guided_glacier import ModifiedLatentCF

# from keras import backend as K

import csv

class ResultWriter:
    def __init__(self, file_name, dataset_name, folder_path="results/csv0"):
        os.makedirs(folder_path, exist_ok=True)
        self.file_name = os.path.join(folder_path, file_name)
        self.dataset_name = dataset_name

    def write_head(self):
        # Only write the header if the file does not exist or is empty
        if not os.path.exists(self.file_name) or os.path.getsize(self.file_name) == 0:
            with open(self.file_name, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "dataset",
                        "fold_id",
                        "method",
                        "classifier_accuracy",
                        "autoencoder_loss",
                        "best_lr",
                        "proximity_l1",
                        "proximity_l2",
                        "proximity_l_inf",
                        "validity",
                        "compactness",
                        "svm_ood",
                        "lof_ood",
                        "ifo_ood",
                        "diversity",
                        "pred_margin_weight",
                        "diversity_weight",
                        "prox_weight",
                        "learning_rate",
                        "probability",
                        "diversity_mode",
                        "process_time",
                        "num_div_cfs"
                    ]
                )

    def write_result(
        self,
        fold_id,
        method_name,
        acc,
        ae_loss,
        best_lr,
        evaluate_res,
        pred_margin_weight,
        diversity_weight,
        prox_weight,
        learning_rate,
        probability,
        diversity_mode,
        process_time,
        num_div_cfs,
    ):
        (
            proxi_l1,
            proxi_l2,
            proxi_l_inf,
            valid,
            compact,
            OOD_svm,
            OOD_lof,
            mean_OOD_ifo,
            diversity,
        ) = evaluate_res

        with open(self.file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.dataset_name,
                    fold_id,
                    method_name,
                    acc,
                    ae_loss,
                    best_lr,
                    proxi_l1,
                    proxi_l2,
                    proxi_l_inf,
                    valid,
                    compact,
                    OOD_svm,
                    OOD_lof,
                    mean_OOD_ifo,
                    diversity,
                    pred_margin_weight,
                    diversity_weight,
                    prox_weight,
                    learning_rate,
                    probability,
                    diversity_mode,
                    process_time,
                    num_div_cfs,
                ]
            )



"""
time series scaling
"""


def time_series_normalize(data, n_timesteps, n_features=1, scaler=None):
    # reshape data to 1 column
    data_reshaped = data.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_reshaped)

    normalized = scaler.transform(data_reshaped)

    # return reshaped data into [samples, timesteps, features]
    return normalized.reshape(-1, n_timesteps, n_features), scaler

def readUCR(ds_name):
    path = "/home/dmlab_a/Peiyu0/CF_minibo/UCRArchive_2018/"
    train_data = np.loadtxt(path + ds_name +'/' + ds_name+ '_TRAIN.tsv' , delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(path + ds_name + '/' + ds_name + '_TEST.tsv', delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    # print(x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test

from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder
import os

def readUEA(ds_name):
    DATA_PATH = "/home/dmlab_a/Peiyu0/MCELS/MTS_DATA"
    train_data, train_label = load_from_tsfile(os.path.join(DATA_PATH, ds_name, ds_name + "_TRAIN.ts"),
                                                       return_data_type="numpy3d")
    test_data, test_label = load_from_tsfile(os.path.join(DATA_PATH, ds_name, ds_name + "_TEST.ts"),
                                             return_data_type="numpy3d")
    encoder = LabelEncoder()
    train_label = encoder.fit_transform(train_label)
    test_label = encoder.fit_transform(test_label)


    return train_data, test_data, train_label, test_label


def time_series_revert(normalized_data, n_timesteps, n_features=1, scaler=None):
    # reshape data to 1 column
    data_reshaped = normalized_data.reshape(-1, 1)

    reverted = scaler.inverse_transform(data_reshaped)

    # return reverted data into [samples, timesteps, features]
    return reverted.reshape(-1, n_timesteps, n_features)


"""
data pre-processing
"""


def conditional_pad(X):
    num = X.shape[1]

    if num % 4 != 0:
        # find the next integer that can be divided by 4
        next_num = (int(num / 4) + 1) * 4
        padding_size = next_num - num
        X_padded = np.pad(
            X, pad_width=((0, 0), (0, padding_size), (0, 0))
        )  # pad for 3d array

        return X_padded, padding_size

    # else return the original X
    return X, 0  # padding size = 0


def remove_paddings(cf_samples, padding_size):
    if padding_size != 0:
        # use np.squeeze() to cut the last time-series dimension, for evaluation
        cf_samples = np.squeeze(cf_samples[:, :-padding_size, :])
    else:
        cf_samples = np.squeeze(cf_samples)
    return cf_samples


# Upsampling the minority class
def upsample_minority(X, y, pos_label=1, neg_label=0, random_state=39):
    # Get counts
    pos_counts = pd.value_counts(y)[pos_label]
    neg_counts = pd.value_counts(y)[neg_label]
    # Divide by class
    X_pos, X_neg = X[y == pos_label], X[y == neg_label]

    if pos_counts == neg_counts:
        # Balanced dataset
        return X, y
    elif pos_counts > neg_counts:
        # Imbalanced dataset
        X_neg_over = resample(
            X_neg, replace=True, n_samples=pos_counts, random_state=random_state
        )
        X_concat = np.concatenate([X_pos, X_neg_over], axis=0)
        y_concat = np.array(
            [pos_label for i in range(pos_counts)]
            + [neg_label for j in range(pos_counts)]
        )
    else:
        # Imbalanced dataset
        X_pos_over = resample(
            X_pos, replace=True, n_samples=neg_counts, random_state=random_state
        )
        X_concat = np.concatenate([X_pos_over, X_neg], axis=0)
        y_concat = np.array(
            [pos_label for i in range(neg_counts)]
            + [neg_label for j in range(neg_counts)]
        )

    # Shuffle the index after up-sampling
    X_concat, y_concat = shuffle(X_concat, y_concat, random_state=random_state)

    return X_concat, y_concat


"""
deep models needed
"""


# Method: For plotting the accuracy/loss of keras models
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


# Method: Fix the random seeds to get consistent models
def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)


"""
evaluation metrics
"""
def cf_ood(X_train, counterfactual_examples):

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(X_train))), novelty=True, metric='euclidean')
    lof.fit(to_sklearn_dataset(X_train))

    novelty_detection = lof.predict(to_sklearn_dataset(counterfactual_examples))

    ood= np.count_nonzero(novelty_detection == -1)
    OOD_lof = ood / len(counterfactual_examples)

    # One-Class SVM (OC-SVM)
    clf = OneClassSVM(gamma='scale', nu=0.02).fit(to_sklearn_dataset(X_train))

    novelty_detection = clf.predict(to_sklearn_dataset(counterfactual_examples))

    ood = np.count_nonzero(novelty_detection == -1)
    OOD_svm = ood/ len(counterfactual_examples)

    # Initialize a list to store OOD results for min_edit_cf
    OOD_ifo = []

    # Loop over different random seeds
    for seed in range(10):
        iforest = IsolationForest(random_state=seed).fit(to_sklearn_dataset(X_train))

        novelty_detection = iforest.predict(to_sklearn_dataset(counterfactual_examples))

        ood = np.count_nonzero(novelty_detection == -1)

        OOD_ifo.append((ood/ len(counterfactual_examples)))

    mean_OOD_ifo = np.mean(OOD_ifo)

    return OOD_svm, OOD_lof, mean_OOD_ifo


def fit_evaluation_models(n_neighbors_lof, n_neighbors_nn, training_data):
    # Fit the LOF model for novelty detection (novelty=True)
    lof_estimator = LocalOutlierFactor(
        n_neighbors=n_neighbors_lof,
        novelty=True,
        metric="euclidean",
    )
    lof_estimator.fit(training_data)

    # Fit an unsupervised 1NN with all the training samples from the desired class
    nn_model = NearestNeighbors(n_neighbors=n_neighbors_nn, metric="euclidean")
    nn_model.fit(training_data)
    return lof_estimator, nn_model


def evaluate(
    X_train,
    X_test,
    cfs,
    target_labels,
    cf_labels,
    num_div_cfs,
):
    l1, l2, l_inf = distance_metrics(X_test, cfs)
    valid = validity_score(target_labels, cf_labels)
    compact = compactness_score(X_test, cfs)
    OOD_svm, OOD_lof, mean_OOD_ifo = cf_ood(X_train, cfs)
    diversity = diversity_score(cfs, num_div_cfs)

    # TODO: add LOF and RP score for debugging training?
    # lof_score = calculate_lof(
    #     cf_samples, pred_labels, lof_estimator_pos, lof_estimator_neg
    # )
    # rp_score = relative_proximity(
    #     X_pred_neg, cf_samples, pred_labels, nn_estimator_pos, nn_estimator_neg
    # )

    return l1, l2, l_inf, valid, compact, OOD_svm, OOD_lof, mean_OOD_ifo, diversity  # lof_score, rp_score,

import numpy as np
import os

def evaluate_and_save_valid(
    X_train,
    X_test_repeat,
    cfs,
    target_labels,
    cf_labels,
    num_div_cfs,
    save_dir="unknow",
    dataset_name="unknown"
):
    os.makedirs(save_dir, exist_ok=True)

    if len(cfs.shape) == 4:
        cfs = cfs.reshape(-1, cfs.shape[2], cfs.shape[3])
    if len(X_test_repeat.shape) == 4:
        X_test_repeat = X_test_repeat.reshape(-1, X_test_repeat.shape[2], X_test_repeat.shape[3])

    # Validity mask
    valid_mask = (cf_labels == target_labels)

    # Save valid counterfactuals and corresponding original inputs
    valid_cfs = cfs[valid_mask]
    valid_x = X_test_repeat[valid_mask]

    np.save(os.path.join(save_dir, f"valid_cfs_{dataset_name}.npy"), valid_cfs)
    np.save(os.path.join(save_dir, f"valid_x_{dataset_name}.npy"), valid_x)

    # If no valid counterfactuals, return NaNs
    if len(valid_cfs) == 0:
        print(f"No valid counterfactuals for {dataset_name}")
        return [np.nan] * 9

    l1, l2, l_inf = distance_metrics(valid_x, valid_cfs)
    valid = len(valid_cfs) / len(cfs)
    compact = compactness_score(valid_x, valid_cfs)
    OOD_svm, OOD_lof, mean_OOD_ifo = cf_ood(X_train, valid_cfs)
    diversity = diversity_score(valid_cfs, num_div_cfs)

    return l1, l2, l_inf, valid, compact, OOD_svm, OOD_lof, mean_OOD_ifo, diversity


def evaluate_diverse_cfs_for_one_sample(
    X_train,
    single_test_sample,
    cfs,  # shape (N, T, 1)
    target_label,
    cf_labels,  # shape (N,)
):
    l1s = []
    l2s = []
    l_infs = []
    valids = []
    compacts = []
    ood_svms = []
    ood_lofs = []
    ood_ifos = []

    for i in range(len(cfs)):
        cf = cfs[i:i+1]  # shape (1, T, 1)
        cf_label = cf_labels[i:i+1]

        l1, l2, l_inf = distance_metrics(single_test_sample, cf)
        valid = validity_score(target_label, cf_label)
        compact = compactness_score(single_test_sample, cf)
        svm, lof, ifo = cf_ood(X_train, cf)

        l1s.append(l1)
        l2s.append(l2)
        l_infs.append(l_inf)
        valids.append(valid)
        compacts.append(compact)
        ood_svms.append(svm)
        ood_lofs.append(lof)
        ood_ifos.append(ifo)

    # Aggregate: use mean
    return (
        np.mean(l1s),
        np.mean(l2s),
        np.mean(l_infs),
        np.mean(valids),
        np.mean(compacts),
        np.mean(ood_svms),
        np.mean(ood_lofs),
        np.mean(ood_ifos),
    )

def euclidean_distance(X, cf_samples, average=True):
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances

# import numpy as np

def distance_metrics(X, cf_samples, average=True):
    """
    Compute vectorized L1, L2 (Euclidean), and L∞ distances between X and cf_samples.

    Args:
        X: numpy array of shape (n_samples, n_timestamps)
        cf_samples: numpy array of same shape as X
        average: if True, return mean of each metric; else, return per-sample values

    Returns:
        A tuple: (l1, l2, l_inf), each either a scalar (mean) or a numpy array (per-sample)
    """
    diffs = X - cf_samples

    l1 = np.sum(np.abs(diffs), axis=1)
    l2 = np.linalg.norm(diffs, axis=1)
    l_inf = np.max(np.abs(diffs), axis=1)

    if average:
        return np.mean(l1), np.mean(l2), np.mean(l_inf)
    else:
        return l1, l2, l_inf

def compute_diversity(cfs_single, metric='l2'):
    """
    cfs_single: numpy array of shape (k, T, 1) — k counterfactuals for one input.
    """
    k = cfs_single.shape[0]
    total_dist = 0.0
    count = 0

    # Flatten each counterfactual (if needed)
    cfs_flat = cfs_single.reshape(k, -1)

    for i in range(k):
        for j in range(i + 1, k):
            if metric == 'l2':
                dist = np.linalg.norm(cfs_flat[i] - cfs_flat[j])
            elif metric == 'l1':
                dist = np.sum(np.abs(cfs_flat[i] - cfs_flat[j]))
            else:
                raise ValueError("Unsupported metric")

            total_dist += dist
            count += 1

    if count == 0:
        return 0
    return total_dist / count

def diversity_score(cf_samples, num_samples):
    diversities = []
    for i in range(num_samples):
        cfs_single = cf_samples[i]  # shape (k, T, 1)
        div = compute_diversity(cfs_single)
        diversities.append(div)

    mean_diversity = np.mean(diversities) if num_samples != 1 else 0
    return mean_diversity


def validity_score(target_labels, cf_labels):
    return accuracy_score(y_true=target_labels, y_pred=cf_labels)


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples):
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.01)

    # return a positive compactness, instead of 1 - np.mean(..)
    return np.mean(c, axis=(1, 0))


# def sax_compactness(X, cf_samples, n_timesteps):
#     from wildboar.transform import symbolic_aggregate_approximation

#     X = symbolic_aggregate_approximation(X, window=window, n_bins=n_bins)
#     cf_samples = symbolic_aggregate_approximation(
#         cf_samples, window=window, n_bins=n_bins
#     )

#     # absolute tolerance atol=0.01, 0.001, OR 0.0001?
#     c = np.isclose(X, cf_samples, atol=0.01)

#     return np.mean(1 - np.sum(c, axis=1) / n_timesteps)


def calculate_lof(cf_samples, pred_labels, lof_estimator_pos, lof_estimator_neg):
    desired_labels = 1 - pred_labels  # for binary classification

    pos_idx, neg_idx = (
        np.where(desired_labels == 1)[0],  # pos_label = 1
        np.where(desired_labels == 0)[0],  # neg_label - 0
    )
    # check if the NumPy array is empty
    if pos_idx.any():
        y_pred_cf1 = lof_estimator_pos.predict(cf_samples[pos_idx])
        n_error_cf1 = y_pred_cf1[y_pred_cf1 == -1].size
    else:
        n_error_cf1 = 0

    if neg_idx.any():
        y_pred_cf2 = lof_estimator_neg.predict(cf_samples[neg_idx])
        n_error_cf2 = y_pred_cf2[y_pred_cf2 == -1].size
    else:
        n_error_cf2 = 0

    lof_score = (n_error_cf1 + n_error_cf2) / cf_samples.shape[0]
    return lof_score


def relative_proximity(
    X_inputs, cf_samples, pred_labels, nn_estimator_pos, nn_estimator_neg
):
    desired_labels = 1 - pred_labels  # for binary classification

    nn_distance_list = np.array([])
    proximity_list = np.array([])

    pos_idx, neg_idx = (
        np.where(desired_labels == 1)[0],  # pos_label = 1
        np.where(desired_labels == 0)[0],  # neg_label = 0
    )
    if pos_idx.any():
        nn_distances1, _ = nn_estimator_pos.kneighbors(
            X_inputs[pos_idx], return_distance=True
        )
        nn_distances1 = np.squeeze(nn_distances1, axis=-1)
        proximity1 = euclidean_distance(
            X_inputs[pos_idx], cf_samples[pos_idx], average=False
        )

        nn_distance_list = np.concatenate((nn_distance_list, nn_distances1), axis=0)
        proximity_list = np.concatenate((proximity_list, proximity1), axis=0)

    if neg_idx.any():
        nn_distances2, _ = nn_estimator_neg.kneighbors(
            X_inputs[neg_idx], return_distance=True
        )
        nn_distances2 = np.squeeze(nn_distances2, axis=-1)
        proximity2 = euclidean_distance(
            X_inputs[neg_idx], cf_samples[neg_idx], average=False
        )

        nn_distance_list = np.concatenate((nn_distance_list, nn_distances2), axis=0)
        proximity_list = np.concatenate((proximity_list, proximity2), axis=0)

    # TODO: paired proximity score for (X_pred_neg, cf_samples), if not average (?)
    # relative_proximity = proximity / nn_distances.mean()
    relative_proximity = proximity_list.mean() / nn_distance_list.mean()

    return relative_proximity


"""
counterfactual model needed
"""


# def find_best_lr(
#     classifier,
#     X_samples,
#     pred_labels,
#     autoencoder=None,
#     encoder=None,
#     decoder=None,
#     lr_list=[0.001, 0.0001],
#     pred_margin_weight=1.0,
#     step_weights=None,
#     random_state=None,
#     padding_size=0,
#     target_prob=0.5,
# ):
#     # Find the best alpha for vanilla LatentCF
#     best_cf_model, best_cf_samples, best_cf_embeddings = None, None, None
#     best_losses, best_valid_frac, best_lr, best_proxi_score = 0, -1, 0, np.inf
#
#     for lr in lr_list:
#         print(f"======================== CF search started, with lr={lr}.")
#         # Fit the LatentCF model
#         # TODO: fix the class name here: ModifiedLatentCF or GuidedLatentCF? from _guided or _composite?
#         if encoder and decoder:
#             cf_model = ModifiedLatentCF(
#                 probability=target_prob,
#                 only_encoder=encoder,
#                 only_decoder=decoder,
#                 optimizer=tf.optimizers.Adam(learning_rate=lr),
#                 pred_margin_weight=pred_margin_weight,
#                 step_weights=step_weights,
#                 random_state=random_state,
#             )
#         else:
#             cf_model = ModifiedLatentCF(
#                 probability=target_prob,
#                 autoencoder=autoencoder,
#                 optimizer=tf.optimizers.Adam(learning_rate=lr),
#                 pred_margin_weight=pred_margin_weight,
#                 step_weights=step_weights,
#                 random_state=random_state,
#             )
#
#         cf_model.fit(classifier)
#
#         if encoder and decoder:
#             cf_embeddings, losses, _ = cf_model.transform(X_samples, pred_labels)
#             cf_samples = decoder.predict(cf_embeddings)
#             # predicted probabilities of CFs
#             z_pred = classifier.predict(cf_embeddings)
#             cf_pred_labels = np.argmax(z_pred, axis=1)
#         else:
#             cf_samples, losses, _ = cf_model.transform(X_samples, pred_labels)
#             # predicted probabilities of CFs
#             z_pred = classifier.predict(cf_samples)
#             cf_pred_labels = np.argmax(z_pred, axis=1)
#
#         valid_frac = validity_score(pred_labels, cf_pred_labels)
#         proxi_score = euclidean_distance(
#             remove_paddings(X_samples, padding_size),
#             remove_paddings(cf_samples, padding_size),
#         )
#
#         # uncomment for debugging
#         print(f"lr={lr} finished. Validity: {valid_frac}, proximity: {proxi_score}.")
#
#         # TODO: fix (padding) dimensions of `lof_estimator` and `nn_estimator` during training, for debugging
#         # proxi_score, valid_frac, lof_score, rp_score, cost_mean, cost_std = evaluate(
#         #     X_pred_neg=X_samples,
#         #     cf_samples=cf_samples,
#         #     z_pred=z_pred,
#         #     n_timesteps=_,
#         #     lof_estimator=lof_estimator,
#         #     nn_estimator=nn_estimator,
#         # )
#
#         # if valid_frac >= best_valid_frac and proxi_score <= best_proxi_score:
#         # if valid_frac >= best_valid_frac:
#         #     best_cf_model, best_cf_samples = cf_model, cf_samples
#         #     best_losses, best_lr, best_valid_frac = losses, lr, valid_frac
#         #     if encoder and decoder:
#         #         best_cf_embeddings = cf_embeddings
#         if valid_frac > best_valid_frac or (valid_frac == best_valid_frac and proxi_score < best_proxi_score):
#             best_cf_model, best_cf_samples = cf_model, cf_samples
#             best_losses, best_lr, best_valid_frac, best_proxi_score = losses, lr, valid_frac, proxi_score
#             if encoder and decoder:
#                 best_cf_embeddings = cf_embeddings
#
#     return best_lr, best_cf_model, best_cf_samples, best_cf_embeddings

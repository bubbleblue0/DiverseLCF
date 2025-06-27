import os
import logging
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
# Optional: For reproducibility
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import logging
logging.getLogger("numba").setLevel(logging.WARNING)


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from help_functions import (
    ResultWriter, conditional_pad, remove_paddings, evaluate_and_save_valid_cfs, find_best_lr,
    reset_seeds, fit_evaluation_models, readUCR
)
from keras_models import *
from _guided_glacier import get_global_weights

# Enable memory growth for all GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def prepare_data(dataset_name, logger):
    X_train, X_test, y_train, y_test = readUCR(dataset_name)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Encode labels to be zero-indexed
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    nb_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    print("Number of classes:", nb_classes)
    y_train_class, y_val_class, y_test_class = y_train.copy(), y_val.copy(), y_test.copy()
    print("Unique values in y_train:", np.unique(y_train))
    print("Number of classes (nb_classes):", nb_classes)
    y_train, y_val, y_test = (
        to_categorical(y_train, nb_classes),
        to_categorical(y_val, nb_classes),
        to_categorical(y_test, nb_classes),
    )

    n_training, n_timesteps = X_train.shape
    n_features = 1

    X_train = X_train.reshape((n_training, n_timesteps, n_features))
    X_val = X_val.reshape((X_val.shape[0], n_timesteps, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    X_train_pad, pad_size = conditional_pad(X_train)
    X_val_pad, _ = conditional_pad(X_val)
    X_test_pad, _ = conditional_pad(X_test)

    logger.info(
        f"Data pre-processed, original #timesteps={n_timesteps}, padded #timesteps={X_train_pad.shape[1]}."
    )

    return (X_train_pad, X_val_pad, X_test_pad, X_train, X_test, y_train, y_val,
            y_test, y_train_class, y_val_class, y_test_class, pad_size, nb_classes,
            n_timesteps, X_train.shape[1])

def train_classifier(X_train, y_train, X_val, y_val, classifier, model_path):
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=50, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=30, min_lr=0.00001
    )

    batch_size = 16
    nb_epochs = 2000

    mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))
    classifier.fit(
        X_train, y_train, epochs=nb_epochs, batch_size=mini_batch_size, shuffle=True,
        validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stop], verbose=True
    )
    classifier.save(model_path)

def run():
    parser = ArgumentParser(description="Run LatentCF evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default= "Glacier_local.csv")
    parser.add_argument("--w-type", type=str, default="local")
    parser.add_argument("--w-value", type=float, default=0.5)
    parser.add_argument("--tau-value", type=float, default=0.5)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--lr-list", nargs="+", type=float, default=[None, None, None])
    args = parser.parse_args()

    model_dir = os.path.join("models", args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.model_path is None:
        args.model_path = os.path.join(model_dir, "classifier_model.h5")

    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    result_writer = ResultWriter(file_name=args.output, dataset_name=args.dataset)
    if not os.path.isfile(args.output):
        result_writer.write_head()

    data = prepare_data(args.dataset, logger)
    (X_train, X_val, X_test, X_train_unpad, X_test_unpad, y_train, y_val, y_test,
     y_train_class, _, y_test_class, pad_size, nb_classes, _, _) = data

    reset_seeds()
    if os.path.exists(args.model_path):
        classifier = keras.models.load_model(args.model_path)
        logger.info("Loaded existing model from file.")
    else:
        # classifier = LSTMFCNClassifier(X_train.shape[1], 1, n_output=2, n_LSTM_cells=args.n_lstmcells)
        classifier = Classifier_FCN(X_train.shape[1:], nb_classes)
        classifier.compile(optimizer=keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        logger.info("Training FCN classifier")
        train_classifier(X_train, y_train, X_val, y_val, classifier, args.model_path)

    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    acc = balanced_accuracy_score(np.argmax(y_test, axis=1), y_pred)
    logger.info(f"Classifier FCN trained. Test accuracy: {acc}")

    probs = classifier.predict(X_test)
    if probs.shape[1] == 2:
        # Binary classification
        target_labels = 1 - np.argmax(probs, axis=1)
    else:
        # Multi-class classification
        target_labels = np.argsort(probs, axis=1)[:, -2]

    if args.w_type == "global":
        step_weights = get_global_weights(X_train, y_train_class, classifier, random_state=42)
    elif args.w_type == "uniform":
        step_weights = np.ones((1, X_train.shape[1], 1))
    elif args.w_type == "local":
        step_weights = "local"
    elif args.w_type == "unconstrained":
        step_weights = np.zeros((1, X_train.shape[1], 1))
    else:
        raise NotImplementedError("Invalid --w-type")

    for idx, label in enumerate(["Autoencoder","No autoencoder"]):
        auto = [Autoencoder, None][idx]

        lr = args.lr_list
        if auto:

            model = auto(X_train.shape[1], 1)
            learning_rate = 0.001
            model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")

            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=30, min_lr=0.000001
            )

            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0.00001, patience=50, restore_best_weights=True
            )

            batch_size = 16
            nb_epochs = 1500
            mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

            model.fit(X_train, X_train, epochs=nb_epochs, batch_size=mini_batch_size, shuffle=True,
                      validation_data=(X_val, X_val),
                      callbacks=[reduce_lr, early_stopping],
                      verbose=True)

            logger.info(f"Autoencoder [{label}] trained.")

        else:
            model = None

        best_lr, _, best_cfs, _ = find_best_lr(
            classifier, X_samples=X_test[:100] if len(X_test) > 150 else X_test,
            target_labels=target_labels[:100] if len(target_labels) > 150 else target_labels,
            autoencoder=model, lr_list=lr, pred_margin_weight=args.w_value, step_weights=step_weights,
            random_state=42, padding_size=pad_size, target_prob=args.tau_value
        )

        z_pred = classifier.predict(best_cfs)
        cf_labels = np.argmax(z_pred, axis=1)
        best_cfs = remove_paddings(best_cfs, pad_size)

        if not os.path.exists("results/cfs_glacier"):
            os.makedirs("results/cfs_glacier")
        np.save(os.path.join("results/cfs_glacier", f"cf_{args.dataset}_{label}_local.npy"), best_cfs)
        np.save(os.path.join("results/cfs_glacier", f"X_test_{args.dataset}.npy"), np.squeeze(X_test_unpad))

        # evaluate_res = evaluate(
        #     np.squeeze(X_train_unpad), np.squeeze(X_test_unpad[:100] if len(X_test_unpad) > 150 else X_test_unpad),
        #     best_cfs, target_labels[:100] if len(target_labels) > 150 else target_labels, cf_labels,
        # )

        evaluate_res = evaluate_and_save_valid_cfs(
            np.squeeze(X_train_unpad), np.squeeze(X_test_unpad[:100] if len(X_test_unpad) > 150 else X_test_unpad),
            best_cfs, target_labels[:100] if len(target_labels) > 150 else target_labels, cf_labels,
            save_dir="results/cfs_glacier",
            dataset_name=args.dataset,
        )

        result_writer.write_result(
            1, label, acc,
            np.min(model.history.history["val_loss"]) if model else 0,
            best_lr, evaluate_res,
            pred_margin_weight=args.w_value,
            step_weight_type=args.w_type,
            threshold_tau=args.tau_value
        )
        logger.info(f"Done for CF search [{label}], pred_margin_weight={args.w_value}.")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    run()

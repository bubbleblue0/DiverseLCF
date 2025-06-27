import os
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import pdist

from help_functions0 import (
    ResultWriter, conditional_pad, remove_paddings, evaluate, evaluate_and_save_valid, reset_seeds, readUCR
)
from keras_models import *
from _guided_Dice import DiceRandomInitCF

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

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    nb_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    y_train_class, y_val_class, y_test_class = y_train.copy(), y_val.copy(), y_test.copy()

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

    X_train_pad, pad_size = conditional_pad(X_train)
    X_val_pad, _ = conditional_pad(X_val)
    X_test_pad, _ = conditional_pad(X_test)

    logger.info(f"Padded timesteps: {X_train_pad.shape[1]}")

    return (X_train_pad, X_val_pad, X_test_pad, X_train, X_test, y_train, y_val,
            y_test, y_train_class, y_val_class, y_test_class, pad_size, nb_classes,
            n_timesteps, X_train.shape[1])

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default= "Dice.csv")
    parser.add_argument("--pred_lambda", type=float, default=1.0)
    parser.add_argument("--div_lambda", type=float, default=1)
    parser.add_argument("--diversity-mode", type=str, default="dpp")
    parser.add_argument("--proxi_lambda", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--probability", type=float, default=1.0)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--n-counterfactuals", type=int, default=1)
    args = parser.parse_args()

    model_dir = os.path.join("models", args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.model_path is None:
        args.model_path = os.path.join(model_dir, "classifier_model.h5")

    logger = logging.getLogger(__name__)
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
        # Build and compile FCN classifier
        classifier = Classifier_FCN(X_train.shape[1:], nb_classes)
        classifier.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=100, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=30, min_lr=0.00001)

        # Training setup
        batch_size = 16
        nb_epochs = 2000
        mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

        # Train and save history
        history = classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=nb_epochs,
            batch_size=mini_batch_size,
            callbacks=[reduce_lr, early_stop],
            verbose=True
        )

        # Save model
        classifier.save(args.model_path)

        # === Plot learning curve ===
        plt.figure(figsize=(10, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("FCN Accuracy")
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("FCN Loss")
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(os.path.dirname(args.model_path), f"fcn_learning_curve_{args.dataset}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Learning curve saved to: {plot_path}")
    X_test = X_test[:100] if len(X_test) > 150 else X_test
    y_test = y_test[:100] if len(y_test) > 150 else y_test

    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    acc = balanced_accuracy_score(np.argmax(y_test, axis=1), y_pred)
    logger.info(f"Test accuracy: {acc}")

    probs = classifier.predict(X_test)
    if probs.shape[1] == 2:
        # Binary classification
        target_labels = 1 - np.argmax(probs, axis=1)
    else:
        # Multi-class classification
        target_labels = np.argsort(probs, axis=1)[:, -2]


    # === Prepare feature min and max for DiCE initialization ===
    X_train_unpad_flat = np.squeeze(X_train_unpad)  # (N, T)
    feature_min = np.min(X_train_unpad_flat, axis=0)
    feature_max = np.max(X_train_unpad_flat, axis=0)

    # Pad feature_min and feature_max to match the padded time series length
    feature_min = np.pad(feature_min, (0, pad_size), mode="constant", constant_values=(0,))
    feature_max = np.pad(feature_max, (0, pad_size), mode="constant", constant_values=(0,))

    # === Create the DiCE CF model ===
    cf_model = DiceRandomInitCF(
        feature_min=feature_min,
        feature_max=feature_max,
        probability=args.probability,
        pred_margin_weight=args.pred_lambda,
        prox_weight=args.proxi_lambda,
        diversity_weight=args.div_lambda,
        learning_rate=args.learning_rate,
        n_counterfactuals=args.n_counterfactuals,
        random_state=42,
        diversity_mode=args.diversity_mode,
    )

    cf_model.fit(classifier)

    import time
    start_time = time.time()
    cfs, _ = cf_model.transform(X_test, target_labels)
    end_time = time.time()
    transform_time = end_time - start_time

    print("cfs shape", cfs.shape)

    # (N, n_cfs, time_steps) -> (N*n_cfs, time_steps)， cfs shape (28, 3, 288, 1)
    num_samples, n_cfs, time_steps, dim = cfs.shape
    cfs_reshaped = cfs.reshape(num_samples * n_cfs, time_steps, dim)

    cf_labels = np.argmax(classifier.predict(cfs_reshaped), axis=1)

    X_test_repeat = np.repeat(X_test, args.n_counterfactuals, axis=0)
    target_labels_repeat = np.repeat(target_labels, args.n_counterfactuals, axis=0)

    # remove paddings
    cfs = remove_paddings(cfs_reshaped, pad_size)
    X_test_repeat = remove_paddings(X_test_repeat, pad_size)

    # evaluate_res = evaluate(
    #     np.squeeze(X_train_unpad),
    #     np.squeeze(X_test_repeat),
    #     cfs,
    #     target_labels_repeat,
    #     cf_labels,
    #     num_div_cfs=args.n_counterfactuals,
    # )

    evaluate_res = evaluate_and_save_valid(
        X_train=np.squeeze(X_train_unpad),
        X_test_repeat=np.squeeze(X_test_repeat),
        cfs=cfs,
        target_labels=target_labels,
        cf_labels=cf_labels,
        num_div_cfs=args.n_counterfactuals,
        save_dir="results/cfs_dice_n_" + str(args.n_counterfactuals),
        dataset_name=args.dataset,

    )

    print(cfs.shape, np.squeeze(X_test_repeat).shape)
    if not os.path.exists("results/cfs_dice_n_" + str(args.n_counterfactuals)):
        os.makedirs("results/cfs_dice_n_" + str(args.n_counterfactuals))
    np.save(os.path.join("results/cfs_dice_n_" + str(args.n_counterfactuals), f"cf_{args.dataset}.npy"), cfs)
    np.save(os.path.join("results/cfs_dice_n_" + str(args.n_counterfactuals), f"X_test_{args.dataset}.npy"),
            np.squeeze(X_test_repeat))

    # Save the generated CFs
    # if not os.path.exists("results/cfs_dice"):
    #     os.makedirs("results/cfs_dice")
    # np.save(os.path.join("results/cfs_dice", f"cf_{args.dataset}.npy"), cfs)
    # np.save(os.path.join("results/cfs_dice", f"X_test_{args.dataset}.npy"), np.squeeze(X_test_repeat))

    # For min_val_loss you can simply set None because no VAE now
    min_val_loss = None

    # Write final results
    result_writer.write_result(
        2, "DiCE", acc,
        min_val_loss,
        best_lr="adam-scheduler",
        evaluate_res=evaluate_res,
        pred_margin_weight=args.pred_lambda,
        diversity_weight=args.div_lambda,
        prox_weight=args.proxi_lambda,
        diversity_mode=args.diversity_mode,
        learning_rate=args.learning_rate,
        probability=args.probability,
        process_time=transform_time,
        num_div_cfs=args.n_counterfactuals,
    )

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    run()

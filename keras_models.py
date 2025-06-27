from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import tensorflow as tf

"""
1dCNN models
"""


def Autoencoder(n_timesteps, n_features):
    # Define encoder and decoder structure
    def Encoder(input):
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )(input)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )(x)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        return x

    def Decoder(input):
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )(input)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )(x)
        # x = keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=1, kernel_size=3, activation="linear", padding="same"
        )(x)
        return x

    # Define the AE model
    orig_input = keras.Input(shape=(n_timesteps, n_features))
    autoencoder = keras.Model(inputs=orig_input, outputs=Decoder(Encoder(orig_input)))

    return autoencoder


def SparseAutoencoder(n_timesteps, n_features, sparsity_weight=1e-4, sparsity_target=0.05):
    # Define encoder and decoder structure
    def Encoder(input):
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same")(input)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same")(x)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)

        # Adding sparsity constraint on the encoder layer
        x = keras.layers.Dense(
            16,
            activation="relu",
            activity_regularizer=regularizers.l1(sparsity_weight)  # Sparse regularizer
        )(x)

        return x

    def Decoder(input):
        x = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same")(input)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same")(x)
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.Conv1D(
            filters=1, kernel_size=3, activation="linear", padding="same")(x)
        return x

    # Define the AE model
    orig_input = keras.Input(shape=(n_timesteps, n_features))
    encoded = Encoder(orig_input)
    decoded = Decoder(encoded)
    autoencoder = keras.Model(inputs=orig_input, outputs=decoded)

    return autoencoder


# def VAE(n_timesteps, n_features, latent_dim=16):
#     # Encoder
#     encoder_inputs = keras.Input(shape=(n_timesteps, n_features))
#     x = layers.Conv1D(64, 3, activation="relu", padding="same")(encoder_inputs)
#     x = layers.MaxPooling1D(2, padding="same")(x)
#     x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
#     x = layers.MaxPooling1D(2, padding="same")(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(64, activation="relu")(x)
#     z_mean = layers.Dense(latent_dim, name="z_mean")(x)
#     z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#     z = layers.Lambda(Sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
#
#     encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#
#     # Decoder
#     latent_inputs = keras.Input(shape=(latent_dim,))
#     x = layers.Dense((n_timesteps // 4) * 32, activation="relu")(latent_inputs)
#     x = layers.Reshape((n_timesteps // 4, 32))(x)
#     x = layers.UpSampling1D(2)(x)
#     x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
#     x = layers.UpSampling1D(2)(x)
#     decoder_outputs = layers.Conv1D(1, 3, activation="linear", padding="same")(x)
#
#     decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#
#     # VAE model
#     outputs = decoder(z)
#     vae = keras.Model(encoder_inputs, outputs, name="vae")
#
#     # Add KL loss
#     reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(encoder_inputs, outputs))
#     reconstruction_loss *= n_timesteps * n_features
#     kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#     vae.add_loss(reconstruction_loss + kl_loss)
#
#     return vae, encoder, decoder


def Sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(n_timesteps, n_features, latent_dim=16):
    # ----- Encoder -----
    encoder_inputs = keras.Input(shape=(n_timesteps, n_features), name="encoder_input")
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(encoder_inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(Sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # ----- Decoder -----
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense((n_timesteps // 4) * 32, activation="relu")(latent_inputs)
    x = layers.Reshape((n_timesteps // 4, 32))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    decoder_outputs = layers.Conv1D(n_features, 3, activation="linear", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # ----- Full VAE Model -----
    vae_outputs = decoder(z)
    vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

    # ----- Add VAE loss -----
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(encoder_inputs, vae_outputs))
    reconstruction_loss *= n_timesteps * n_features

    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )

    vae.add_loss(reconstruction_loss + kl_loss)

    return vae, encoder, decoder

def build_split_latent_vae(n_timesteps, n_features, latent_dim=16, class_dim=8, num_classes=2,
                            beta_class=0.01, beta_general=1.0, lambda_cls=1.0):
    general_dim = latent_dim - class_dim

    # ----- Encoder -----
    encoder_inputs = keras.Input(shape=(n_timesteps, n_features), name="encoder_input")
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(encoder_inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)

    # Class-specific latent part
    z_class_mean = layers.Dense(class_dim, name="z_class_mean")(x)
    z_class_log_var = layers.Dense(class_dim, name="z_class_log_var")(x)
    z_class = layers.Lambda(Sampling, name="z_class")([z_class_mean, z_class_log_var])

    # General latent part
    z_general_mean = layers.Dense(general_dim, name="z_general_mean")(x)
    z_general_log_var = layers.Dense(general_dim, name="z_general_log_var")(x)
    z_general = layers.Lambda(Sampling, name="z_general")([z_general_mean, z_general_log_var])

    # Concatenate
    z = layers.Concatenate(name="z")([z_class, z_general])

    # Class prediction head (only supervised part)
    class_output = layers.Dense(num_classes, activation="softmax", name="class_output")(z_class)

    encoder = keras.Model(encoder_inputs, [z_class_mean, z_class_log_var, z_general_mean, z_general_log_var, z, class_output], name="encoder")

    # ----- Decoder -----
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense((n_timesteps // 4) * 32, activation="relu")(latent_inputs)
    x = layers.Reshape((n_timesteps // 4, 32))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    decoder_outputs = layers.Conv1D(n_features, 3, activation="linear", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    # ----- Full VAE Model -----
    vae_outputs = decoder(z)
    vae = keras.Model(inputs=encoder_inputs, outputs=[vae_outputs, class_output], name="vae")

    # ----- Custom loss function -----
    def vae_loss(x, x_recon, z_class_mean, z_class_log_var, z_general_mean, z_general_log_var, y_true, y_pred):
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_recon)) * n_timesteps * n_features

        kl_class = -0.5 * tf.reduce_mean(1 + z_class_log_var - tf.square(z_class_mean) - tf.exp(z_class_log_var))
        kl_general = -0.5 * tf.reduce_mean(1 + z_general_log_var - tf.square(z_general_mean) - tf.exp(z_general_log_var))

        class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

        return recon_loss + beta_general * kl_general + beta_class * kl_class + lambda_cls * class_loss

    return vae, encoder, decoder, vae_loss

def Classifier(
    n_timesteps, n_features, n_conv_layers=1, add_dense_layer=True, n_output=1
):
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
    inputs = keras.Input(shape=(n_timesteps, n_features), dtype="float32")

    if add_dense_layer:
        x = keras.layers.Dense(128)(inputs)
    else:
        x = inputs

    for i in range(n_conv_layers):
        x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Flatten()(x)

    if n_output >= 2:
        outputs = keras.layers.Dense(n_output, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    classifier = keras.models.Model(inputs=inputs, outputs=outputs)

    return classifier


"""
LSTM models
"""


def AutoencoderLSTM(n_timesteps, n_features):
    # Define encoder and decoder structure
    # structure from medium post: https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    def EncoderLSTM(input):
        # x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(input)
        x = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(input)
        # encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(x)
        encoded = keras.layers.LSTM(32, activation="tanh", return_sequences=False)(x)
        return encoded

    def DecoderLSTM(encoded):
        x = keras.layers.RepeatVector(n_timesteps)(encoded)
        # x = keras.layers.LSTM(32, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(32, activation="tanh", return_sequences=True)(x)
        # x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(x)
        decoded = keras.layers.TimeDistributed(
            keras.layers.Dense(n_features, activation="sigmoid")
        )(x)
        return decoded

    # Define the AE model
    orig_input2 = keras.Input(shape=(n_timesteps, n_features))

    autoencoder2 = keras.Model(
        inputs=orig_input2, outputs=DecoderLSTM(EncoderLSTM(orig_input2))
    )

    return autoencoder2


def ClassifierLSTM(n_timesteps, n_features, extra_lstm_layer=True, n_output=1):
    # Define the model structure - only LSTM layers
    # https://www.kaggle.com/szaitseff/classification-of-time-series-with-lstm-rnn
    inputs = keras.Input(shape=(n_timesteps, n_features), dtype="float32")
    if extra_lstm_layer:
        x = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(
            inputs
        )  # set return_sequences true to feed next LSTM layer
    else:
        x = keras.layers.LSTM(32, activation="tanh", return_sequences=False)(
            inputs
        )  # set return_sequences false to feed dense layer directly
    x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.LSTM(32, activation='tanh', return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    if extra_lstm_layer:
        x = keras.layers.LSTM(16, activation="tanh", return_sequences=False)(x)
        x = keras.layers.BatchNormalization()(x)

    if n_output >= 2:
        outputs = keras.layers.Dense(n_output, activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    classifier2 = keras.Model(inputs, outputs)

    return classifier2


"""
composite autoencoder
"""


def CompositeAutoencoder(n_timesteps, n_features, n_output_classifier=1):
    # https://machinelearningmastery.com/lstm-autoencoders/
    # https://stackoverflow.com/questions/48603328/how-do-i-split-an-convolutional-autoencoder

    # Composite 1dCNN Autoencoder
    # encoder:
    inputs = keras.layers.Input(shape=(n_timesteps, n_features))
    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, activation="relu", padding="same"
    )(inputs)
    x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
    x = keras.layers.Conv1D(
        filters=32, kernel_size=3, activation="relu", padding="same"
    )(x)
    encoded = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)

    encoder3 = keras.models.Model(inputs, encoded)

    _, encoded_dim1, encoded_dim2 = encoder3.layers[-1].output_shape

    # decoder
    decoder_inputs = keras.layers.Input(shape=(encoded_dim1, encoded_dim2))
    x = keras.layers.Conv1D(
        filters=32, kernel_size=3, activation="relu", padding="same"
    )(decoder_inputs)
    x = keras.layers.UpSampling1D(size=2)(x)
    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, activation="relu", padding="same"
    )(x)
    # x = keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu")(x)
    x = keras.layers.UpSampling1D(size=2)(x)
    decoded = keras.layers.Conv1D(
        filters=1, kernel_size=3, activation="linear", padding="same"
    )(x)

    decoder3 = keras.models.Model(decoder_inputs, decoded, name="only_decoder")

    # classifier based on previous encoder
    classifier_inputs = keras.layers.Input(shape=(encoded_dim1, encoded_dim2))
    x = keras.layers.Conv1D(
        filters=16, kernel_size=3, activation="relu", padding="same"
    )(classifier_inputs)
    x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation="relu")(x)
    if n_output_classifier >= 2:
        classifier_outputs = keras.layers.Dense(
            n_output_classifier, activation="softmax"
        )(x)
    else:
        classifier_outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    classifier3 = keras.models.Model(
        classifier_inputs, classifier_outputs, name="only_classifier"
    )

    # composite autoencoder
    encoded = encoder3(inputs)
    decoded = decoder3(encoded)
    classified = classifier3(encoded)

    composite_autoencoder = keras.models.Model(inputs, [decoded, classified])

    return composite_autoencoder, encoder3, decoder3, classifier3


def LSTMFCNClassifier(n_timesteps, n_features, n_output=2, n_LSTM_cells=8):
    # https://github.com/titu1994/LSTM-FCN/blob/master/hyperparameter_search.py
    inputs = keras.Input(shape=(n_timesteps, n_features), dtype="float32")

    x = keras.layers.LSTM(units=n_LSTM_cells)(inputs)
    x = keras.layers.Dropout(rate=0.8)(x)

    y = keras.layers.Permute((2, 1))(inputs)
    y = keras.layers.Conv1D(128, 8, padding="same", kernel_initializer="he_uniform")(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    y = keras.layers.Conv1D(256, 5, padding="same", kernel_initializer="he_uniform")(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    y = keras.layers.Conv1D(128, 3, padding="same", kernel_initializer="he_uniform")(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)

    y = keras.layers.GlobalAveragePooling1D()(y)

    x = keras.layers.concatenate([x, y])

    outputs = keras.layers.Dense(n_output, activation="softmax")(x)

    classifier = keras.models.Model(inputs=inputs, outputs=outputs)

    return classifier


def Classifier_FCN(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation="relu")(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation("relu")(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation("relu")(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation="softmax")(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

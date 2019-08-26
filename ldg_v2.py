from keras.layers import Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Reshape, Concatenate, Dropout
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
from keras.optimizers import Adam
from keras import initializers
import numpy as np
import os
from mnist import MNIST

label_str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
lebel_dict = {i: v for i, v in enumerate(label_str)}
n_label = len(label_str)

img_rows = 28
img_cols = 28
img_channel = 1
orig_dimension = img_rows * img_cols
image_shape = (img_rows, img_cols, img_channel)
batch_size = 128
epochs = 2
latent_dim = 10
learning_rate = 0.001
w_init = initializers.random_normal(stddev=0.02)
gamma_init = initializers.random_normal(mean=1.0, stddev=0.02)

def load_preprocess_EMNIST():
    emndata = MNIST('emnist_data')
    X_train, y_train = emndata.load('emnist_data/emnist-byclass-train-images-idx3-ubyte',
                                    'emnist_data/emnist-byclass-train-labels-idx1-ubyte')
    X_test, y_test = emndata.load('emnist_data/emnist-byclass-test-images-idx3-ubyte',
                                  'emnist_data/emnist-byclass-test-labels-idx1-ubyte')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.astype('float32') / (255 / 2) - 1  # for tanh
    X_test = X_test.astype('float32') / (255 / 2) - 1
    y_train = to_categorical(y_train, n_label)
    y_test = to_categorical(y_test, n_label)

    return X_train, y_train, X_test, y_test


def sampling(arg):
    z_mean = arg[0]
    z_log_var = arg[1]
#    arg = [z_mean, z_log_var]
    dim = K.int_shape(z_mean)[1]
    # reparameterization trick
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], dim), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae():
    img_inputs = Input(shape=(orig_dimension,), name='image_input')
    label_inputs = Input(shape=(n_label,), name='label_input')
    encoder_inputs = Concatenate()([img_inputs, label_inputs])

    x = Dense(orig_dimension, kernel_initializer=w_init,activation='relu')(encoder_inputs)
    x = Reshape(image_shape)(x)
    x = Conv2D(16, 3, strides=1, padding='same', kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same', kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, strides=2, padding='same', kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)
    before_flatten_shape = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(128, kernel_initializer=w_init, activation='relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model([img_inputs, label_inputs], [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='latent_inputs')
    decoder_inputs = Concatenate()([latent_inputs, label_inputs])

    x = Dense(128, kernel_initializer=w_init,
              activation='relu')(decoder_inputs)
    x = Dense(before_flatten_shape[1] * before_flatten_shape[2] * before_flatten_shape[3], activation='relu', kernel_initializer=w_init)(x)
    x = Reshape((before_flatten_shape[1], before_flatten_shape[2], before_flatten_shape[3]))(x)

    x = Conv2DTranspose(64, 3, strides=1, padding='same',kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, 3, strides=2, padding='same',kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(16, 3, strides=1, padding='same',kernel_initializer=w_init)(x)
    x = BatchNormalization(gamma_initializer=gamma_init)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(img_channel, 3, activation='tanh',padding='same', kernel_initializer=w_init)(x)
    outputs = Flatten()(x)

    decoder = Model([latent_inputs, label_inputs], outputs, name='decoder')

    outputs = decoder([encoder([img_inputs, label_inputs])[2], label_inputs])
    vae = Model([img_inputs, label_inputs], outputs)

    beta = 1  # 1 --> regular VAE
    reconstruction_loss = mse(img_inputs, outputs)
    reconstruction_loss *= img_rows * img_cols
    kl_loss = -0.5 * beta * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    # in this case, we set y value to NONE during the fit
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(lr=learning_rate))

    return vae


def train_or_load_weights(flag, weight_name=None):
    saving_folder = 'best_weight_ldg_v2_conv-cvae'
    if flag == 'train':
        file_name = 'ldg_v2_conv-cvae'
        os.makedirs(saving_folder, exist_ok=True)
        best_model_weight_path = os.path.join(
            saving_folder, file_name + '-best-wiehgts' + '-{epoch:03d}-{loss:.3f}-{val_loss:.3f}.h5')
        save_best_model = ModelCheckpoint(best_model_weight_path, monitor='val_loss',
                                          verbose=0, save_weights_only=True, save_best_only=True, mode='min')
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

        hist = vae.fit([X_train, y_train],
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=([X_test, y_test], None),
                       verbose=1,
                       callbacks=[save_best_model, learning_rate_reduction])  # ,NEpochPrinter])
        return hist

    if flag == 'load':
        # load all the weights for encoder and decoder when loading for vae
        vae.load_weights(os.path.join(saving_folder, weight_name))

def load_weights_vae(weight_name=None):
    # load all the weights for encoder and decoder when loading for vae
    vae.load_weights(os.path.join(saving_folder, weight_name))

if __name__ == '__main__':    
    X_train, y_train, X_test, y_test = load_preprocess_EMNIST()
    vae = build_vae()
    train_or_load_weights(flag='train')
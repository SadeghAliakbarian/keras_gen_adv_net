import matplotlib as mpl
mpl.use('Agg')

import keras.backend as K
from backend import unpack_assignment, variable_key
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation,Lambda
from keras.models import Model
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist

import pandas as pd
import numpy as np

from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix
from adversarial_utils import simple_gan, gan_targets, normal_latent_sampling

from keras.layers import Dense, BatchNormalization, Convolution2D
from image_grid_callback import ImageGridCallback
from adversarial_model import AdversarialModel
from adversarial_optimizers import AdversarialOptimizerSimultaneous

from six import iteritems

IN_CH = 3
ROWS = 256
COLS = 256

def leaky_relu(x):
    return K.relu(x, 0.2)

# EDIT: The input to the generator should be the conditioned image+noise

def model_generator():
    g_input = Input((IN_CH+1, ROWS, COLS))
    e1 = BatchNormalization(mode=2)(g_input)
    e1 = Convolution2D(64, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e1)
    e1 = BatchNormalization(mode=2)(e1)
    e2 = Convolution2D(128, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e1)
    e2 = BatchNormalization(mode=2)(e2)
    e3 = Convolution2D(256, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e2)
    e3 = BatchNormalization(mode=2)(e3)    
    e4 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e3)
    e4 = BatchNormalization(mode=2)(e4)
    e5 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e4)
    e5 = BatchNormalization(mode=2)(e5)
    e6 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e5)
    e6 = BatchNormalization(mode=2)(e6)  
    e7 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e6)
    e7 = BatchNormalization(mode=2)(e7)
    e8 = Convolution2D(512, 4, 4, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(e7)
    e8 = BatchNormalization(mode=2)(e8)

    d1 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 2, 2), border_mode='same')(e8)
    d1 = merge([d1, e7], mode='concat', concat_axis=1)
    d1 = BatchNormalization(mode=2)(d1)
    d2 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 4, 4), border_mode='same')(d1)
    d2 = merge([d2, e6], mode='concat', concat_axis=1)
    d2 = BatchNormalization(mode=2)(d2)
    d3 = Dropout(0.2)(d2)
    d3 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 8, 8), border_mode='same')(d3)
    d3 = merge([d3, e5], mode='concat', concat_axis=1)
    d3 = BatchNormalization(mode=2)(d3)    
    d4 = Dropout(0.2)(d3)
    d4 = Deconvolution2D(512, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 512, 16, 16), border_mode='same')(d4)
    d4 = merge([d4, e4], mode='concat', concat_axis=1)
    d4 = BatchNormalization(mode=2)(d4)        
    d5 = Dropout(0.2)(d4)
    d5 = Deconvolution2D(256, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 256, 32, 32), border_mode='same')(d5) 
    d5 = merge([d5, e3], mode='concat', concat_axis=1)
    d5 = BatchNormalization(mode=2)(d5)    
    d6 = Dropout(0.2)(d5)
    d6 = Deconvolution2D(128, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 128, 64, 64), border_mode='same')(d6)
    d6 = merge([d6, e2], mode='concat', concat_axis=1)    
    d6 = BatchNormalization(mode=2)(d6)       
    d7 = Dropout(0.2)(d6)
    d7 = Deconvolution2D(64, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 64,128, 128), border_mode='same')(d7)        
    d7 = merge([d7,e1], mode='concat', concat_axis=1)        
    d7 = BatchNormalization(mode=2)(d7)
    d8 = Deconvolution2D(3, 5, 5, subsample=(2,2),  activation='relu',init='uniform', output_shape=(None, 3, 256, 256), border_mode='same')(d7)   
    d8 = BatchNormalization(mode=2)(d8)
    d9 = Activation('sigmoid')(d8)

    return Model(input=g_input, output=d9)


# EDIT: The input to the discriminator is now a RGB image (fake or real)
# NOTE: At this time, we don't condition the discriminator on the first frame of the video

def model_discriminator(input_shape=(IN_CH, ROWS, COLS), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    # nch = 128
    H = Convolution2D(int(nch / 2), 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(nch, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)

    return Model(d_input, d_V)


def data_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def data():
    (xtrain, ytrain), (xtest, ytest) = load_data()
    return data_process(xtrain), data_process(xtest)


# EDIT: This function should load the dataset as the first frame and last frame of each video.
# NOTE: The format of the data should follow MNIST dataset.

def load_data():
    pass

def generator_sampler(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = dim_ordering_unfix(generator.predict(zsamples))
        return gen.reshape((10, 10, 28, 28))

    return fun


if __name__ == "__main__":
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    input_shape = (IN_CH, ROWS, COLS)

    # generator (z -> x)
    generator = model_generator()
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape=input_shape)
    # gan (x - > yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()

    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss='binary_crossentropy')

    # train model
    generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",
                                     generator_sampler(latent_dim, generator))

    xtrain, xtest = mnist_data()
    xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
    xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))
    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb], nb_epoch=100,
                        batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv("output/gan_convolutional/history.csv")

    generator.save("output/gan_convolutional/generator.h5")
    discriminator.save("output/gan_convolutional/discriminator.h5")


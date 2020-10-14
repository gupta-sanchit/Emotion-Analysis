from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)


def generateModel(n_mfcc):
    nclass = 6
    inp = Input(shape=(n_mfcc, 105, 1))
    mod = Convolution2D(32, (4, 10), padding="same")(inp)
    mod = BatchNormalization()(mod)
    mod = Activation("relu")(mod)
    mod = MaxPool2D()(mod)
    mod = Dropout(rate=0.2)(mod)

    mod = Convolution2D(32, (4, 10), padding="same")(mod)
    mod = BatchNormalization()(mod)
    mod = Activation("relu")(mod)
    mod = MaxPool2D()(mod)
    mod = Dropout(rate=0.2)(mod)

    mod = Convolution2D(32, (4, 10), padding="same")(mod)
    mod = BatchNormalization()(mod)
    mod = Activation("relu")(mod)
    mod = MaxPool2D()(mod)
    mod = Dropout(rate=0.2)(mod)

    mod = Convolution2D(32, (4, 10), padding="same")(mod)
    mod = BatchNormalization()(mod)
    mod = Activation("relu")(mod)
    mod = MaxPool2D()(mod)
    mod = Dropout(rate=0.2)(mod)

    mod = Flatten()(mod)
    mod = Dense(64)(mod)
    mod = Dropout(rate=0.2)(mod)
    mod = BatchNormalization()(mod)
    mod = Activation("relu")(mod)
    mod = Dropout(rate=0.2)(mod)

    out = Dense(nclass, activation=softmax)(mod)
    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model

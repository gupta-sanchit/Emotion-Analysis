import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def oneHotEncode(train, test):
    encoder = LabelEncoder()
    train = to_categorical(encoder.fit_transform(train))
    test = to_categorical(encoder.fit_transform(test))
    return train, test


def normalize(train, test):
    m = np.mean(train, axis=0)
    sd = np.std(train, axis=0)

    train = (train - m) / sd
    test = (test - m) / sd

    return train, test

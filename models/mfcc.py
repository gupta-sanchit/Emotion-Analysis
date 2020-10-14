import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from models.preProcess import oneHotEncode, normalize
from models.model import generateModel

warnings.simplefilter(action='ignore')

x = np.load("../MFCC/mfcc.npy")
y = pd.read_pickle("../labels.pkl")

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)

x_train, x_test = normalize(x_train, x_test)
y_train, y_test = oneHotEncode(y_train, y_test)

print(x_train.shape, x_test.shape)

data = np.array([x_train, x_test, y_train, y_test])
np.save("mfccTrainingData.npy", data)
# model = generateModel(n_mfcc=30)
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, verbose=1, epochs=100)


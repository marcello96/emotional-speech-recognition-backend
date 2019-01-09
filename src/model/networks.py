import numpy as np
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def dnn(x, y, test_x, test_y, batch_size=32, epochs=1500, validation_split=0.25):
    # prepare labels
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    test_y = np_utils.to_categorical(lb.fit_transform(test_y))

    # build model
    model = Sequential()

    model.add(Dense(128, input_shape=(x.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    accuracy = model.evaluate(test_x, test_y, batch_size=batch_size)

    return model, history, accuracy


def cnn(x, y, test_x, test_y, batch_size=32, epochs=150, validation_split=0.25):
    # expand for input layer
    x = np.expand_dims(x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

    # prepare labels
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    test_y = np_utils.to_categorical(lb.fit_transform(test_y))

    # build model
    model = Sequential()

    model.add(Conv1D(63, 5, padding='same', input_shape=(x.shape[1], 1)))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling1D(pool_size=4))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(y.shape[1]))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    accuracy = model.evaluate(test_x, test_y, batch_size=batch_size)

    return model, history, accuracy

import numpy as np
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def dnn(x, y, val_x, val_y, batch_size=32, epochs=70):
    # prepare labels
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    val_y = np_utils.to_categorical(lb.fit_transform(val_y))

    # build model
    model = Sequential()

    model.add(Dense(256, input_shape=(x.shape[1],)))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(y.shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))

    return model, history, lb.classes_


def cnn(x, y, val_x, val_y, batch_size=32, epochs=70):
    # expand set for cnn
    x = np.expand_dims(x, axis=2)
    val_x = np.expand_dims(val_x, axis=2)
    # prepare labels
    lb = LabelEncoder()
    y = np_utils.to_categorical(lb.fit_transform(y))
    val_y = np_utils.to_categorical(lb.fit_transform(val_y))
    # build model
    model = Sequential()

    model.add(Conv1D(128, 5, padding='same', input_shape=(x.shape[1], 1)))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(MaxPooling1D(pool_size=8))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(26, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(y.shape[1]))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))

    return model, history, lb.classes_

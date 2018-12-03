import numpy as np
import tensorflow as tf

import model.preprocessing as mp
from configuration import read_database_path
from model.networks import dnn, cnn
from model.os import read_model
from model.utils import NetworkType, prepare_prediction_response


def load_models():
    global cnn_model
    cnn_model = read_model('cnn')

    global dnn_model
    dnn_model = read_model('dnn')

    global graph
    graph = tf.get_default_graph()


def train_model(network_type, training_data_rate, batch_size, epochs):
    if network_type is not NetworkType:
        return None, None

    # prepare data
    data = mp.load_files(read_database_path(), mp.map_ravdess_filename_to_label)
    x, y, val_x, val_y = mp.prepare_learning_data(data, training_data_rate)

    if network_type == NetworkType.DNN:
        return dnn(x, y, val_x, val_y, batch_size=batch_size, epochs=epochs)

    elif network_type == NetworkType.CNN:
        return cnn(x, y, val_x, val_y, batch_size=batch_size, epochs=epochs)


def predict_emotion(x, network_type):
    x = np.array(x).reshape(1, -1)
    if network_type == NetworkType.DNN:
        with graph.as_default():
            prediction = dnn_model.predict(x)
    elif network_type == NetworkType.CNN:
        with graph.as_default():
            prediction = cnn_model.predict(np.expand_dims(x, axis=2))
    else:
        return None

    return prepare_prediction_response(prediction[0], mp.MODEL_LABELS)




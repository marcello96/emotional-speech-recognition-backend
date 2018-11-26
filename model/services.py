import numpy as np

from configuration import read_database_path
from model.networks import dnn, cnn
from model.os import read_model
from model.preprocessing import MODEL_LABELS, prepare_learning_data, load_files, map_ravdess_filename_to_label
from model.utils import NetworkType, prepare_prediction_response

# load models
cnn_model = read_model('cnn')
dnn_model = read_model('dnn')


def train_model(network_type, training_data_rate, batch_size, epochs):
    if network_type is not NetworkType:
        return None, None

    # prepare data
    data = load_files(read_database_path(), map_ravdess_filename_to_label)
    x, y, val_x, val_y = prepare_learning_data(data, training_data_rate)

    if network_type == NetworkType.DNN:
        return dnn(x, y, val_x, val_y, batch_size=batch_size, epochs=epochs)

    elif network_type == NetworkType.CNN:
        return cnn(x, y, val_x, val_y, batch_size=batch_size, epochs=epochs)


def predict_emotion(x, network_type):
    if network_type == NetworkType.DNN:
        prediction = dnn_model.predict(x)
    elif network_type == NetworkType.CNN:
        prediction = cnn_model.predict(np.expand_dims(x, axis=2))
    else:
        return None

    return prepare_prediction_response(prediction, MODEL_LABELS)




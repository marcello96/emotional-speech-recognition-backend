import numpy as np

import model.preprocessing as mp
from configuration import read_database_path
from model.networks import dnn, cnn
from model.os import read_model, save_model
from model.utils import NetworkType, prepare_prediction_response


def load_models():
    global cnn_model
    cnn_model = read_model('cnn')
    cnn_model._make_predict_function()

    global dnn_model
    dnn_model = read_model('dnn')
    dnn_model._make_predict_function()


def train_model(network_type, batch_size, epochs, training_data_rate=0.8, validation_split=0.25):
    if type(network_type) is not NetworkType:
        raise ValueError('Wrong value of NetworkType: ', network_type)

    # prepare data
    data = mp.load_files(read_database_path(), mp.map_ravdess_filename_to_label)
    x, y, test_x, test_y = mp.prepare_learning_data(data, training_data_rate)

    if network_type == NetworkType.DNN:
        return dnn(x, y, test_x, test_y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    elif network_type == NetworkType.CNN:
        return cnn(x, y, test_x, test_y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


def predict_emotion(x, network_type):
    x = np.array(x).reshape(1, -1)
    if network_type == NetworkType.DNN:
        prediction = dnn_model.predict(x)
    elif network_type == NetworkType.CNN:
        prediction = cnn_model.predict(np.expand_dims(x, axis=2))
    else:
        return None

    return prepare_prediction_response(prediction[0], mp.MODEL_LABELS)


if __name__ == '__main__':
    model, _, accuracy = train_model(NetworkType.DNN, 32, 1500)
    save_model(model, NetworkType.DNN)
    print('Model loss:', accuracy[0])
    print('Model accuracy:', accuracy[1])

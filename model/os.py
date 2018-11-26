import os

from keras.models import model_from_json

# constants
MODEL_DIRECTORY = 'data'


def save_model(model, network_type):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(MODEL_DIRECTORY, "{}_model.json".format(network_type)), 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(MODEL_DIRECTORY, "{}_model.h5".format(network_type)))


def read_model(network_type):
    # load json and create model
    json_file = open(os.path.join(MODEL_DIRECTORY, "{}_model.json".format(network_type)), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(MODEL_DIRECTORY, "{}_model.h5".format(network_type)))

    return loaded_model

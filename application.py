from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from model.services import predict_emotion
from model.utils import NetworkType

application = Flask(__name__)


@application.route('/train/<model_type>', methods=['POST'])
def train_network_model(model_type):
    return 


@application.route('/prediction/<model_type>', methods=['POST'])
def predict_emotion_from_feature(model_type):
    try:
        network_type = NetworkType(model_type)
        data = request.get_json()
        mfcc = data['mfcc']
        prediction = predict_emotion(mfcc, network_type)

        return jsonify(prediction)
    except ValueError:
        raise BadRequest('Wrong learning network type')


# run the app.
if __name__ == "__main__":

    application.debug = True
    application.run()

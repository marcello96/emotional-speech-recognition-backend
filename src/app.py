import json

import falcon

from model.services import predict_emotion, load_models
from model.utils import NetworkType
from model.converter import SimpleEncoder, map_to_json_response


class Prediction:
    def on_post(self, req, resp, model_type):
        try:
            network_type = NetworkType(model_type.lower())
            if req.content_length:
                mfccs = json.load(req.stream)['mfcc']

                prediction = predict_emotion(mfccs, network_type)

                resp.status = falcon.HTTP_200
                resp.body = json.dumps(map_to_json_response(prediction), cls=SimpleEncoder)
            else:
                resp.status = falcon.HTTP_400
                resp.body = 'Wrong number of mfcc features'
        except ValueError as e:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps(str(e))


# run the app.

load_models()
app = falcon.API()
app.add_route('/prediction/{model_type}', Prediction())

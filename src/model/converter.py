import json
import numpy


class SimpleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(SimpleEncoder, self).default(obj)
        
        
def map_to_json_response(predictions):
    def map_prediction_elem(key, value):
        return {'emotionType': key,
                'prediction': value}

    return {'results': [map_prediction_elem(key, value) for key, value in predictions.items()]}

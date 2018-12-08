from enum import Enum


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class NetworkType(NoValue):
    DNN = 'dnn'
    CNN = 'cnn'


def prepare_prediction_response(prediction, labels):
    return {labels[i]: prediction[i] for i in range(len(labels))}

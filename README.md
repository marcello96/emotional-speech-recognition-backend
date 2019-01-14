# Emotional speech recognition server
Server application written in Python 3.6, which makes it possible for clients to predict emotions from acoustic characteristics.

Prerequisites:
- Docker

Used technologies:
1. Python 3.6
2. Falcon
3. Gunicorn
4. Nginx
5. Supervisord

Model was trained using [Ravdess database](https://smartlaboratory.org/ravdess/).

## Run Project

To run this project:
1. Build docker image `docker build -t emotion-recognition .`
1. Run docker container `docker run -p 8080:80 emotion-recognition`
1. Base url for endpoints will be `localhost:8080`

## API

Server exposes two endpoints:
- `POST /prediction/{networkType}` -- where `networkType` may be `CNN` or `DNN`
-- predicts emotions for specified MFCC coefficients in request.

Request (amount of MFCCs was specified to 25):
```
{
    "mfccs": [{Float},...,{Float}]
}
```
Response:
```
{
    "results": [
        {
            "emotionType": "angry",
            "prediction": {Float}
        },
        {
            "emotionType": "calm",
            "prediction": {Float}
        },
        {
            "emotionType": "disgust",
            "prediction": {Float}
        },
        {
            "emotionType": "fearful",
            "prediction": {Float}
        },
        {
            "emotionType": "happy",
            "prediction": {Float}
        },
        {
            "emotionType": "neutral",
            "prediction": {Float}
        },
        {
            "emotionType": "sad",
            "prediction": {Float}
        },
        {
            "emotionType": "surprised",
            "prediction": {Float}
        }
    ]
}
```
- `GET /init/configuration` -- returns initial configuration, actually it is amount of MFCC coefficients, which should be sent in `/prediction` request

Response:
```
{
    "numberOfMfccs": 25
}
```


## Train model locally
If you want to retrain model, set up project locally.

Prerequisites:
1. pip
1. Python 3.6

Install virtualenv via pip:
```
pip install virtualenv
```
Create a virtual environment for a project:
```
cd emotional-speech-recognition-backend
virtualenv venv
```
Install libraries:
```
pip install -r requirements.txt
```

If you want to retrain model, download Ravdess audio database and set root directory path in `config.ini` file.
```
[data]
source_path = E:\Python\Ravness
```


Specify model parameters in `model/services.py` and run it.
from flask import Flask

application = Flask(__name__)


@application.route('/analyze')
def analyze_emotion():
    return 'HAPPY'


# run the app.
if __name__ == "__main__":

    application.debug = True
    application.run()

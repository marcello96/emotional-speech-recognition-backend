from flask import Flask, jsonify, request

application = Flask(__name__)


@application.route('/analyze', methods=['POST'])
def analyze_emotion():
    if request.method == 'POST':
        data = request.get_json()

        return jsonify({
            "emotion": "HAPPY"
        })


# run the app.
if __name__ == "__main__":

    application.debug = True
    application.run()

import json
import os
import threading
import time

import cv2
from flask import Flask, Response, request, jsonify

import vision

app = Flask(__name__)


def root_dir():
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


@app.route("/jquery.min.js")
def jquery():
    content = get_file('jquery.min.js')
    return Response(content, mimetype="application/javascript")


@app.route("/")
def root():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")


def load_config():
    config = json.loads(get_file('config.json'))
    return config


def save_config(config):
    with open('config.json', 'w') as outfile:
        outfile.write(json.dumps(config))
        outfile.close()


def get_image():
    img = cv2.imread('/Users/vmagro/Desktop/tower.png', cv2.IMREAD_COLOR)
    return img


state = {}
config = load_config()


def worker():
    while True:
        img = get_image()
        state['img'] = img
        args = config.copy()
        args['img'] = img
        args['output_images'] = {}

        targets = vision.find(**args)
        state['targets'] = targets
        state['output_images'] = args['output_images']
        # 30fps
        time.sleep(0.33)
    return


@app.route('/config', methods=['GET', 'POST'])
def config_route():
    if request.method == 'POST':
        global config
        config = request.form
        return jsonify(**config)
    else:
        return jsonify(**config)


@app.route('/image')
def image_route():
    _, jpeg = cv2.imencode('.jpg', state['output_images']['img'])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/result')
def result_image_route():
    _, jpeg = cv2.imencode('.jpg', state['output_images']['result'])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/binary')
def bin_image_route():
    _, jpeg = cv2.imencode('.jpg', state['output_images']['bin'])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


if __name__ == "__main__":
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run()

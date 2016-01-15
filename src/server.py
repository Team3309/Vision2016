#!/usr/bin/env python
import json
import os
import signal
import sys
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
    config = json.loads(get_file('config.json'))['target']
    return config


def save_config(config):
    with open('config.json', 'w') as outfile:
        outconfig = {'target': config}
        outfile.write(json.dumps(outconfig, indent=2, separators=(',', ': ')))
        outfile.close()


image_count = 0


def get_image():
    img = cv2.imread('/Users/vmagro/Developer/frc/RealFullField/7.jpg', cv2.IMREAD_COLOR)
    # global image_count
    # path = '/Users/vmagro/Developer/frc/RealFullField/' + str(image_count) + '.jpg'
    # print(path)
    # img = cv2.imread(path, cv2.IMREAD_COLOR)
    # image_count = (image_count + 1) % 350
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
        config = dict((key, int(request.form.get(key))) for key in request.form.keys())
        save_config(config)
        return jsonify(**config)
    else:
        return jsonify(**config)


@app.route('/targets')
def targets_route():
    if state is not None and 'targets' in state:
        targets = state['targets']
        return Response(json.dumps(targets), mimetype='application/json')
    else:
        return Response("[]", mimetype='application/json')


def image_generator(name):
    while True:
        _, frame = cv2.imencode('.jpg', state['output_images'][name])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        time.sleep(0.33)


@app.route('/binary')
def binary_image_route():
    return Response(image_generator('bin'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result_image_route():
    return Response(image_generator('result'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def shutdown_server():
    print('Shutting down server')
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    sys.exit(0)


if __name__ == "__main__":
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=True, threaded=True)
    signal.signal(signal.SIGINT, shutdown_server)
    signal.pause()

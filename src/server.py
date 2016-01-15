#!/usr/bin/env python
import json
import os
import sys
import threading

import cv2
from flask import Flask, Response, request, jsonify

import vision

app = Flask(__name__)


def root_dir():
    return os.path.abspath(os.path.dirname(__file__))


def get_file(filename):
    try:
        src = os.path.join(root_dir(), filename)
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


state = {}
config = load_config()


@app.route('/config', methods=['GET', 'POST'])
def config_route():
    if request.method == 'POST':
        global config
        config = dict((key, int(request.form.get(key))) for key in request.form.keys())
        save_config(config)
        return jsonify(**config)
    else:
        return jsonify(**config)


new_data_lock = threading.RLock()
new_data_condition = threading.Condition(new_data_lock)


@app.route('/targets')
def targets_route():
    try:
        new_data_condition.acquire()
        new_data_condition.wait()
        targets = state['targets']
        return Response(json.dumps(targets), mimetype='application/json')
    finally:
        new_data_condition.release()


def image_generator(name):
    while True:
        try:
            new_data_condition.acquire()
            new_data_condition.wait()
            _, frame = cv2.imencode('.jpg', state['output_images'][name])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        finally:
            new_data_condition.release()


@app.route('/binary')
def binary_image_route():
    return Response(image_generator('bin'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result_image_route():
    return Response(image_generator('result'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_server():
    app.run(host='0.0.0.0', debug=False, threaded=True)


def shutdown_server():
    print('Shutting down server')
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    sys.exit(0)


if __name__ == "__main__":
    print("main init")
    video_cap = cv2.VideoCapture(0)
    thread = threading.Thread(target=start_server)
    thread.daemon = True
    thread.start()

    capture = cv2.VideoCapture()
    print('Opening camera')
    capture.open(0)
    print('Opened camera')
    while True:
        success, img = capture.read()
        if not success:
            print('Failed to get image from camera')
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        new_data_condition.acquire()
        state['img'] = hsv
        args = config.copy()
        args['img'] = hsv
        args['output_images'] = {}

        targets = vision.find(**args)
        state['targets'] = targets
        state['output_images'] = args['output_images']
        new_data_condition.notify_all()
        new_data_condition.release()

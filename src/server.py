#!/usr/bin/env python
# FRC Vision 2016
# Copyright 2016 Vinnie Magro
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import socket
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
    return json.loads(get_file('config.json'))['target']


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


def handle_image(img):
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


def camera_loop():
    capture = cv2.VideoCapture()
    print('Opening camera')
    capture.open(0)
    print('Opened camera')
    while True:
        success, img = capture.read()
        if not success:
            print('Failed to get image from camera')
            continue

        handle_image(img)


def image_loop():
    image_counter = 0
    while True:
        # path = '/Users/vmagro/Developer/frc/RealFullField/11.jpg'
        # path = '/Users/vmagro/Developer/frc/Vision2016/test/img/11.jpg'
        path = '/Users/vmagro/Developer/frc/RealFullField/' + str(image_counter) + '.jpg'
        print(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        image_counter = (image_counter + 1) % 350

        handle_image(img)
        time.sleep(0.5)  # slow down so its visible


def comm_loop():
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    while True:
        try:
            new_data_condition.acquire()
            new_data_condition.wait()

            targets = state['targets']
            message = json.dumps(targets)
            sock.sendto(message, ('localhost', 3309))
        finally:
            new_data_condition.release()


if __name__ == "__main__":
    print("main init")
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    comm_thread = threading.Thread(target=comm_loop)
    comm_thread.daemon = True
    comm_thread.start()

    # camera_loop()
    image_loop()

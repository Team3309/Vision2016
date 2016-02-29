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
import math
import os
import socket
import threading
import time

import cv2
from flask import Flask, Response, request, jsonify
from flask_sockets import Sockets

import vision

app = Flask(__name__)
sockets = Sockets(app)


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
    return json.loads(get_file('config.json'))


def save_config(outconfig):
    with open('config.json', 'w') as outfile:
        outfile.write(json.dumps(outconfig, indent=2, separators=(',', ': ')))
        outfile.close()


state = {'ack': False}
config = load_config()


@app.route('/target-config', methods=['GET', 'POST'])
def config_route():
    if request.method == 'POST':
        config['target'] = dict((key, int(request.form.get(key))) for key in request.form.keys())
        save_config(config)
        return jsonify(**(config['target']))
    else:
        return jsonify(**(config['target']))


new_data_lock = threading.RLock()
new_data_condition = threading.Condition(new_data_lock)


@sockets.route('/socket')
def update_socket(ws):
    print 'websocket connection request'
    while not ws.closed:
        new_data_condition.acquire()
        new_data_condition.wait()
        result = {
            'targets': state['targets'],
            'fps': state['fps'],
            'connected': state['ack']
        }
        message = json.dumps(result)
        new_data_condition.release()
        ws.send(message)


def image_generator(name):
    while True:
        try:
            new_data_condition.acquire()
            new_data_condition.wait()
            _, frame = cv2.imencode('.jpg', state['output_images'][name])
            new_data_condition.release()
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
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()


def handle_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    new_data_condition.acquire()
    state['img'] = hsv
    args = config['target'].copy()
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
        start_time = time.time()
        success, img = capture.read()
        if not success:
            print('Failed to get image from camera')
            continue

        handle_image(img)

        elapsed_time = time.time() - start_time
        elapsed_time_s = elapsed_time / 1000
        max_fps = int(math.floor(1 / elapsed_time_s))
        state['fps'] = max_fps
        print 'Processed in', elapsed_time, 'ms, max fps =', max_fps


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


def ack_loop():
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_sock.bind(('', 9033))
    listen_sock.settimeout(5)  # 5 second timeout

    # wait for an ack
    try:
        data, addr = listen_sock.recvfrom(2048)
        if json.loads(data)['ack']:
            print 'Got ack from', addr
            state['ack'] = True
        else:
            state['ack'] = False
    except socket.timeout:
        print 'Didn\'t get ACK from client'
        state['ack'] = False


def comm_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            new_data_condition.acquire()
            new_data_condition.wait()

            targets = state['targets']
            message = json.dumps(targets)
            new_data_condition.release()
            try:
                sock.sendto(message, (config['destination'], 3309))
            except socket.error:
                state['ack'] = False
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

    ack_thread = threading.Thread(target=ack_loop)
    ack_thread.daemon = True
    ack_thread.start()

    camera_loop()
    # image_loop()

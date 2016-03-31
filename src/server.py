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

import base64
import datetime
import json
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


state = {'ack': False, 'draw_output': False}
config = load_config()


@app.route('/config', methods=['GET'])
def config_route():
    return jsonify(**(config))


new_data_lock = threading.RLock()
new_data_condition = threading.Condition(new_data_lock)


@sockets.route('/socket')
def update_socket(ws):
    print 'websocket connection request'
    state['draw_output'] = True
    while not ws.closed:
        new_data_condition.acquire()
        new_data_condition.wait()
        new_data_condition.release()
        result = {
            'targets': state['targets'],
            'fps': state['fps'],
            'connected': state['ack']
        }
        _, binframe = cv2.imencode('.jpg', state['output_images']['bin'])
        result['binaryImg'] = base64.b64encode(binframe)
        _, binframe = cv2.imencode('.jpg', state['output_images']['result'])
        result['resultImg'] = base64.b64encode(binframe)
        message = json.dumps(result)
        ws.send(message)
        received = json.loads(ws.receive())
        if 'thresholds' in received:
            config['target'] = received['thresholds']
            save_config(config)
        if 'camera' in received:
            config['camera'] = received['camera']
            save_config(config)

    print 'websocket disconnected'
    state['draw_output'] = False


def start_server():
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()


def handle_image(img):
    new_data_condition.acquire()
    state['img'] = img
    args = config['target'].copy()
    args['img'] = img
    args['draw_output'] = state['draw_output']
    args['output_images'] = {}

    targets = vision.find(**args)
    state['targets'] = targets
    state['output_images'] = args['output_images']
    new_data_condition.notify_all()
    new_data_condition.release()


def camera_loop():
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    print('Opening camera')
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)

    camera.resolution = (640, 480)
    time.sleep(1)

    camera.start_preview()
    camera.awb_mode = 'off'
    camera.exposure_mode = 'off'

    # use initial values from config file
    camera.awb_gains = (config['camera']['awb_red_gain'], config['camera']['awb_red_gain'])
    camera.shutter_speed = config['camera']['shutter_speed']
    camera.iso = config['camera']['iso']

    print('Opened camera')

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image
        image = frame.array

        handle_image(image)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # set all the settings if they changed in the config
        camera.awb_gains = (config['camera']['awb_red_gain'], config['camera']['awb_red_gain'])
        camera.shutter_speed = config['camera']['shutter_speed']
        camera.iso = config['camera']['iso']


def image_loop():
    image_counter = 0
    while True:
        # path = '/Users/vmagro/Developer/frc/RealFullField/11.jpg'
        # path = '/Users/vmagro/Developer/frc/Vision2016/test/img/11.jpg'
        path = '/Users/vmagro/Developer/frc/RealFullField/' + str(image_counter) + '.jpg'
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        image_counter = (image_counter + 1) % 350

        handle_image(img)
        time.sleep(33 / 1000)  # slow down so its visible


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
    fps_smoothed = None
    fps_smoothing_factor = 0.5
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        start = datetime.datetime.now()
        try:
            new_data_condition.acquire()
            new_data_condition.wait()
            new_data_condition.release()

            targets = state['targets']
            message = json.dumps(targets)
            try:
                sock.sendto(message, (config['destination'], 5809))
            except socket.error:
                state['ack'] = False
        finally:
            end = datetime.datetime.now()
            delta = end - start
            # frame time = 1 second / fps
            # fps = 1 second / frame time
            max_fps = 1 / delta.total_seconds()

            # apply exponential smoothing to fps calculation
            if not fps_smoothed:
                fps_smoothed = max_fps
            fps_smoothed = fps_smoothing_factor * max_fps + (1 - fps_smoothing_factor) * fps_smoothed

            state['fps'] = round(fps_smoothed, 1)
            print 'Processed in', delta.total_seconds() * 1000, 'ms, max fps =', round(fps_smoothed, 1)


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

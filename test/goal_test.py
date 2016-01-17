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
from os import listdir
from os.path import isfile, join

import cv2

from src.vision import find

config = json.loads(open('../src/config.json').read())['target']


def is_close(expected, actual, tolerance):
    return expected - tolerance < actual < expected + tolerance


def check_image(name):
    expected_data = json.loads(open('./img/' + name + '.json').read())
    if not expected_data['enabled']:
        return

    expected_targets = expected_data['targets']

    img = cv2.imread('./img/' + name + '.jpg', cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    args = config.copy()
    args['img'] = hsv
    args['output_images'] = {}

    actual_targets = find(**args)

    # make sure same number of targets are detected
    assert len(expected_targets) == len(actual_targets)

    # targets is a list of 2-tuples with expected and actual results
    targets = zip(expected_targets, actual_targets)
    # compare all the different features of targets to make sure they match
    for pair in targets:
        expected, actual = pair
        # make sure that the targets are close to where they are supposed to be
        assert is_close(expected['pos']['x'], actual['pos']['x'], 0.02)
        assert is_close(expected['pos']['y'], actual['pos']['y'], 0.02)
        # make sure that the targets are close to the size they are supposed to be
        assert is_close(expected['size']['width'], actual['size']['width'], 0.02)
        assert is_close(expected['size']['height'], actual['size']['height'], 0.02)


def test_all_images():
    images_path = './img/'
    all_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    names = [name for name in all_files if name != '.DS_Store']
    names = [name[:name.rfind('.')] for name in names]
    names = set(names)
    for name in names:
        check_image(name)

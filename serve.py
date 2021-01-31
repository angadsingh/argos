import logging
import sys

from detection.object_detector_base import BaseTFObjectDetector

for _ in ("colormath.color_conversions", "colormath.color_objects"):
    logging.getLogger(_).setLevel(logging.CRITICAL)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)

import base64
import importlib
import json
import os

from werkzeug.utils import secure_filename

from base_detector import DetectorView

log.info("package import START")
import argparse
import threading

import cv2
from flask import Flask
from flask import jsonify
from flask import request

from lib.fps import FPS
from flask_classful import route

log.info("package import END")


class ServingDetectorView(DetectorView):
    def __init__(self, upload_folder):
        super().__init__()
        self.upload_folder = upload_folder
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        m = importlib.import_module(args["config"])
        self.config = getattr(m, "Config")()
        log.info("TFObjectDetector init START")
        self.od = BaseTFObjectDetector(self.config)
        log.info("TFObjectDetector init END")
        self.fps = FPS(50, 100)
        self.total_frames = 0
        self.init = False

    @route('/detect', methods=['POST'])
    def detect(self):
        if not self.init:
            self.od.initialize_tf_model()
            self.init = True
        file = request.files['file']
        filename = secure_filename(file.filename)
        image_path = os.path.join(self.upload_folder, filename)
        file.save(image_path)
        frame = cv2.imread(image_path)
        threshold = None
        if request.args.get('threshold'):
            threshold = float(request.args.get('threshold'))
        nmasks = []
        if request.args.get('nmask'):
            nmasks = [json.loads(base64.urlsafe_b64decode(request.args.get('nmask')).decode())]
        detected_boxes = self.od.detect_image(frame, threshold, nmasks=nmasks)
        if detected_boxes is not None and len(detected_boxes) > 0:
            log.info(str(detected_boxes))
        self.fps.count()
        if self.total_frames % self.config.fps_print_frames == 0:
            log.info("FPS=%.2f" % self.fps.fps)
        self.total_frames += 1
        os.remove(image_path)
        return jsonify(detected_boxes)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-c", "--config", type=str, required=True,
                    help="path to the python config file")
    ap.add_argument("-u", "--uploadfolder", type=str, required=True,
                    help="where to keep uploaded jpeg files")
    args = vars(ap.parse_args())

    log.info("flask init..")
    app = Flask(__name__)
    ServingDetectorView.register(app, init_argument=args["uploadfolder"], route_base='/')
    f = threading.Thread(target=app.run, kwargs={'host': args["ip"], 'port': args["port"], 'debug': False,
                                                 'threaded': True, 'use_reloader': False})
    f.daemon = True
    f.start()
    f.join()

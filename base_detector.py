import jsonpickle
from flask import request, jsonify, Response
from flask_classful import FlaskView, route


class DetectorView(FlaskView):
    @route('/config')
    def apiconfig(self):
        self.config.fps_print_frames = int(request.args.get('fps_print_frames', self.config.fps_print_frames))
        self.config.tf_accuracy_threshold = float(request.args.get('tf_accuracy_threshold', self.config.tf_accuracy_threshold))
        self.config.tf_detection_buffer_duration = int(
            request.args.get('tf_detection_buffer_duration', self.config.tf_detection_buffer_duration))
        self.config.tf_detection_buffer_threshold = int(
            request.args.get('tf_detection_buffer_threshold', self.config.tf_detection_buffer_threshold))

        return Response(jsonpickle.encode(self.config.__dict__, unpicklable=False, max_depth=2), mimetype='application/json')
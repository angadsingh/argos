import time
import unittest
from lib.detection_buffer import SlidingWindowDetectionBuffer

class TestDetectionBuffer(unittest.TestCase):
	def test1(self):
		now = int(round(time.time()))
		test_case = [
				(now, ((338, 96, 654, 475, 'personA', 0.6414415240287781), None)),
				(now, ((346, 93, 648, 401, 'personA', 0.5375432968139648), None)),
				(now, ((357, 95, 641, 414, 'personA', 0.6364365220069885), None)),
				(now, ((332, 100, 643, 453, 'personA', 0.6406891942024231), None)),
				(now, ((312, 107, 624, 450, 'personA', 0.6004123091697693), None)),
				(now, ((311, 97, 612, 443, 'personA', 0.5544843673706055), None)),
				(now, ((313, 82, 627, 435, 'personB', 0.880099606513977), None)),
				(now, ((317, 112, 621, 484, 'personB', 0.9625759840011597), None)),
				(now, ((327, 119, 622, 475, 'personB', 0.6868389248847961), None)),
				(now, ((351, 145, 623, 477, 'personB', 0.5230631828308105), None)),
				(now, ((342, 125, 640, 473, 'personC', 0.5845738053321838), None)),
				(now, ((340, 125, 630, 483, 'personC', 0.5292268395423889), None)),
				(now, ((373, 126, 610, 470, 'personC', 0.6521285176277161), None)),
				(now, ((330, 113, 643, 462, 'personC', 0.5796730518341064), None))
			]

		detection_buffer = SlidingWindowDetectionBuffer()
		for t in test_case:
			ts, box = t
			detection_buffer.add_detection(box, ts)

		self.assertEqual(detection_buffer.get_max_cumulative_accuracy_label(), ('personA', 3.6110072135925293, None))


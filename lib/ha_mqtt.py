import json
import paho.mqtt.client as mqtt
import logging
log = logging.getLogger(__name__)

class HaMQTT:
    def __init__(self, broker, port, username, passwd):
        self.client = mqtt.Client("pi-mqtt-client-1")
        self.client.connect(broker, port=port)
        self.client.username_pw_set(username=username, password=passwd)
        self.client.loop_start()

    def publish(self, topic, payload):
        log.info("mqttPublish: %s" % str(payload))
        if type(payload) in [dict, list]:
            payload = json.dumps(payload)
        self.client.publish(topic, payload)

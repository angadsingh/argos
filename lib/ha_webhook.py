import os
from urllib.parse import quote_plus

import requests
from retrying import retry

from paramiko import SSHClient
from scp import SCPClient

import logging
log = logging.getLogger(__name__)

class HaWebHook():
    def __init__(self, webhook_url, ssh_host = None, ssh_username = None, target_dir = None):
        self.webhook_url = webhook_url
        self.ssh_host = ssh_host
        if self.ssh_host:
            ssh = SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(ssh_host, username=ssh_username)
            self.scp = SCPClient(ssh.get_transport())
            self.target_dir = target_dir

    @retry(wait_fixed=2000, stop_max_delay=60000)
    def send(self, label, img_path):
        log.info(str(label))
        if self.ssh_host:
            log.info("scp %s to %s on %s" % (img_path, self.target_dir, self.ssh_host))
            self.scp.put(img_path, self.target_dir)
        url = self.webhook_url.format(
            quote_plus(label), os.path.basename(img_path))
        requests.post(url)
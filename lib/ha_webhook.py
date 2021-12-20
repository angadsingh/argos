import os
import threading
import time
from urllib.parse import quote_plus

import requests
from retrying import retry, Retrying

import logging

log = logging.getLogger(__name__)

rt = Retrying(wait_fixed=2000)

def log_retry(fn):
    def fn_with_logging(*args, **kwargs):
        log.warning('Attempt {:d} failed after {:d} milliseconds'.format(args[0], args[1]))
        return fn(*args, **kwargs)
    return fn_with_logging

class HaWebHook():
    def __init__(self, webhook_url, ssh_host = None, ssh_username = None, target_dir = None):
        self.webhook_url = webhook_url
        if ssh_host:
            from paramiko import SSHClient
            from scp import SCPClient
        self.ssh_host = ssh_host
        self.target_dir = target_dir
        self.ssh_username = ssh_username

    def do_scp(self, img_path):
        if self.ssh_host:
            ssh = SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(self.ssh_host, username=self.ssh_username, timeout=2)
            scp = SCPClient(ssh.get_transport(), socket_timeout=2)
            scp.put(img_path, self.target_dir)
            scp.close()
            ssh.close()

    @retry(stop_max_delay=60000, wait_func=log_retry(rt.fixed_sleep))
    def send(self, label, img_path = None):
        if self.ssh_host:
            scp_log = "%s to %s on %s" % (img_path, self.target_dir, self.ssh_host)
            log.info("scp'ing %s" % scp_log)
            try:
                self.do_scp(img_path)
            except TimeoutError as e:
                log.warning("timed out during scp of %s" % scp_log)
        url = None
        if img_path:
            url = self.webhook_url.format(
                quote_plus(label), os.path.basename(img_path))
        else:
            url = self.webhook_url.format(
                quote_plus(label))
        log.info(f"webhook: {url}")
        requests.post(url)
import logging
import os
import socket
import subprocess
import sys
import threading
import time

LOG_LEVEL = os.getenv("PORTFORWARDER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SSH_HOST = os.getenv("PORTFORWARDER_SSH_HOST", "andersan.net")
SSH_PORT = int(os.getenv("PORTFORWARDER_SSH_PORT", "22"))
SSH_USER = os.getenv("PORTFORWARDER_SSH_USER", "ubuntu")
REMOTE_PORT = int(os.getenv("PORTFORWARDER_REMOTE_PORT", "8087"))
LOCAL_PORT = int(os.getenv("PORTFORWARDER_LOCAL_PORT", "8087"))
WAIT_LOCAL_SEC = int(os.getenv("PORTFORWARDER_WAIT_LOCAL_SEC", "120"))
SSH_VERBOSE = os.getenv("PORTFORWARDER_SSH_VERBOSE", "").lower() in ("1", "true", "yes")


def wait_for_local_port(port: int, timeout_sec: int) -> bool:
    """andersan-api が listen するまで待つ（PM2 並列起動向け）。"""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                logger.info("Local port %s is ready", port)
                return True
        except OSError:
            time.sleep(2)
    logger.warning(
        "Local port %s not ready after %ss; starting SSH tunnel anyway",
        port,
        timeout_sec,
    )
    return False


def build_ssh_command(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    remote_port: int,
    local_port: int,
) -> list[str]:
    ssh_command = [
        "ssh",
        "-N",
        "-R",
        f"{remote_port}:localhost:{local_port}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        f"{ssh_user}@{ssh_host}",
        "-p",
        str(ssh_port),
    ]
    if SSH_VERBOSE:
        ssh_command.insert(1, "-v")
    return ssh_command


def _spawn_ssh(ssh_command: list[str]) -> subprocess.Popen:
    process = subprocess.Popen(
        ssh_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    def log_output(pipe, prefix: str) -> None:
        for line in pipe:
            line = line.strip()
            if line:
                logger.debug("%s: %s", prefix, line)

    stdout_thread = threading.Thread(target=log_output, args=(process.stdout, "STDOUT"))
    stderr_thread = threading.Thread(target=log_output, args=(process.stderr, "STDERR"))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    return process


def forward_ssh_port(
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    remote_port: int,
    local_port: int,
) -> None:
    """SSH リバース転送を張り、切れたら再接続する。"""
    ssh_command = build_ssh_command(
        ssh_host, ssh_port, ssh_user, remote_port, local_port
    )
    logger.info(
        "Starting SSH port forwarding: remote %s:%s -> localhost:%s",
        ssh_host,
        remote_port,
        local_port,
    )
    process = _spawn_ssh(ssh_command)

    while True:
        if process.poll() is not None:
            logger.warning(
                "SSH process terminated (exit=%s). Restarting in 5s...",
                process.returncode,
            )
            time.sleep(5)
            process = _spawn_ssh(ssh_command)
        time.sleep(10)


if __name__ == "__main__":
    # Internet上の中継サーバ側で、sshdに GatewayPorts yes が必要
    wait_for_local_port(LOCAL_PORT, WAIT_LOCAL_SEC)
    forward_ssh_port(SSH_HOST, SSH_PORT, SSH_USER, REMOTE_PORT, LOCAL_PORT)

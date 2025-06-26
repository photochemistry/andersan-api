import subprocess
import threading
import time
import logging
import sys

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def forward_ssh_port(ssh_host, ssh_port, ssh_user, remote_port, local_port):
    """Set up SSH port forwarding to an existing local port"""
    try:
        # SSHコマンドを構築
        ssh_command = [
            "ssh",
            "-N",  # コマンドを実行しない
            "-R", f"{remote_port}:localhost:{local_port}",  # リモートポートフォワーディング
            "-o", "ServerAliveInterval=30",  # キープアライブ
            "-o", "ServerAliveCountMax=3",   # リトライ回数
            "-o", "TCPKeepAlive=yes",        # TCPキープアライブを有効化
            "-o", "ExitOnForwardFailure=yes", # フォワーディング失敗時に終了
            "-v",  # 詳細なデバッグ情報を表示
            f"{ssh_user}@{ssh_host}",
            "-p", str(ssh_port)
        ]
        
        logger.info(f"Starting SSH port forwarding: {' '.join(ssh_command)}")
        
        # プロセスを開始
        process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 標準出力と標準エラー出力を監視
        def log_output(pipe, prefix):
            for line in pipe:
                logger.debug(f"{prefix}: {line.strip()}")
        
        stdout_thread = threading.Thread(target=log_output, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=log_output, args=(process.stderr, "STDERR"))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        logger.info(f"SSH port forwarding started. Remote port {remote_port} -> Local port {local_port}")
        
        # プロセスの終了を待機
        while True:
            if process.poll() is not None:
                logger.warning("SSH process terminated. Restarting...")
                process = subprocess.Popen(
                    ssh_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                stdout_thread = threading.Thread(target=log_output, args=(process.stdout, "STDOUT"))
                stderr_thread = threading.Thread(target=log_output, args=(process.stderr, "STDERR"))
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()
            time.sleep(10)

    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == "__main__":
    ssh_host = "andersan.riis.okayama-u.ac.jp"
    ssh_port = 22
    ssh_user = "andersan"
    remote_port = 8087
    local_port = 8087

    # Run in thread
    thread = threading.Thread(target=forward_ssh_port, args=(ssh_host, ssh_port, ssh_user, remote_port, local_port))
    thread.start()
    
    
# INternet上の中継サーバ側で、sshdに以下の設定を追加
# GatewayPorts yes

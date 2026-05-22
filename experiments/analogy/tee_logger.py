import sys
import os


class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self.log = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.log.close()


def setup_logger(log_path):
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee
    return tee

import sys
import os
import time
timestr = time.strftime("%Y%m%d-%H%M%S")


class OutputLogger:
    def __init__(self, options):
        self.file = open(os.path.join(options.output_dir(), "OutputLog.txt"), "w+")
        self.stdout = sys.stdout
        sys.stdout = self
        self.last_char = '\n'

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        prefix = time.strftime("[%Y-%m-%d %H:%M:%S] ") if self.last_char == '\n' else ""
        self.last_char = data[-1]

        self.file.write(prefix + data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class ErrorLogger:
    def __init__(self, options):
        self.file   = open(os.path.join(options.output_dir(), "ErrorLog.txt"), "w+")
        self.stderr = sys.stderr
        sys.stderr  = self
        self.last_char = '\n'

    def __del__(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        prefix = time.strftime("[%Y-%m-%d %H:%M:%S] ") if self.last_char == '\n' else ""
        self.last_char = data[-1]

        self.file.write(prefix + data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()
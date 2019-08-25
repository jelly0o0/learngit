# -*- coding: utf-8 -*-

import time
import sys


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.eta = 0.0
        self.total_time = 0.0
        self.last_time = self.start

    def elapsed_time(self):
        self.last_time = time.time()
        return self.last_time - self.start

    def calc_eta(self):
        elapsed = self.elapsed_time()
        if self.i == 0 or elapsed < 0.001:
            return None
        rate = float(self.i) / elapsed
        self.eta = (float(self.max_steps) - float(self.i)) / rate

    def get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        self.calc_eta()
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '\r' + '[' + '=' * num_arrow + ' ' * num_line + ']'\
                      + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        self.close()

    def close(self):
        if self.i >= self.max_steps:
            self.total_time = self.elapsed_time()
            print('\nTotal time elapsed: ' + self.get_time(self.total_time))


if __name__ == '__main__':
    max_steps = 10

    process_bar = ShowProcess(max_steps)
    for i in range(max_steps):
        time.sleep(10)
        process_bar.show_process()

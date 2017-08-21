import pandas as pd
import numpy as np
import sys
import re


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to end.'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


train = pd.read_csv('train_data_set.csv')
train.head()

print("Train X: ")
temp = []
progress = ProgressBar(len(train.value), fmt=ProgressBar.FULL)
for value in train.value:
    val = value.split(";")[:-1]
    tmp = []
    for v in val:
        tmp.append(int(v))
    if len(tmp) is not 64:
        print(value)
    temp.append(tmp)
    progress.current += 1
    progress()

train_x = np.stack(temp)
progress.done()


print("Train Y: ")
temp = []
progress = ProgressBar(len(train.gesture), fmt=ProgressBar.FULL)
for value in train.gesture:
    tmp = [0, 0, 0, 0, 0, 0]
    tmp[value] = 1
    temp.append(tmp)
    progress.current += 1
    progress()

progress.done()
train_y = np.stack(temp)


np.savez("./train_set.npz", x=train_x, y=train_y)
print("Saved")

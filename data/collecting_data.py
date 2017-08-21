import collections
import myo
import threading
import time
import numpy as np


fd = open("train_data_set.csv", 'a')
fd.write("value,gesture")


class MyListener(myo.DeviceListener):

    def __init__(self, queue_size=8):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)

    def on_connect(self, device, timestamp, firmware_version):
        device.set_stream_emg(myo.StreamEmg.enabled)

    def on_emg_data(self, device, timestamp, emg_data):
        with self.lock:
            self.emg_data_queue.append((timestamp, emg_data))

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)


myo.init()
hub = myo.Hub()
i = 0
try:
    listener = MyListener()
    hub.run(2000, listener)
    while True:
        data = listener.get_emg_data()
        if len(data) > 0:
            tmp = []
            for v in listener.get_emg_data():
                tmp.append(v[1])
            tmp = list(np.stack(tmp).flatten())
            i += 1
            val = ""
            for k in tmp:
                val += str(k) + ";"
            if len(val) >= 72:
                fd.write("\n"+val + ",0")
                gesture = 0

                print(i)
                i += 1
        time.sleep(0.01)
finally:
    hub.shutdown()
    fd.close()

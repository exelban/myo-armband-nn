import collections
import myo
import threading
import time
import numpy as np
import tensorflow as tf
from include.model import model


x, y, output, global_step, y_pred_cls = model()

saver = tf.train.Saver()
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/"
sess = tf.Session()


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    print(last_chk_path)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


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
start = time.time()
temp = []
try:
    listener = MyListener()
    hub.run(2000, listener)
    while True:
        data = listener.get_emg_data()
        if time.time() - start >= 1:
            response = np.argmax(np.bincount(temp))
            print("Predicted gesture: {0}".format(response))
            temp = []
            start = time.time()
        if len(data) > 0:
            tmp = []
            for v in listener.get_emg_data():
                tmp.append(v[1])
            tmp = list(np.stack(tmp).flatten())
            if len(tmp) >= 64:
                pred = sess.run(y_pred_cls, feed_dict={x: np.array([tmp])})
                temp.append(pred[0])
        time.sleep(0.01)
finally:
    hub.shutdown()
    sess.close()

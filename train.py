import numpy as np
import tensorflow as tf
from time import time
from include.data import get_data_set
from include.model import model


train_x, train_y = get_data_set()

_BATCH_SIZE = 300
_CLASS_SIZE = 6
_SAVE_PATH = "./data/tensorflow_sessions/myo_armband/"

x, y, output, global_step, y_pred_cls = model(_CLASS_SIZE)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
tf.summary.scalar("Loss", loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)


correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)


init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_iterations = 1000):
    for i in range(num_iterations):
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]

        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        duration = time() - start_time

        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
            train_writer.add_summary(data_merged, global_1)
            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")


train(75000)


sess.close()

import os
import sys


from input_data import InputData
import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_boolean('predict', False, """推論モードフラグ""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', """チェックポイント保存先""")

FLAGS = tf.app.flags.FLAGS


def train():
  with tf.Graph().as_default():
    tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
    data = InputData()
    input_ph = tf.placeholder(tf.int32, [None, data.max_len])
    label_ph = tf.placeholder(tf.float32, [None, 6])
    with tf.Session() as sess:
      convolution_op = convolution(input_ph, data.num_chars, train=True)
      loss_op = loss(convolution_op, label_ph)
      train_op = minimize(loss_op)
      accuracy_op = accuracy(convolution_op, label_ph)
      saver = tf.train.Saver()
      load_checkpoint(sess, saver)
      for i in range(10000):
        label_, _, text_ = data.next_batch()
        _ = sess.run(train_op, feed_dict={input_ph: text_, label_ph: label_})
        if i % 10 == 0:
          loss_, accuracy_ = sess.run([loss_op, accuracy_op], feed_dict={input_ph: text_, label_ph: label_})
          print('global step: %04d, train loss: %01.7f, train accuracy %01.5f' % (i, loss_, accuracy_))
        if i % 100 == 0:
          loss_, accuracy_ = sess.run([loss_op, accuracy_op], feed_dict={input_ph: data.test_text, label_ph: data.test_label})
          print('Test loss: %s Test accuracy: %s' % (loss_, accuracy_))
          saver.save(sess, FLAGS.checkpoint_dir, global_step=i)


def convolution(input_, num_chars, embed_size=64, train=False):
  w = tf.get_variable("embedding", [num_chars, embed_size])
  embed = tf.gather(w, input_)
  convs = []
  for filter_size in (2,3,4,5):
    conv = tf.layers.conv1d(embed, 64, filter_size, 1, padding='SAME')
    # conv = tf.layers.batch_normalization(conv, training=train)
    conv = tf.nn.relu(conv)
    convs.append(conv)
  
  output = tf.concat(convs, axis=1)
  output = tf.contrib.layers.flatten(output)

  output = tf.layers.dense(output, 64)
  # output = tf.layers.batch_normalization(output, training=train)
  output = tf.nn.relu(output)

  output = tf.layers.dense(output, 6)
  output = tf.nn.softmax(output)
  return output


def loss(logits, labels):
  loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
  return loss


def minimize(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def predict():
  data = InputData(train=False)
  input_ph = tf.placeholder(tf.int32, [None, data.max_len])
  output_str = {
    0: '0 ~ 499円'
    , 1: '500 ~ 999円'
    , 2: '1000 ~ 4999円'
    , 3: '5000 ~ 9999円'
    , 4: '10000 ~ 49999円'
    , 5: '50000円以上'
  }
  with tf.Session() as sess:
    output = convolution(input_ph, data.num_chars, train=False)
    saver = tf.train.Saver()
    load_checkpoint(sess, saver)
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      vector = [data.sentence_to_vector(sentence)]
      logits = sess.run(output, feed_dict={input_ph: vector})
      index = np.argmax(logits[0])
      print(logits[0])
      print(output_str[index])
      sys.stdout.write("> ")
      sys.stdout.flush()
      sentence = sys.stdin.readline()



def load_checkpoint(sess, saver):
  if os.path.exists(FLAGS.checkpoint_dir + 'checkpoint'):
    print('restore parameters...')
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
  else:
    print('initirize parameters...')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


def main():
  if FLAGS.predict:
    predict()
  else:
    train()

if __name__ == '__main__':
  main()
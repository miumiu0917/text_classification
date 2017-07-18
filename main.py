from input_data import InputData
import tensorflow as tf

def main():
  data = InputData()
  input_ph = tf.placeholder(tf.int32, [None, data.max_len])
  label_ph = tf.placeholder(tf.float32, [None, 6])
  with tf.Session() as sess:
    pred_op = prediction(input_ph, data.num_chars)
    loss_op = loss(pred_op, label_ph)
    train_op = train(loss_op)
    accuracy_op = accuracy(pred_op, label_ph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(10000):
      label_, _, text_ = data.next_batch()
      _ = sess.run(train_op, feed_dict={input_ph: text_, label_ph: label_})
      if i % 10 == 0:
         loss_, accuracy_ = sess.run([loss_op, accuracy_op], feed_dict={input_ph: text_, label_ph: label_})
         print('train', loss_, accuracy_)
      if i % 100 == 0:
         loss_, accuracy_ = sess.run([loss_op, accuracy_op], feed_dict={input_ph: data.test_text, label_ph: data.test_label})
         print('test', loss_, accuracy_)


def prediction(input_, num_chars, embed_size=64):
  w = tf.get_variable("embedding", [num_chars, embed_size])
  embed = tf.gather(w, input_)
  convs = []
  for filter_size in (2,3,4,5):
    conv = tf.layers.conv1d(embed, 64, filter_size, 1, padding='SAME')
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.relu(conv)
    convs.append(conv)
  
  output = tf.concat(convs, axis=1)
  output = tf.contrib.layers.flatten(output)

  output = tf.layers.dense(output, 64)
  output = tf.layers.batch_normalization(output)
  output = tf.nn.relu(output)

  output = tf.layers.dense(output, 6)
  output = tf.nn.softmax(output)
  return output


def loss(logits, labels):
  loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits + 1e-10), reduction_indices=[1]))
  return loss


def train(loss):
    return tf.train.AdamOptimizer().minimize(loss)


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


if __name__ == '__main__':
  main()
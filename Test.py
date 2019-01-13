import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Hello World")
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))
print(sess.run([node1, node2]))

import tensorflow as tf
x = tf.Variable([1.0, 2.0, 5.0])
z = tf.math.sigmoid(x)
tf.print(x,[x])
tf.print(z,[z])

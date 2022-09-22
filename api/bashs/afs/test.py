import tensorflow.compat.v1 as tf

a = [1, 2, 3, 4, 5, 6, 7]
b = [1, 2, 3, 5, 6, 6, 7]

print(tf.equal(a, b))
print(tf.reduce_mean(tf.cast(tf.equal(a, b), tf.float32)))

import tensorflow as tf

@tf.function(experimental_compile=True)
def RSE(label, pred):
    num = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(label - pred)))
    print(num)
    den = tf.math.reduce_std(label, axis=None)
    print(den)
    return num/den

@tf.function(experimental_compile=True)
def SMAPE(label, pred):
    smape = tf.math.reduce_mean(2 * tf.math.abs(label - pred) / (tf.abs(label) + tf.abs(pred)))
    return smape
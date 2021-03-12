import tensorflow as tf

def RSE(label, pred):
    _label = tf.cast(label, dtype='float32')
    _pred  = tf.cast(pred, dtype='float32')

    num = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(_label - _pred)))
    den = tf.math.reduce_std(_label,axis=None)
    rse = num / den
    return rse

def SMAPE(label, pred):
    _label = tf.cast(label, dtype='float32')
    _pred  = tf.cast(pred, dtype='float32')

    smape = tf.math.reduce_mean(2 * tf.math.abs(_label - _pred) / (tf.abs(_label) + tf.abs(_pred)))
    return smape

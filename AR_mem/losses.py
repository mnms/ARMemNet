import tensorflow as tf
from tensorflow.keras.losses import Loss

class RSE(Loss):
    def call(self, label, pred):
        error = tf.math.reduce_sum((label - pred) ** 2) ** 0.5
        denom = tf.math.reduce_sum((label - tf.reduce_mean(label) ** 2) ** 0.5
        rse = error / denom
        return rse

class SMAPE(Loss):
    def call(self, label, pred):
        mape = tf.math.reduce_mean(tf.math.abs((label - pred)/label))
        smape = tf.math.reduce_mean(2 * tf.math.abs(label - pred) / (tf.abs(label) + tf.abs(pred)))
        return smape

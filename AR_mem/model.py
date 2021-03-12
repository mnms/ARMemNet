import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import *

#customAutoRegressive
class customAutoRegressive(tf.keras.layers.Layer):
    def __init__(self, ar_lambda, x_len, nfeatures):
        super(customAutoRegressive, self).__init__()
        self.ar_labda = ar_lambda
        self.x_len = x_len
        self.nfeatures = nfeatures
        self.initializer_glorot_uniform=tf.keras.initializers.GlorotUniform()
        self.initializer_zeros = tf.keras.initializers.Zeros()

    def build(self, input_shape) :
        self.w = self.add_weight(name="w", shape=[self.x_len, self.nfeatures], dtype='float32',\
                initializer=self.initializer_glorot_uniform)
        self.b = self.add_weight(name="bias", shape=[self.nfeatures], dtype='float32',\
                initializer=self.initializer_zeros)

    def call(self, inputs):
        _w = tf.expand_dims(self.w, axis=0)
        weighted = tf.math.reduce_sum(inputs * _w, axis=1) + self.b
        ar_loss = self.ar_labda * tf.math.reduce_sum(tf.math.square(self.w))
        return weighted, ar_loss

#customAttention
class customAttention(tf.keras.layers.Layer):
    def __init__(self, attention_size, mstep, nfeatures, regularizer):
        super(customAttention, self).__init__()
        self.attention_size = attention_size
        self.mstep = mstep
        self.nfeatures = nfeatures
        self.initializer_glorot_uniform = tf.keras.initializers.GlorotUniform()
        self.regularizer = regularizer
        self.Query = Dense(self.attention_size, \
                activation=None, use_bias=False, kernel_initializer=self.initializer_glorot_uniform, \
                kernel_regularizer = self.regularizer)
        self.Key_dense = Dense(self.attention_size, \
                activation=None,use_bias=False, kernel_initializer=self.initializer_glorot_uniform,\
                kernel_regularizer=self.regularizer)
        self.Projection_act = Activation(tf.nn.tanh)
        self.SimMatrix_dense = Dense(1, activation=None, \
                use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=self.regularizer)
        self.SimMatrix_softmax = Softmax(1)

    def build(self, input_shape):
        self.attention_bias = self.add_weight(name="attention_bias", shape=[self.attention_size], dtype='float32',\
                initializer=self.initializer_glorot_uniform)
        
    def call(self, inputs): #inputs[0]=inputs inputs[1]=memories
        x = tf.expand_dims(inputs[0], axis=1)
        query = self.Query(x)
        key = tf.reshape(inputs[1], shape=[-1, self.mstep, self.nfeatures])
        key = self.Key_dense(key)
        projection = query + key + self.attention_bias
        projection = self.Projection_act(projection)
        sim_matrix = self.SimMatrix_dense(projection)
        sim_matrix = self.SimMatrix_softmax(sim_matrix)
        sim_matrix = tf.transpose(sim_matrix, [0,2,1])
        context = tf.matmul(sim_matrix, key)
        context = tf.squeeze(context, axis=1)
        return context

# AR_memory
class ARMemNet(tf.keras.Model):
    def __init__(self, config):
        super(ARMemNet, self).__init__()
        self.config = config
        self.regularizer = regularizers.l2(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.auto_regressive_input  = customAutoRegressive(self.config.ar_lambda, self.config.x_len, self.config.nfeatures)
        self.auto_regressive_memory = customAutoRegressive(self.config.ar_lambda, self.config.x_len+1, self.config.nfeatures)
        self.attention = customAttention(self.config.attention_size, self.config.msteps, self.config.nfeatures, self.regularizer)
        self.prediction = Dense(self.config.nfeatures, \
                activation=tf.nn.tanh, use_bias=False, kernel_initializer='glorot_uniform')

    #def call(self, inputs, memories, training=True):
    #    input_ar, ar_loss_input = self.auto_regressive_input(inputs)
    #    memories = tf.concat(tf.split(memories, self.config.msteps, axis=1), axis=0)
    #    memory_ar, ar_loss_memories = self.auto_regressive_memory(memories)
    #    context=self.attention([input_ar, memory_ar])
    #    linear_inputs = tf.concat([input_ar, context], axis=1)
    #    predictions = self.prediction(linear_inputs)
    #    return predictions

    def call(self, inputs, training=True): #inputs[0]=inputs inputs[1]=memories
        input_ar, ar_loss_input = self.auto_regressive_input(inputs[0])
        memories = tf.concat(tf.split(inputs[1], self.config.msteps, axis=1), axis=0)
        memory_ar, ar_loss_memories = self.auto_regressive_memory(memories)
        context=self.attention([input_ar, memory_ar])
        linear_inputs = tf.concat([input_ar, context], axis=1)
        predictions = self.prediction(linear_inputs)
        return predictions


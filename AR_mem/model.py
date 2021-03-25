import tensorflow as tf
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import *

@tf.keras.utils.register_keras_serializable()
class CustomAutoRegressive(tf.keras.layers.Layer):
    def __init__(self, ar_lambda, x_len, nfeatures, **kwargs):
        super(CustomAutoRegressive, self).__init__(**kwargs)
        self.ar_lambda = ar_lambda
        self.x_len = x_len
        self.nfeatures = nfeatures
        self.initializer_glorot_uniform = tf.keras.initializers.GlorotUniform()
        self.initializer_zeros = tf.keras.initializers.Zeros()

    def build(self, input_shape) :
        self.w = self.add_weight(name='w', shape=[self.x_len, self.nfeatures], dtype='float32',\
                initializer=self.initializer_glorot_uniform)
        self.b = self.add_weight(name='bias', shape=[self.nfeatures], dtype='float32',\
                initializer=self.initializer_zeros)

    @tf.function(experimental_compile=True)
    def call(self, inputs):
        _w = tf.expand_dims(self.w, axis=0)
        weighted = tf.math.reduce_sum(inputs * _w, axis=1) + self.b
        ar_loss = self.ar_lambda * tf.math.reduce_sum(tf.math.square(self.w))
        return weighted, ar_loss

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'ar_lambda': self.ar_lambda,
            'x_len': self.x_len,
            'nfeatures': self.nfeatures,
        }

@tf.keras.utils.register_keras_serializable()
class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, attention_size, mstep, nfeatures, regularizer, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.mstep = mstep
        self.nfeatures = nfeatures
        self.initializer_glorot_uniform = tf.keras.initializers.GlorotUniform()
        self.regularizer = regularizer

    def build(self, input_shape):
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
        self.attention_bias = self.add_weight(name='attention_bias', shape=[self.attention_size], dtype='float32',\
                initializer=self.initializer_glorot_uniform)

    @tf.function(experimental_compile=True)
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

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'attention_size': self.attention_size,
            'mstep': self.mstep,
            'nfeatures': self.nfeatures,
            'regularizer': self.regularizer,
        }

# AR_memory
@tf.keras.utils.register_keras_serializable()
class ARMemNet(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(ARMemNet, self).__init__(**kwargs)
        self.config = config
        self.regularizer = regularizers.l2(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.auto_regressive_input  = CustomAutoRegressive(self.config.ar_lambda, self.config.x_len, self.config.nfeatures)
        self.auto_regressive_memory = CustomAutoRegressive(self.config.ar_lambda, self.config.x_len+1, self.config.nfeatures)
        self.attention = CustomAttention(self.config.attention_size, self.config.msteps, self.config.nfeatures, self.regularizer)
        self.prediction = Dense(self.config.nfeatures, \
                activation=tf.nn.tanh, use_bias=False, kernel_initializer='glorot_uniform')

    @tf.function(experimental_compile=True)
    def call(self, inputs, training=True): # inputs == (input_x, memory_m)
        input_ar, ar_loss_input = self.auto_regressive_input(inputs[0])
        memories = tf.concat(tf.split(inputs[1], self.config.msteps, axis=1), axis=0)
        memory_ar, ar_loss_memories = self.auto_regressive_memory(memories)
        context = self.attention([input_ar, memory_ar])
        linear_inputs = tf.concat([input_ar, context], axis=1)
        predictions = self.prediction(linear_inputs)
        return predictions

    def get_config(self):
        return {
            'config': self.config,
            'regularizer': self.regularizer,
            'auto_regressive_input': self.auto_regressive_input,
            'auto_regressive_memory': self.auto_regressive_memory,
            'attention': self.attention,
            'prediction': self.prediction,
            'layers': self.layers,
            'name': self.config.model
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = ARMemNet(config)
    print("done")

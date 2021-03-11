import tensorflow as tf
from tensorflow.keras import regularizers, initializers, Model
from tensorflow.keras.layers import *

# AR_memory
class AR_Mem(Model):
    def __init__(self, config):
        super(AR_Mem, self).__init__()
        self.config = config
        self.regularizer = regularizers.l2(self.config.l2_lambda)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.dropout = tf.constant(dtype=tf.float32, name='dropout')
        self.auto_regressive = customAutoRegressive(self.config.ar_lambda, self.config.x_len, self.config.nfeatures)
        self.attention = customAttention(self.config.ar_lambda, self.config.msteps, self.config.nfeatures, self.regularizer)
        self.prediction = Dense(self.config.nfeatures, \
                activation=tf.nn.tanh, use_bias=False, kernel_initializer='glorot_uniform')

    class customAutoRegressive(Layer):
        def __init__(self, ar_lambda, x_len, nfeatures):
            super(AutoRegressive, self).__init__()
            self.ar_labda = ar_lambda
            self.x_len = x_len
            self.nfeatures = nfeatures

        def build(self):
            w_init = initializers.GlorotUniform()
            self.w = tf.Variable(initial_value=w_init(shape=(self.x_len, self.nfeatures]), dtype='float32'), trainable=True, name='w')
            b_init = initializers.Zeros()
            self.b = tf.Variable(initial_value=b_init(shape=(self.nfeatures), dtype='float32'), trainable=True, name='bias')

        def call(self, inputs):
            _w = tf.expand_dims(self.w, exis=0)
            weighted = tf.math.reduce_sum(inputs * _w, axis=1) + bias
            ar_loss = self.ar_labda * tf.math.reduce_sum(tf.math.square(w))
            return weighted, ar_loss

    class customAttention(Layer):
        def __init__(self, attention_size, mstep, nfeatures, regularizer):
            super(Attention, self).__init__()
            self.attention_size = attention_size
            self.mstep = mstep
            self.nfeatures = nfeatures
            self.regularizer = regularizer
            self.Query = Dense(self.config.attention_size, \
                    activation=None, use_bias=False, kernel_initializer='glorot_uniform', \
                    kernel_regularizer = self.regularizer)
            self.Key_reshape = Reshape((-1, self.msteps, self.nfeatures))
            self.Key_dense = Dense(self.config.attention_size, \
                    activation=None,use_bias=False, kernel_initializer='glorot_uniform',\
                    kernel_regularizer=self.regularizer)
            self.Projection_add = Add()
            self.Projection_act = Activation(tf.nn.tanh)
            self.SimMatrix_dense = Dense(1, activation=None \
                    use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer = self.regularizer)
            self.SimMatrix_softmax = Softmax(1)
            self.Context_matmul = Multiply()

        def build(self):
            self.attention_bias = tf.Variable(initializers.GlorotUniform(shape=[self.attention_size]), name='attention_bias')
            
        def call(self, inputs, memories):
            x = tf.expand_dims(inputs, axis=1)
            query = self.Queary(x)
            key = self.Key_reshape(memories)
            key = self.Key_dense(key)
            projection = self.Projection_add(query, key, bias)
            projection = self.Projection_act(projection)
            sim_matrix = self.SimMatrix_dense(projection)
            sim_matrix = self.SimMatrix_softmax(sim_matrix)
            sim_matrix = tf.transpose(sim_matrix, [0,2,1])
            context = self.Context_matmul(sim_matrix, key)
            context = tf.squeeze(context, axis=1)
            return context

    def call( self, inputs, memories, training=True):
        input_ar, ar_loss_input = self.auto_regressive(inputs)
        memories = tf.concat(tf.split(memories, self.config.msteps, axis=1), axis=0)
        memory_ar, ar_loss_memories = self.auto_regressive(memories)

        context=self.attention(input_ar, memory_ar)
        linear_inputs = tf.concat([input_ar, context], axis=1)
        predictions = self.prediction(linear_input)
        return predictions


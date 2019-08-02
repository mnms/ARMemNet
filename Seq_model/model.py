import tensorflow as tf
from tensorflow.contrib import layers


class Model(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.regularizer = layers.l2_regularizer(self.config.l2_lambda)
        self.sess = None
        self.saver = None
        self._build_model()

    def _build_model(self):
        self.add_placeholder()

        # self.total_loss = []
        self.total_states = []
        total_pred_ = []

        for i in range(self.config.ncells):
            with tf.variable_scope("gru_encoder_{}".format(i)):
                gru_outputs = self.gru(self.input_x[:, i, :, :], scope="short_gru")  # [b, t, d]   
                final_states = gru_outputs[:,-1,:]

                pred = tf.layers.dense(final_states, self.config.nfeatures, tf.nn.tanh, kernel_initializer=layers.xavier_initializer(), use_bias=False)
                # loss = tf.losses.mean_squared_error(labels=self.targets, predictions=pred)

                total_pred_.append(pred)
                self.total_states.append(final_states)
                # self.total_loss.append(loss)

        self.total_pred = tf.stack(total_pred_, axis=1)
        self.stacked_total_states = tf.stack(self.total_states, axis=1)

        self.total_pred = tf.reshape(self.total_pred, [-1, self.config.ncells * self.config.nfeatures], name="pred_reshape")

        self.targets_ = tf.reshape(self.targets, [-1, self.config.ncells * self.config.nfeatures], name="total_reshape")

        self.loss = tf.losses.mean_squared_error(labels=self.targets_, predictions=self.total_pred)

        error = tf.reduce_sum((self.targets_ - self.total_pred) ** 2) ** 0.5
        denom = tf.reduce_sum((self.targets_ - tf.reduce_mean(self.targets)) ** 2) ** 0.5

        self.rse = error / denom
        self.mae = tf.reduce_mean(tf.abs(self.targets_ - self.total_pred))
        self.mape = tf.reduce_mean(tf.abs((self.targets_ - self.total_pred) / self.targets_))
        self.smape = tf.reduce_mean(2*tf.abs(self.targets_ - self.total_pred)/(tf.abs(self.targets_)+tf.abs(self.total_pred)))

        '''
        if self.config.l2_lambda > 0:
            reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = layers.apply_regularization(self.regularizer, reg_vars)
            self.loss += reg_term
        '''
        #self.loss += ar_loss

        self.add_train_op()
        self.initialize_session()

    def add_placeholder(self):
        self.input_x = tf.placeholder(shape=[None, self.config.ncells, None, self.config.nfeatures], dtype=tf.float32, name="x")                             
        self.targets = tf.placeholder(shape=[None, self.config.ncells, self.config.nfeatures], dtype=tf.float32, name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

    def gru(self, inputs, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.nn.rnn_cell.GRUCell(self.config.gru_size, activation=tf.nn.relu)
            outputs, final = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            # output = tf.nn.dropout(outputs, self.dropout)
        return outputs

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.config.lr)
        # opt = tf.train.RMSPropOptimizer(self.config.lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.config.clip)
        self.train_op = opt.apply_gradients(zip(grads, vars), global_step=self.global_step)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        if not self.config.allow_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self, model_name):
        """Saves session = weights"""
        self.saver.save(self.sess, model_name)

    def restore_session(self, dir_model):
        """Reload weights into session
        Args:
            sess: tf.Session()
            dir_model: dir with weights
        """
        self.saver.restore(self.sess, tf.train.latest_checkpoint(dir_model))

    def train(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
            self.targets: targets,
            self.dropout: self.config.dropout
        }
        output_feed = [self.train_op, self.loss, self.rse, self.smape, self.mae, self.global_step]
        _, loss, rse, smape, mae, step = self.sess.run(output_feed, feed_dict)

        return loss, rse, smape, mae, step


    def eval(self, input_x, targets):
        feed_dict = {
            self.input_x: input_x,
            self.targets: targets,
            self.dropout: 1.0
        }
        output_feed = [self.total_pred, self.loss, self.rse, self.smape, self.mae]
        total_pred, loss, rse, smape, mae = self.sess.run(output_feed, feed_dict)

        return total_pred, loss, rse, smape, mae

    def extract(self, input_x):
        feed_dict = {
            self.input_x: input_x,
            self.dropout: 1.0
        }
        output_feed = [self.stacked_total_states]
        [stacked_total_states] = self.sess.run(output_feed, feed_dict)
        
        return stacked_total_states


if __name__ == "__main__":
    from config import Config
    config = Config()
    model = Model(config)
    print("done")


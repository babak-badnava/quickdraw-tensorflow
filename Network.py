import numpy as np
import tensorflow as tf

from pre_processing.Config import Config
from QueueLoader import QueueLoader

class Network:
    
    def __init__(self):
        self.model_name = Config.NETWORK_NAME

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.nb_class = Config.NB_CLASSES
        self.batch_size = Config.BATCH_SIZE
        
        self.learning_rate = Config.LEARNING_RATE_START
        
        train_tfrecords_filename = "./data/tf/train/" + "train" + ".tfrecords"
        test_tfrecords_filename = "./data/tf/test/" + "test" + ".tfrecords"
        
        
        
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.train_q_loader = QueueLoader(train_tfrecords_filename)
            self.test_q_loader = QueueLoader(test_tfrecords_filename)
            with tf.device(Config.DEVICE):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
                self.sess.run(init_op)
                
                self.coord = tf.train.Coordinator()
                self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
                

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                
                self.graph.finalize()
                    
    def _create_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_width, self.img_height, 1], name='X')
        self.y = tf.placeholder(tf.int64, shape=[None], name='y')
        
        
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.batch_size * Config.ANNEALING_EPOCH_COUNT, 0.9, staircase=True)
        
        
        self.y_one_hot = tf.one_hot(self.y, self.nb_class, 1., 0.)
        
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step_inc = self.global_step.assign_add(1)
        
        self.l1 = self.conv2d_layer(self.x, 8, 16, 'conv1', strides=[1, 4, 4, 1])
        
        _input = self.l1
        
        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]
        
        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')
        
        self.output = self.dense_layer(self.d1, self.nb_class, 'output', func=tf.nn.softmax)
        
        self.prediction = tf.argmax(self.output, axis=1, name='prediction')
        
#         self.label_minus_predicted = tf.subtract(self.y_one_hot, self.output)
#         self.loss = tf.square(tf.reduce_sum(self.label_minus_predicted), 'loss')
        self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.output)
        
        
        self.compare = tf.equal(self.prediction, self.y)
        self.true_predicted_count = tf.reduce_sum(tf.cast(self.compare, tf.int32))
        self.acc = tf.reduce_mean(tf.cast(self.compare, tf.float16))
        
        
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_optimizer = self.optimizer.minimize(self.loss, global_step=self.global_step)
            
    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        summaries.append(tf.summary.scalar("loss", self.loss))
        summaries.append(tf.summary.scalar("learning rate", self.var_learning_rate))
        
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        summaries.append(tf.summary.histogram("activation_n1", self.l1))
        summaries.append(tf.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.summary.histogram("activation_p", self.output))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

        
    def __get_base_feed_dict(self):
        return {self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step
        
    def train_batch(self):
        images, labels = self.sess.run([self.train_q_loader.images, self.train_q_loader.labels], {})

        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: images, self.y: labels})
        self.sess.run(self.train_optimizer, feed_dict=feed_dict)
    
    def test_batch(self):
        images, labels = self.sess.run([self.test_q_loader.images, self.test_q_loader.labels], {})

        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: images, self.y: labels})
        if Config.TENSORBOARD:
            fetch_list = [self.loss, self.acc, self.summary_op, self.global_step_inc]
            loss, acc, summary, step = self.sess.run(fetch_list, feed_dict=feed_dict)
            self.log_writer.add_summary(summary, step)
        else:
            fetch_list = [self.loss, self.acc, self.global_step_inc]
            loss, acc, step = self.sess.run(fetch_list, feed_dict=feed_dict)
        
        return loss, acc
    
    def predict(self, x):
        feed_dict = {self.x: x}
        return self.sess.run(self.prediction, feed_dict)
        
    def _checkpoint_filename(self, episode):
        return 'logs/checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
    
    
    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output
    
    def finish(self):
        self.coord.request_stop()
        self.coord.join(self.threads)


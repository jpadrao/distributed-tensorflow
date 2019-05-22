import threading
import pickle
import socket
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import time
import network


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def test(model):
    test_accuracy = tfe.metrics.Accuracy()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_test = x_test.reshape(10000, 28, 28, 1)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(100)

    for (x, y) in test_dataset:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, 'int32')

        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def init():  # initiate weights

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    optimizer = tf.train.AdamOptimizer()
    batch_size = 1
    num_epochs = 1

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 28, 28, 1)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(batch_size)

    # ------------------------------------

    # print('warming weights')

    # init weights
    for epoch in range(num_epochs):

        for x, y in dataset:
            x = tf.cast(x, tf.float32)
            y = tf.cast(y, 'int32')

            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            break

    return model


class Server(threading.Thread):

    def __init__(self, socket, init_weights, model, model_lock, barrier):
        threading.Thread.__init__(self)
        self.socket = socket
        self.init_weights = init_weights
        self.model = model
        self.model_lock = model_lock
        self.barrier = barrier

        self.optimizer = tf.train.AdamOptimizer()

    def run(self):

        global start_time
        global stoped_clock

        while True:

            msg = network.recv_msg(self.socket)

            if msg is not None:

                (header, payload) = pickle.loads(msg)

                if header == 'start':

                    print('start')

                    self.barrier.wait()  # wait for all nodes to send start message

                    self.model_lock.acquire()
                    if start_time == 0:
                        start_time = time.time()
                    self.model_lock.release()

                    p = pickle.dumps(self.init_weights, -1)
                    self.barrier.wait()  # wait until all workers are ready to receive the initial weights
                    start_time = time.time()
                    network.send_msg(self.socket, p)  # send initial weights to the worker

                elif header == 'train':  # received new gradients

                    grads_and_vars = payload

                    if FLAGS.sync_mode == 'sync':
                        self.model_lock.acquire()
                        self.optimizer.apply_gradients(zip(grads_and_vars, self.model.trainable_variables))
                        self.model_lock.release()

                        self.barrier.wait()  # wait for all workers
                        new_weights = self.model.get_weights()

                    else:  # no need to wait for workers here
                        self.model_lock.acquire()
                        self.optimizer.apply_gradients(zip(grads_and_vars, self.model.trainable_variables))
                        new_weights = self.model.get_weights()
                        self.model_lock.release()

                    p = pickle.dumps(new_weights, -1)
                    network.send_msg(self.socket, p)

                    pass

                elif header == 'end':

                    print('end')

                    self.barrier.wait()  # wait for all workers to finish

                    self.model_lock.acquire()
                    if not stoped_clock:
                        elapsed_time = time.time() - start_time
                        stoped_clock = True
                        print('elapsed time = ' + str(elapsed_time))
                    self.model_lock.release()
                    break

                else:
                    print('Received unknown message')

            else:
                print('Connection closed')
                self.socket.close()

        pass


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('sync_mode', 'sync', 'sync or async')
    tf.app.flags.DEFINE_integer('n_nodes', 0, 'number of nodes')
    tf.app.flags.DEFINE_string('address', '127.0.0.1', 'address of the server')
    tf.app.flags.DEFINE_integer('n_cores', 1, 'number of cores available to the parameter server')
    tf.app.flags.DEFINE_string('output_file', None, 'file to output the logs from the execution')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.n_cores, inter_op_parallelism_threads=FLAGS.n_cores,
                            allow_soft_placement=True, device_count={'CPU': FLAGS.n_cores})
    tf.enable_eager_execution(config=config)

    model_init = init()  # init weights
    model_lock_init = threading.Lock()
    init_weights = model_init.get_weights()
    barrier_init = threading.Barrier(FLAGS.n_nodes)
    stoped_clock = False
    start_time = 0

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind((FLAGS.address, 6000))
    serversocket.listen(50)

    print('accepting')
    workers = []

    while True:
        (clientsocket, address) = serversocket.accept()  # waiting for workers to connect
        ct = Server(clientsocket, init_weights, model_init, model_lock_init, barrier_init)
        ct.start()
        workers.append(ct)
        if len(workers) == FLAGS.n_nodes:
            break

    for w in workers:
        w.join()

    test(model_init)

import argparse
import pickle
import socket
import struct
import threading
import time

import network
import psutil
import tensorflow as tf


def loss(model, x, y):
    y_ = model(x)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def test(model, test_dataset):
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, 'int32')

        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

    return float(format(test_accuracy.result()))


class Server(threading.Thread):

    def __init__(self, socket, init_weights, model, model_lock, barrier, sync_mode, optimizer=None, sup_socket=None):
        threading.Thread.__init__(self)
        self.sync_mode = sync_mode
        self.sup_socket = sup_socket
        self.socket = socket
        self.init_weights = init_weights
        self.model = model
        self.model_lock = model_lock
        self.barrier = barrier

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

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

                    network.send_msg(self.socket, p)  # send initial weights to the worker

                elif header == 'train':  # received new gradients

                    grads_and_vars = payload

                    if self.sync_mode == 'sync':
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

                        if self.sup_socket is not None:
                            p = pickle.dumps(('done', elapsed_time), -1)
                            msg = struct.pack('>I', len(p)) + p
                            self.sup_socket.sendall(msg)

                    self.model_lock.release()
                    break

                else:
                    print('Received unknown message')

            else:
                print('Connection closed')
                self.socket.close()

        pass


def start(sync_mode, model_fn, dataset_fn, n_nodes, optimizer=None, supervisor_address=None, cpu_affinity=False,
          server_address='127.0.0.1'):
    global stoped_clock
    global start_time

    if cpu_affinity:
        p = psutil.Process()
        p.cpu_affinity([0])  # TODO change from 0 to qq cena

    model = model_fn()  # get model
    model_lock = threading.Lock()  # create distributed lock
    init_weights = model.get_weights()  # get initial model weights
    barrier = threading.Barrier(n_nodes)  # create barrier, only used for synchronous training
    stoped_clock = False
    start_time = 0

    sup_socket = None
    if supervisor_address is not None:  # try to connect to supervisor processes, not important
        sup_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sup_socket.connect((server_address, 4000))

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind((server_address, 6000))  # bind server address
    serversocket.listen(50)

    print('accepting')
    workers = []

    while True:
        (clientsocket, address) = serversocket.accept()  # waiting for workers to connect
        ct = Server(clientsocket, init_weights, model, model_lock, barrier, sync_mode, optimizer=optimizer)
        ct.start()  # start new server worker
        workers.append(ct)
        if len(workers) == n_nodes:
            break

    for w in workers:
        w.join()

    test_dataset = dataset_fn(100, type='test', shard=False)
    result = test(model, test_dataset)  # test the model

    if supervisor_address is not None:  # send results to the supervisor
        p = pickle.dumps(('results', result), -1)
        msg = struct.pack('>I', len(p)) + p
        sup_socket.sendall(msg)

        sup_socket.close()


def model_fn():
    raise NotImplementedError("model function not implemented")


def dataset_fn(batch_size, type='train', shard=True, index=0, buffer_size=10000):
    raise NotImplementedError("dataset function not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', action='store',
                        dest='n_nodes', type=int,
                        help='number of nodes on the network')

    parser.add_argument('-s', action='store',
                        dest='server_address', default='127.0.0.1',
                        help='server address')

    parser.add_argument('-sm', action='store',
                        dest='sync_mode', default='sync',
                        help='syn mode: sync or async')

    results = parser.parse_args()

    n_nodes = results.n_nodes
    server_address = results.server_address
    sync_mode = results.sync_mode

    start(sync_mode=sync_mode, model_fn=model_fn, dataset_fn=dataset_fn, n_nodes=n_nodes, optimizer=None,)

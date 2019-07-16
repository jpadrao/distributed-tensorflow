import argparse
import json
import logging
import os
import pickle
import socket
import struct
import time

import numpy as np
import psutil
import tensorflow as tf


# TODO change this function to other file
def send_msg_socket(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def train(strategy, model_fn=None, dataset_fn=None, index=0, n_nodes=0, batch_size=100, supervisor_address=None):

    train_dataset = dataset_fn(batch_size, type='train', shard=True, index=index, reshape=False)
    test_dataset = dataset_fn(batch_size, type='test', shard=True, index=index, reshape=False)

    with strategy.scope():
        model = model_fn()

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

    sup_socket = None
    if supervisor_address is not None:
        sup_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sup_socket.connect(('localhost', 4000))
        p = pickle.dumps('start', -1)
        send_msg_socket(sup_socket, p)

    s_time = time.time()
    model.fit(x=train_dataset, epochs=1, verbose=0)
    e_time = time.time() - s_time

    if supervisor_address is not None:
        p = pickle.dumps('done', -1)
        send_msg_socket(sup_socket, p)

    print('-----------------------------------------')
    print('elapsed time: ' + str(e_time) + ' seconds')
    print('-----------------------------------------')

    results = model.evaluate(x=test_dataset)
    print('eval: ' + str(results[1]))

    if supervisor_address is not None:
        p = pickle.dumps(('results', results[1]), -1)
        send_msg_socket(sup_socket, p)


def init_train(n_nodes, index, worker_addresses, model_fn=None, dataset_fn=None, batch_size=100, cpu_affinity=False):

    if cpu_affinity:
        p = psutil.Process()
        p.cpu_affinity([index])  # cpu 0 is parameter server

    logging.disable(logging.INFO)
    logging.disable(logging.WARNING)

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': worker_addresses
        },
        'task': {'type': 'worker', 'index': index}
    })

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.
                                                                      CollectiveCommunication.RING)

    train(strategy, model_fn=model_fn, dataset_fn=dataset_fn, index=index, n_nodes=n_nodes, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', action='store',
                        dest='n_nodes', type=int,
                        help='number of node on the network')

    parser.add_argument('-b', action="store",
                        dest="batch_size", type=int, default=100,
                        help='batch size')

    parser.add_argument('-i', action="store",
                        dest="index", type=int, default=0,
                        help='index')

    parser.add_argument('-w', action="store",
                        dest='worker_addresses',
                        help='worker addresses')

    results = parser.parse_args()

    n_nodes = results.n_nodes
    batch_size = results.batch_size
    index = results.index
    worker_addresses = results.worker_addresses

    buffer_size = 10000

    logging.disable(logging.INFO)
    logging.disable(logging.WARNING)

    worker_addresses = [x.strip() for x in worker_addresses.split(',')]

    init_train(n_nodes, batch_size, index, worker_addresses)

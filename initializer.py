import argparse
import multiprocessing

import numpy as np
import tensorflow as tf

from centralized import client as centralized_client
from centralized import server as parameter_server
from decentralized.native import dist_keras
from decentralized.graph import node


def model_fn():  # example function
    # specify shapes to initialize models
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return model


def dataset_fn(batch_size, type='train', shard=True, index=0, buffer_size=10000, reshape=True):  # example function
    if type == 'train':
        ((images, labels), (_x, _y)) = tf.keras.datasets.mnist.load_data()

        if reshape:
            images = images.reshape(60000, 28, 28, 1)

    elif type == 'test':
        ((_x, _y), (images, labels)) = tf.keras.datasets.mnist.load_data()

        if reshape:
            images = images.reshape(10000, 28, 28, 1)

    else:
        raise ValueError('\"' + str(type) + '\" is an invalid dataset type')

    images = images / np.float32(255)
    labels = labels.astype(np.int32)

    if shard:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shard(n_nodes, index).batch(batch_size) \
            .shuffle(buffer_size)

    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size).shuffle(buffer_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard = False

    dataset = dataset.with_options(options)

    return dataset


# argparse helper function
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', action='store',
                        dest='mode', default=None,
                        help='training mode: centralized | decentralized')

    parser.add_argument('-cs', action='store',
                        dest='centralized_strategy', default=None,
                        help='centralized strategy: sync | async')

    parser.add_argument('-ds', action='store',
                        dest='decentralized_strategy', default=None,
                        help='decentralized strategy: graph | keras | custom')

    parser.add_argument('-n', action="store",
                        dest="n_nodes", type=int,
                        help='number of worker nodes on the network')

    # parser.add_argument('-d', action="store",
    #                     dest="degree", type=int,
    #                     help='degree of the network (only used if mode == graph)')

    parser.add_argument('-b', action="store",
                        dest="batch_size", type=int, default=100,
                        help='batch size')

    parser.add_argument('-ca', action="store",
                        dest="cpu_affinity", type=str2bool, default=False,
                        help='restrain each process to only 1 cpu core (y/n)')

    arg_results = parser.parse_args()

    mode = arg_results.mode
    centralized_strategy = arg_results.centralized_strategy
    decentralized_strategy = arg_results.decentralized_strategy
    n_nodes = arg_results.n_nodes
    # degree = arg_results.degree
    batch_size = arg_results.batch_size
    cpu_affinity = arg_results.cpu_affinity

    if mode == 'centralized':

        if centralized_strategy == 'sync' or centralized_strategy == 'async':

            ps = multiprocessing.Process(target=parameter_server.start, args=[centralized_strategy, model_fn,
                                                                              dataset_fn, n_nodes],
                                         kwargs={'cpu_affinity': cpu_affinity})
            ps.start()

            nodes = []
            for index in range(0, n_nodes):
                p = multiprocessing.Process(target=centralized_client.train, args=[batch_size],
                                            kwargs={'index': index, 'model_fn': model_fn, 'dataset_fn': dataset_fn,
                                                    'cpu_affinity': cpu_affinity})
                p.start()
                nodes.append(p)
    elif mode == 'decentralized':

        if decentralized_strategy == 'keras':

            start_port = 6000
            worker_addresses = []
            for d in range(0, n_nodes):
                worker_addresses.append('localhost:' + str(start_port + d))

            for index in range(0, n_nodes):
                p = multiprocessing.Process(target=dist_keras.init_train, args=[n_nodes, index, worker_addresses],
                                            kwargs={'model_fn': model_fn, 'dataset_fn': dataset_fn,
                                                    'batch_size': batch_size, 'cpu_affinity': cpu_affinity})
                p.start()

        elif decentralized_strategy == 'graph':



            pass

    else:
        raise ValueError('Invalid mode')

import argparse
import pickle
import socket
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


def test(model):
    test_accuracy = tf.keras.metrics.Accuracy()

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_test = x_test.reshape(10000, 28, 28, 1)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(500)

    for (x, y) in test_dataset:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, 'int32')

        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def train(batch_size, model_fn=None, dataset_fn=None, num_epochs=1, index=0, n_nodes=0, address='127.0.0.1',
          cpu_affinity=False):

    if cpu_affinity:
        p = psutil.Process()
        p.cpu_affinity([index + 1])  # cpu 0 is parameter server

    batch_size = int(batch_size)
    sock = None

    is_connected = False
    while not is_connected:  # connect to the parameter server
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((address, 6000))
            is_connected = True
        except ConnectionRefusedError:
            time.sleep(1)

    dataset = dataset_fn(batch_size, index=index)  # get dataset
    model = model_fn()  # get model

    p = pickle.dumps(('start', 'payload'), -1)
    network.send_msg(sock, p)  # send initial message to parameter server

    p = network.recv_msg(sock)  # wait and receive initial weights
    init_weights = pickle.loads(p)
    model.set_weights(init_weights)  # set initial weights

    iteration = 0

    for epoch in range(num_epochs):

        for x, y in dataset:

            x = tf.cast(x, tf.float32)
            y = tf.cast(y, 'int32')

            loss_value, grads_and_vars = grad(model, x, y)  # calculate gradients

            p = pickle.dumps(('train', grads_and_vars), -1)
            network.send_msg(sock, p)  # send new gradients to parameter server

            p = network.recv_msg(sock)  # wait and receive new weights from parameter server
            weights = pickle.loads(p)
            model.set_weights(weights)  # load new weights to the model

            if iteration % 50 == 0:
                print(iteration)

            iteration += 1

    p = pickle.dumps(('end', 'end'), -1)
    network.send_msg(sock, p)
    sock.close()

    # test(model)


def model_fn():
    raise NotImplementedError("model function not implemented")


def dataset_fn(batch_size, type='train', shard=True, index=0, buffer_size=10000):
    raise NotImplementedError("dataset function not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ps', action='store',
                        dest='index', type=int,
                        help='node position on the network')

    parser.add_argument('-n', action='store',
                        dest='n_nodes', type=int,
                        help='number of nodes on the network')

    parser.add_argument('-b', action="store",
                        dest="batch_size", type=int, default=100,
                        help='batch size')

    parser.add_argument('-ba', action='store',
                        dest='base_address', default='127.0.0.',
                        help='base node address')

    parser.add_argument('-sa', action='store',
                        dest='supervisor_address', default='127.0.0.1',
                        help='supervisor address')

    parser.add_argument('-s', action='store',
                        dest='server_address', default='127.0.0.1',
                        help='server address')

    results = parser.parse_args()

    index = results.index
    n_nodes = results.n_nodes
    batch_size = results.batch_size
    server_address = results.server_address

    train(batch_size, model_fn=model_fn, dataset_fn=dataset_fn, index=index, n_nodes=n_nodes, address=server_address)

import os
import pickle
import socket
import tensorflow as tf
import multiprocessing
import tensorflow.contrib.eager as tfe
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
    test_dataset = test_dataset.batch(500)

    for (x, y) in test_dataset:
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, 'int32')

        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def run(n_sequence):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((FLAGS.address, 6000))

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 28, 28, 1)

    if FLAGS.split_dataset:  # split dataset
        n_batches = 60000 / FLAGS.batch_size
        n_batches_per_node = n_batches / FLAGS.n_nodes

        start = int(n_batches_per_node * FLAGS.batch_size * n_sequence)
        end = int(n_batches_per_node * FLAGS.batch_size * (n_sequence + 1))

        x_train = x_train[start:end]  # split dataset
        y_train = y_train[start:end]  # split dataset

        print("[" + str(n_sequence) + '] Start training [' + str(start) + '-' + str(end) + ']')

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(FLAGS.batch_size)

    # define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    p = pickle.dumps(('start', 'payload'), -1)
    network.send_msg(sock, p)  # send initial message to parameter server

    p = network.recv_msg(sock)  # wait and receive inital weights
    init_weights = pickle.loads(p)
    model.set_weights(init_weights)  # set initial weights

    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    train_loss_results = []
    train_accuracy_results = []

    iteration = 0

    for epoch in range(FLAGS.n_epochs):

        for x, y in dataset:

            x = tf.cast(x, tf.float32)
            y = tf.cast(y, 'int32')

            # Optimize the model

            loss_value, grads_and_vars = grad(model, x, y)  # calculate gradients

            p = pickle.dumps(('train', grads_and_vars), -1)
            network.send_msg(sock, p)  # send new gradients to parameter server

            p = network.recv_msg(sock)  # wait and receive new weights from parameter server
            weights = pickle.loads(p)
            model.set_weights(weights)  # load new weights to the model

            epoch_loss_avg(loss_value)
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

            format(epoch_accuracy.result())

            if iteration % 50 == 0:
                print(iteration)

            iteration += 1

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    p = pickle.dumps(('end', 'end'), -1)
    network.send_msg(sock, p)
    sock.close()

    x_train = []
    y_train = []

    test(model)


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_integer('batch_size', 100, 'size of the batch')
    tf.app.flags.DEFINE_integer('n_epochs', 1, 'number of epochs')
    tf.app.flags.DEFINE_integer('n_nodes', 0, 'number of nodes')
    tf.app.flags.DEFINE_string('address', '127.0.0.1', 'address of the server')
    tf.app.flags.DEFINE_string('output_file', None, 'file to output the logs from the execution')
    tf.app.flags.DEFINE_integer('split_dataset', 1, 'If true, the dataset will be splited')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                            allow_soft_placement=True, device_count={'CPU': 1})
    tf.enable_eager_execution(config=config)

    workers = []

    for i in range(0, FLAGS.n_nodes):
        p = multiprocessing.Process(target=run, args=[i])
        p.start()  # start an independent node
        workers.append(p)

    for x in workers:
        x.join()

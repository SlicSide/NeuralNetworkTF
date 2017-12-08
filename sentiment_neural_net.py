import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np

#mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# What one_hot is:
'''
means that in this case for the number class 0-9 only one class is "hot". The others are not. Examples:
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]
5 = [0,0,0,0,0,1,0,0,0,0]
6 = [0,0,0,0,0,0,1,0,0,0]
7 = [0,0,0,0,0,0,0,1,0,0]
8 = [0,0,0,0,0,0,0,0,1,0]
9 = [0,0,0,0,0,0,0,0,0,1]
'''

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('res/pos.txt', 'res/neg.txt')

# defining the number of units per hidden-layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# defining the number of digit classes for MNIST and the minibatch size
n_classes = 2
batch_size = 100

# defining learining rate and number of epochs
learning_rate = 0.001
n_epochs = 100

# defining the input and output layer
'''
Input Layer has to be 784, because MNIST dataset stores 28 * 28 Pixel images, which are 784 Pixel 
'''
x = tf.placeholder('float',[None, len(train_x[0])],name="Input")
y = tf.placeholder('float',[None, n_classes],name="Output")

def neural_network_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Setting time before training
        timeBeforeTraining = datetime.now()

        for epoch in range(n_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        # Setting time after Training and calculate the difference
        timeAfterTraining = datetime.now()
        timeDelta = timeAfterTraining - timeBeforeTraining

        print("Tuning completed!")
        print("It took the network this time to train:", timeDelta)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)

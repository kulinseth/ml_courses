'''
Tested with Python 3.4.1 and Tensorflow 1.3.0
'''
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time
import matplotlib.pyplot as plt

DATA_PATH = 'art_data/'
#DATA_FILE = DATA_PATH + 'art_data.pickle'
DATA_FILE = DATA_PATH + 'augmented_art_data.pickle'
IMAGE_SIZE = 50
NUM_CHANNELS = 3
NUM_LABELS = 2
INCLUDE_TEST_SET = True

BATCH_SIZE = 10
NUM_TRAINING_STEPS = 10001
LEARNING_RATE = 0.01

L2_CONST = 0.01  # Set to > 0 to use L2 regularization
DROPOUT_RATE = 0.0  # Set to > 0 to use dropout
POOL1 = True  # Set to True to add pooling after first conv layer
POOL2 = True # Set to True to add pooling after second conv layer
POOL3 = False  # Set to True to add pooling after third conv layer
POOL4 = False  # Set to True to add pooling after fourth conv layer
POOL5 = True  # Set to True to add pooling after fifth conv layer

BN1 = True  # Set to True to use batch normalization after 1st conv layer
BN2 = True  # Set to True to use batch normalization after 2nd conv layer
BN3 = False  # Set to True to use batch normalization after 3rd conv layer
BN4 = False  # Set to True to use batch normalization after 4th conv layer
BN5 = False  # Set to True to use batch normalization after 5th conv layer

BN_fc1 = True  # Set to True to use batch normalization for FC_1 layer
BN_fc2 = True  # Set to True to use batch normalization for FC_2 layer
BN_fc3 = True  # Set to True to use batch normalization for FC_3 layer

class ArtistConvNet:
  def __init__(self, invariance=False):
    '''Initialize the class by loading the required datasets 
    and building the graph'''
    self.load_pickled_dataset(DATA_FILE)
    self.invariance = invariance
    if invariance:
      self.load_invariance_datasets()
    self.graph = tf.Graph()
    self.build_graph()


  def build_graph(self):
    with self.graph.as_default():
      # Input data
      self.images = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
      self.labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
      self.training = tf.placeholder(tf.bool)

      # Network
      regularizer = tf.contrib.layers.l2_regularizer(scale=L2_CONST)

      conv1 = tf.layers.conv2d(inputs=self.images, filters=16, kernel_size=10, strides=2, padding="same", activation=tf.nn.relu)
      if POOL1:
        conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
      if BN1: 
        conv1 = tf.layers.batch_normalization(inputs=conv1, axis=3, training=self.training)

      conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if POOL2:  
        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
      if BN2:
        conv2 = tf.layers.batch_normalization(inputs=conv2, axis=3, training=self.training)
        
      conv3 = tf.layers.conv2d(inputs=conv2, filters=16, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
      if POOL3:  
        conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=1)
      if BN3:
        conv3 = tf.layers.batch_normalization(inputs=conv3, axis=3, training=self.training)

      conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
      if POOL4:  
        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=1)
      if BN4:
        conv4 = tf.layers.batch_normalization(inputs=conv4, axis=3, training=self.training)
    
      conv5 = tf.layers.conv2d(inputs=conv4, filters=16, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
      if POOL5:  
        conv5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=2, strides=1)
      if BN5:
        conv5 = tf.layers.batch_normalization(inputs=conv5, axis=3, training=self.training)
        

      flat = tf.contrib.layers.flatten(inputs=conv5)      
      flat = tf.layers.dropout(inputs=flat, rate=DROPOUT_RATE, training=self.training)

      fc1 = tf.layers.dense(inputs=flat, units=64, activation=tf.nn.relu, kernel_regularizer=regularizer)
      if BN_fc1:
        fc1 = tf.layers.batch_normalization(inputs=fc1, axis=1, training=self.training)
      fc1 = tf.layers.dropout(inputs=fc1, rate=DROPOUT_RATE, training=self.training)

      fc2 = tf.layers.dense(inputs=fc1, units=32, activation=tf.nn.relu, kernel_regularizer=regularizer)
      if BN_fc2:
        fc2 = tf.layers.batch_normalization(inputs=fc2, axis=1, training=self.training)
      fc2 = tf.layers.dropout(inputs=fc2, rate=DROPOUT_RATE, training=self.training)
     
      fc3 = tf.layers.dense(inputs=fc2, units=16, activation=tf.nn.relu, kernel_regularizer=regularizer)
      if BN_fc3:
        fc3 = tf.layers.batch_normalization(inputs=fc3, axis=1, training=self.training)
      fc3 = tf.layers.dropout(inputs=fc3, rate=DROPOUT_RATE, training=self.training)
      
      logits = tf.layers.dense(inputs=fc3, units=NUM_LABELS, activation=None, kernel_regularizer=regularizer)
     
      # Compute loss
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
      self.loss += tf.losses.get_regularization_loss()
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)

      self.preds = tf.argmax(logits, 1)
      self.acc = 100*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), self.preds), dtype=tf.float32))


  def train_model(self, num_steps=NUM_TRAINING_STEPS):
    '''Train the model with minibatches in a tensorflow session'''
    with tf.Session(graph=self.graph) as session:
      session.run(tf.global_variables_initializer())
      print('Initializing variables...')
      ACCURACY_TRAIN = []
      ACCURACY_VAL = []
      ACCURACY_TEST = []
      STEP_PLOT = []
      ACCURACY_TRAIN_PLOT = []
      ACCURACY_VAL_PLOT = []
      ACCURACY_TEST_PLOT = []
      for step in range(num_steps):
        offset = (step * BATCH_SIZE) % (self.train_Y.shape[0] - BATCH_SIZE)
        batch_data = self.train_X[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = self.train_Y[offset:(offset + BATCH_SIZE), :]
        if (step > num_steps - 100):
          val_acc = session.run(self.acc, feed_dict={self.images: self.val_X, self.labels: self.val_Y, self.training: False})
          test_acc = session.run(self.acc, feed_dict={self.images: self.test_X, self.labels: self.test_Y, self.training: False})
          train_acc = session.run(self.acc, feed_dict={self.images: self.train_X, self.labels: self.train_Y, self.training: False})
          ACCURACY_TRAIN.append(train_acc)
          ACCURACY_VAL.append(val_acc)
          ACCURACY_TEST.append(test_acc)
        # Data to feed into the placeholder variables in the tensorflow graph
        feed_dict = {self.images: batch_data, self.labels: batch_labels, self.training: True}
        _, l, acc = session.run([self.optimizer, self.loss, self.acc], feed_dict=feed_dict)
        if (step % 100 == 0):
          val_acc = session.run(self.acc, feed_dict={self.images: self.val_X, self.labels: self.val_Y, self.training: False})
          test_acc = session.run(self.acc, feed_dict={self.images: self.test_X, self.labels: self.test_Y, self.training: False})
          train_acc = session.run(self.acc, feed_dict={self.images: self.train_X, self.labels: self.train_Y, self.training: False})
          STEP_PLOT.append(step)
          ACCURACY_TRAIN_PLOT.append(train_acc)
          ACCURACY_VAL_PLOT.append(val_acc)
          ACCURACY_TEST_PLOT.append(test_acc)
          print('')
          print('Batch loss at step %d: %f' % (step, l))
          print('Batch training accuracy: %.1f%%' % acc)
          print('Full training accuracy: %.1f%%' % train_acc)
          print('Validation accuracy: %.1f%%' % val_acc)
          print('Test accuracy: %.1f%%' % test_acc)
      plt.plot(STEP_PLOT, ACCURACY_TRAIN_PLOT, label = 'train')
      plt.plot(STEP_PLOT, ACCURACY_VAL_PLOT, label = 'validation')
      plt.plot(STEP_PLOT, ACCURACY_TEST_PLOT, label = 'test')
      plt.legend()
      plt.xlabel('steps')
      plt.ylabel('accuracy')
      plt.title('Accuracy = f(steps), training, validation, testing')
      plt.savefig('/Users/Alan/Desktop/mit_6.867/hw3/pb2_3_earlystopping.jpg')
      print('Final training accuracy: %.1f+-%.1f%%' % (np.mean(ACCURACY_TRAIN), np.std(ACCURACY_TRAIN)))
      print('Final Validation accuracy: %.1f+-%.1f%%' % (np.mean(ACCURACY_VAL), np.std(ACCURACY_VAL)))
      print('Final Test accuracy: %.1f+-%.1f%%' % (np.mean(ACCURACY_TEST), np.std(ACCURACY_TEST)))
      # This code is for the final question
      if self.invariance:
        print("\nObtaining final results on invariance sets!")
        sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X, 
            self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X, 
            self.inverted_val_X,]
        set_names = ['normal validation', 'translated', 'brightened', 'darkened', 
               'high contrast', 'low contrast', 'flipped', 'inverted']
        
        for i in range(len(sets)):
          acc = session.run(self.acc, feed_dict={self.images: sets[i], self.labels: self.val_Y, self.training: False})
          print('Accuracy on', set_names[i], 'data: %.1f%%' % acc)


  def load_pickled_dataset(self, pickle_file):
    print("Loading datasets...")
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      self.train_X = save['train_data']
      self.train_Y = save['train_labels']
      self.val_X = save['val_data']
      self.val_Y = save['val_labels']

      if INCLUDE_TEST_SET:
        self.test_X = save['test_data']
        self.test_Y = save['test_labels']
      del save  # hint to help gc free up memory
    print('Training set', self.train_X.shape, self.train_Y.shape)
    print('Validation set', self.val_X.shape, self.val_Y.shape)
    if INCLUDE_TEST_SET:
      print('Test set', self.test_X.shape, self.test_Y.shape)


  def load_invariance_datasets(self):
    with open(DATA_PATH + 'invariance_art_data.pickle', 'rb') as f:
      save = pickle.load(f)
      self.translated_val_X = save['translated_val_data']
      self.flipped_val_X = save['flipped_val_data']
      self.inverted_val_X = save['inverted_val_data']
      self.bright_val_X = save['bright_val_data']
      self.dark_val_X = save['dark_val_data']
      self.high_contrast_val_X = save['high_contrast_val_data']
      self.low_contrast_val_X = save['low_contrast_val_data']
      del save  


if __name__ == '__main__':
  invariance = True
  if len(sys.argv) > 1 and sys.argv[1] == 'invariance':
    print("Testing finished model on invariance datasets!")
    invariance = True
  
  t1 = time.time()
  conv_net = ArtistConvNet(invariance=invariance)
  conv_net.train_model()
  t2 = time.time()
  print("Finished training. Total time taken:", t2-t1)


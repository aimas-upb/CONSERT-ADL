import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TimeDistributed, RepeatVector, GRU
import os
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['AUTOGRAPH_VERBOSITY'] = '10'
tf.autograph.set_verbosity(0)

class Act2Vec(tf.keras.Model):
    """
    Implementation of activty2vec model as described in the paper<https://arxiv.org/abs/1907.05597>.
    """
    def __init__(self, units, input_dim):
        super(Act2Vec, self).__init__()
        self.layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units),input_shape=(input_dim[1:]), name='layer1')
        self.layer2 = tf.keras.layers.RepeatVector(input_dim[1], name='layer2')
        self.layer3 = tf.keras.layers.GRU(units, return_sequences=True, name='layer3')
        self.layer4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim[2]), name='layer4')

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
    
    def encoder(self, inputs):
        return self.layer1(inputs)

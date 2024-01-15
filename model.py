import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math
import keras

class TemporalConvNetBranch(tf.keras.Model):
    def __init__(self, k, df, d):
        super(TemporalConvNetBranch, self).__init__()
        # Initialization
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01,seed=42)
        # Normalization layer 1
        self.layer1 = layers.LayerNormalization()
        # ReLU Activation Layer 1
        self.ac1 = layers.Activation('relu')
        # Conv unit 1
        self.conv1 = layers.Conv1D(filters=df, kernel_size=1, dilation_rate=1, kernel_initializer=init)
        # Normalization layer 2
        self.layer2 = layers.LayerNormalization()
        # ReLU Activation Layer 2
        self.ac2 = layers.Activation('relu')
        # Conv unit 2
        self.conv2 = layers.Conv1D(filters=df, kernel_size=k, dilation_rate=d, kernel_initializer=init,padding="same")

    def call(self, x, training):
        x = self.layer1(x)
        x = self.ac1(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.ac2(x)
        x = self.conv2(x)
        return x

class MBTemporalConvNetBlock(tf.keras.Model):
    def __init__(self, k, df, d, dmodel):
        super(MBTemporalConvNetBlock, self).__init__()
        # Initializing
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01,seed=42)
        # Branches
        self.branch1 = TemporalConvNetBranch(k, df, d)
        self.branch2 = TemporalConvNetBranch(k, df, d)
        self.branch3 = TemporalConvNetBranch(k, df, d)
        self.branch4 = TemporalConvNetBranch(k, df, d)
        self.branch5 = TemporalConvNetBranch(k, df, d)
        self.branch6 = TemporalConvNetBranch(k, df, d)
        self.branch7 = TemporalConvNetBranch(k, df, d)
        self.branch8 = TemporalConvNetBranch(k, df, d)
        # Concatenate layer
        self.cc = layers.Concatenate()
        # Batch normalization
        self.layer = layers.LayerNormalization()
        # Activation relu
        self.ac = layers.Activation('relu')
        # Conv unit
        self.conv = layers.Conv1D(filters=dmodel, kernel_size=1, dilation_rate=1, kernel_initializer=init)
    def call(self, x, training):
        prev_x = x
        x1 = self.branch1(x, training=training)
        x2 = self.branch2(x, training=training)
        x3 = self.branch3(x, training=training)
        x4 = self.branch4(x, training=training)
        x5 = self.branch5(x, training=training)
        x6 = self.branch6(x, training=training)
        x7 = self.branch7(x, training=training)
        x8 = self.branch8(x, training=training)
        out = self.cc([x1, x2, x3, x4, x5, x6, x7, x8])
        out = self.layer(out)
        out = self.ac(out)
        out=self.conv(out)
        return prev_x+out

class MBTemporalConvNet(tf.keras.Model):
    def __init__(self, N, k, df, D, dmodel):
        super(MBTemporalConvNet, self).__init__()
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01,seed=42)
        model = tf.keras.Sequential()
        model.add(layers.Dense(dmodel, kernel_initializer=init))
        for i in range(1, N + 1):
            d = int(2**((i - 1) % (math.log2(D) + 1)))
            model.add(MBTemporalConvNetBlock(k, df, d, dmodel))
        model.add(layers.Dense(dmodel, activation="sigmoid", kernel_initializer=init))
        self.network = model

    def call(self, x, training):
        return self.network(x, training=training)

# Test
x = tf.convert_to_tensor(np.random.random((1, 10,10)))
model = MBTemporalConvNet(1, 3, 64, 16, 256)
y=model(x)
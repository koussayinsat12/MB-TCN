from model import MBTemporalConvNet
import yaml
import tensorflow as tf
import numpy as np

config_path='config.yaml'
with open(config_path,'r') as yaml_file:
    config=yaml.safe_load(yaml_file)
x = tf.convert_to_tensor(np.random.random((1, 10,10)))
model = MBTemporalConvNet(config["N"],config['k'],config['df'],config["D"],config["dmodel"])
y=model(x)
print(y.shape)
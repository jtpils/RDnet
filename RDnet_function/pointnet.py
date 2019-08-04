import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from transform_nets import input_transform_net, feature_transform_net
import numpy as np

def cls(input_tensor = None,num_cls = 40, C= None,):

    num_point = C.num_point
    batch_size = C.batch_size
    is_training = C.is_training
    end_points = {}

    input_shape = (1024,3)
    if input_tensor is None:
        input_pointcloud = layers.Input(shape = input_shape)
    else:
        input_pointcloud = input_tensor

    transform = input_transform_net(input_pointcloud,C,block = 'pointnet_cls',stage = 1,is_training = is_training)
    point_cloud_transformed = tf.matmul(input_pointcloud,transform)

    input_image = tf.expand_dims(point_cloud_transformed,-1)

    x = layers.Conv2D(64,(1,3),strides = (1,1),trainable = is_training,name = 'conv1')(input_image)
    x = layers.BatchNormalization(name = 'conv1_bn')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (1, 1), strides=(1, 1), trainable=is_training, name='conv2')(x)
    x = layers.BatchNormalization(name='conv2_bn')(x)
    x = layers.Activation('relu')(x)

    transform = feature_transform_net(x,C,block = 'pointnet_cls',stage = 1,is_training = is_training)
    end_points['transform'] = transform

    x_transformed = tf.matmul(tf.squeeze(x,axis = -2),transform)
    x_transformed = tf.expand_dims(x_transformed,[2])

    x = layers.Conv2D(64, (1, 1), strides=(1, 1), trainable=is_training, name='conv3')(x_transformed)
    x = layers.BatchNormalization(name='conv3_bn')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, (1, 1), strides=(1, 1), trainable=is_training, name='conv4')(x)
    x = layers.BatchNormalization(name='conv4_bn')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(1024, (1, 1), strides=(1, 1), trainable=is_training, name='conv5')(x)
    x = layers.BatchNormalization(name='conv5_bn')(x)
    x = layers.Activation('relu')(x)

    # Symmetric function from PointNet: max pooling
    x = layers.MaxPool2D((num_point,1),name = 'maxpool1')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu', trainable=is_training, name='fcl1')(x)
    x = layers.BatchNormalization(name='fcl1_bn')(x)
    x = layers.Dropout(rate = 0.7, name ='dp1')(x)

    x = layers.Dense(256, activation='relu', trainable=is_training, name='fcl2')(x)
    x = layers.BatchNormalization(name='fcl2_bn')(x)
    x = layers.Dropout(rate=0.6, name = 'dp2')(x)

    output_cls = layers.Dense(num_cls, activation=None, trainable=is_training, name='fcl3')(x)

    return output_cls, end_points

if __name__ == '__main__':

    import config

    C = config.Config()
    input_shape = (1024, 2)

    inputt = layers.Input(shape=input_shape)

    output_cls,_= cls(input_tensor=inputt,num_cls= 3, C=C)

    model = tf.keras.Model(inputs= inputt,outputs = output_cls)
    model.summary()

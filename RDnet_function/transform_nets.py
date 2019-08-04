import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

class Linear_lambda(layers.Layer):

    def __init__(self,units=32, input_dim=256,):
        super(Linear_lambda, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='zeros',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, C=None):
        b_c = tf.add(self.b,C)
        return tf.matmul(inputs, self.w) + b_c

def input_transform_net(input_tensor,C,block,stage,is_training, k =3):
    """
    input (x,y,z) transform net for BxNx3 point_cloud
    :param input_tensor: (BxNx3)
    :param is_training:
    :param bn_decay:
    :param k:
    :return: Trainsformation matrix of size 3xk
    """
    num_point = C.num_point
    batch_size = C.batch_size

    conv_name_base = 'input_conv_'  + block +'_'+ str(stage) + '_branch'
    bn_name_base = 'input_bn_' + block +'_'+ str(stage) + '_branch'
    fc_name_base = 'input_fc_' + block +'_'+ str(stage) + '_branch'

    input_tensor = tf.expand_dims(input_tensor,-1)
    x = layers.Conv2D(64,kernel_size = (1,input_tensor.shape[-2]),padding = 'valid',strides =(1,1),name = conv_name_base+'_1',trainable = is_training, data_format = 'channels_last' )(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base+'_1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128,kernel_size = (1,1),padding = 'valid',strides = (1,1),name = conv_name_base+'_2',trainable = is_training)(x)
    x = layers.BatchNormalization(name = bn_name_base+'_2')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(1024,kernel_size = (1,1),padding = 'valid',strides = (1,1),name = conv_name_base+'_3',trainable = is_training)(x)
    x = layers.BatchNormalization(name = bn_name_base+'_3')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size = [num_point,1], padding = 'valid',name = 'input_maxpool' + block + str(stage) )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation = 'relu',trainable = is_training, name = fc_name_base + '_1')(x)
    x = layers.BatchNormalization(name=bn_name_base + '_fc_1')(x)

    x = layers.Dense(256, activation = 'relu',trainable = is_training, name = fc_name_base + '_2')(x)
    x = layers.BatchNormalization(name=bn_name_base + '_fc_2')(x)


    assert (k==3)
    if input_tensor.shape[-2] == 3:
        plus_biases = K.constant([1, 0, 0, 0, 1, 0, 0, 0, 1])
        transform = Linear_lambda(input_tensor.shape[-2] * k, 256)(x, C=plus_biases)
        transform = tf.reshape(transform, [-1, input_tensor.shape[-2], k])
    else:
        x = layers.Dense(9, activation='relu', trainable=is_training, name=fc_name_base + '_-1')(x)
        x = layers.BatchNormalization(name=bn_name_base + '_fc_-1')(x)
        transform = tf.reshape(x,[-1, input_tensor.shape[-2], k])


    return transform

def feature_transform_net(input_tensor,C,block,stage,is_training,k=64):
    '''
    Feature Transform net!
    :param input_tensor: BxNx1xK
    :param num_point: int
    :param batch_size: int
    :param is_training: bool
    :param K: int
    :return: KxK transformat matrix
    '''

    num_point = C.num_point
    batch_size = C.batch_size

    conv_name_base = 'feature_conv_'  + block +'_'+ str(stage) + '_branch'
    bn_name_base = 'feature_bn_' + block +'_'+ str(stage) + '_branch'
    fc_name_base = 'feature_fc_' + block +'_'+ str(stage) + '_branch'

    x = layers.Conv2D(64,(1,1),padding = 'valid',trainable = is_training,name = conv_name_base + '_1')(input_tensor)
    x = layers.BatchNormalization(name = bn_name_base + '_1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, (1, 1), padding='valid', trainable=is_training, name=conv_name_base + '_2')(x)
    x = layers.BatchNormalization(name=bn_name_base + '_2')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(1024, (1, 1), padding='valid', trainable=is_training, name=conv_name_base + '_3')(x)
    x = layers.BatchNormalization(name=bn_name_base + '_3')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D((num_point,1),padding = 'valid',name = 'feature_maxpool'+ block + str(stage))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512,activation = 'relu',trainable = is_training,name = fc_name_base + '_1')(x)
    x = layers.BatchNormalization(name = bn_name_base + '_fc_1')(x)
    x = layers.Dense(256, activation = 'relu',trainable=is_training, name= fc_name_base + '_2')(x)
    x = layers.BatchNormalization(name=bn_name_base + '_fc_2')(x)

    plus_biases = tf.constant(np.eye(int(k)).flatten(),dtype = tf.float32)
    transform = Linear_lambda(k * k, 256)(x, C = plus_biases)
    transform = tf.reshape(transform,[-1,k,k])


    return transform


if __name__ == '__main__':


    point_cloud = np.zeros([32,1024,3])
    is_training = True
    num_point = point_cloud.shape[1]
    batch_size = 32
    from config import Config
    C = Config()

    input_image = tf.keras.Input(shape=(num_point, 2))
    transform = input_transform_net(input_image,C,block = 'test',stage = 1, is_training = is_training)
    model_input = tf.keras.Model(inputs=input_image, outputs=transform,name = 'model_input')
    model_input.summary()

    input_feature = tf.keras.Input(shape = (1024,1,64))
    transform_feature = feature_transform_net(input_feature, C,block = 'test',stage = 1, is_training = is_training)
    model_feature = tf.keras.Model(inputs = input_feature,outputs=transform_feature,name = 'model_feature')
    model_feature.summary()

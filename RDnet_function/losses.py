from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np



def pointnet_cls_loss(end_points,reg_weight = 0.001):

    def kernel(y_true,y_pred):
        """

        :param y_true: shape = （[batch_size，num_classes]）  one-hot
        :param y_pred: shape = （[batch_size，num_classes]） one-hot
        :return:
        """

        loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred,labels = y_true)
        classify_loss = tf.reduce_mean(loss)

        #Enforce the transformation as orthogonal matrix
        transform = end_points['transform'] #B*K*K
        # transform = end_points
        eyes = transform.shape[1]
        mat_diff = tf.matmul(transform,tf.transpose(transform,perm=[0,2,1]))
        mat_diff -= tf.constant(np.eye(eyes),dtype = tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)

        return classify_loss + mat_diff_loss * reg_weight
    return kernel

if __name__ == '__main__':
    end_points = tf.constant(np.zeros(shape =(32, 64, 64)),
                                      dtype = tf.float32,
                                      name = 'transform')
    y_true = tf.constant(np.array([[0, 1, 0], [0, 1, 0]]),dtype = tf.float32)
    y_pred = tf.constant(np.array([[0, 0.8, 0], [0, 0.3, 0.8]]),dtype = tf.float32)

    losses =pointnet_cls_loss(end_points)(y_true ,
                                          y_pred )
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
import cook, losses


def get_data_cls(C):
    f = h5py.File(C.path, 'r')
    list_classes = list(f['classes'])
    all_pointcloud = np.zeros(shape=(1, C.num_point, 3))
    all_labels = np.zeros(shape=(1, 3))
    map = {}
    for i, name in enumerate(list_classes):
        strr = 'classes/' + name
        data = f[strr]
        batches = data.shape[0] // C.num_point
        data = data[:batches * C.num_point, ...].reshape((-1, C.num_point, 3))

        label = np.zeros(shape=(1, len(list_classes)))
        label[0, i] = 1
        label = np.zeros(shape=(data.shape[0], 3)) + label

        all_pointcloud = np.vstack((all_pointcloud, data))
        all_labels = np.vstack((all_labels, label))
        map[name] = i

    all_pointcloud = all_pointcloud[1:, ...]
    all_labels = all_labels[1:, ...]

    num_class = len(list_classes)
    f.close()


    data, all_labels = cook.rotate_pointcloud(all_pointcloud, all_labels, C)
    data, idx = cook.shuffle_data(data)
    all_labels = all_labels[idx, ...]

    batches = len(data)
    train_clip, val_clip, test_clip = np.round(batches * C.allocation).tolist()

    train_clip = int(train_clip)
    val_clip = int(val_clip)

    train_data = data[:train_clip, ...]
    train_label = all_labels[:train_clip, ...]

    val_data = data[train_clip:train_clip + val_clip, ...]
    val_label = all_labels[train_clip:train_clip + val_clip, ...]

    test_data = data[train_clip + val_clip:, ...]
    test_label = all_labels[train_clip + val_clip:, ...]


    r_train,r_label = cook.rotate_block(train_data,train_label)
    rd_train = cook.density_block(r_train,C)

    return [rd_train[:,3:], r_label], [val_data, val_label], [test_data, test_label], num_class, map


def get_data_seg(C):
    f = h5py.File(C.path, 'r')
    all_data = f['data'][:]
    all_labels = f['label'][:]
    num_class = f['num_classes'][()]
    map = list(f['classes'])
    f.close()
    ex_batches = len(all_data) // C.num_point
    all_data = np.concatenate((all_data, all_labels), axis=1)
    all_data = all_data[:ex_batches * C.num_point, :].reshape((ex_batches, C.num_point, -1))

    data = cook.rotate_data(all_data, C)
    data, _ = cook.shuffle_data(data)

    batches = len(data)
    train_clip, val_clip, test_clip = np.round(batches * C.allocation).tolist()

    train_clip = int(train_clip)
    val_clip = int(val_clip)

    train_data = data[:train_clip, ..., :3]
    train_label = data[:train_clip, ..., 3:]

    val_data = data[train_clip:train_clip + val_clip, ..., :3]
    val_label = data[train_clip:train_clip + val_clip, ..., 3:]

    test_data = data[train_clip + val_clip:, ..., :3]
    test_label = data[train_clip + val_clip:, ..., 3:]

    r_train,r_label = cook.rotate_block(train_data,train_label)
    rd_train = cook.density_block(r_train,C)

    return [rd_train, r_label], [val_data, val_label], [test_data, test_label], num_class, map


def get_model(C, num_classes):
    if C.network == 'pointnet':
        import pointnet as nn

    input_shape = (C.num_point, 2)

    tensor_input = layers.Input(shape=input_shape)
    pred, end_points = nn.cls(input_tensor=tensor_input, num_cls=num_classes, C=C)

    model = tf.keras.Model(inputs=tensor_input, outputs=pred)

    optimizer = tf.keras.optimizers.Adam(learning_rate=C.lr,
                                         beta_1=C.beta_1, beta_2=C.beta_2,
                                         amsgrad=C.amsgrad)

    model.compile(optimizer=optimizer,
                  loss=losses.pointnet_cls_loss(end_points),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

if __name__ == '__main__':

    import config

    C = config.Config()


    C.network = 'pointnet'
    # cls_train, cls_val, cls_test, num_classes, classes_map = get_data_cls(C)
    num_classes = 3
    model = get_model(C,num_classes)
    model.summary()

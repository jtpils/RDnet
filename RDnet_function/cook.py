import os
import sys
import numpy as np
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'Dataset'))

def shuffle_data(data):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx,...],idx

def rotate_data(data,C):
    if C.rotate_angle == []:
        angles = [np.random.uniform() * 2 * np.pi for i in range(C.rotate_times)]
    else:
        angles = C.rotate_angle

    rotated_datas = np.zeros(shape=(1, data.shape[1], data.shape[2]))
    for i in range(C.rotate_times):

        angle = angles[i]
        rotate_maxtrix = [np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]]),

                          np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]]),

                          np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])]


        if C.rotate_mode == 0:
            return data
        elif C.rotate_mode == 4:

            rotated_pc = np.dot(data[:,:,:3],rotate_maxtrix[np.random.randint(0,3)])
            rotated_data = np.dstack((rotated_pc, data[:, :, 3:]))
            rotated_datas = np.concatenate((rotated_datas,rotated_data),axis = 0)
        else:
            rotated_pc = np.dot(data[:, :, :3], rotate_maxtrix[C.rotate_mode-1])
            rotated_data = np.dstack((rotated_pc, data[:, :, 3:]))
            rotated_datas=np.concatenate((rotated_datas,rotated_data),axis = 0)

    all_data = np.concatenate((data,rotated_datas[1:,...]),axis = 0)

    return all_data

def rotate_pointcloud(data,label,C):
    if C.rotate_angle == []:
        angles = [np.random.uniform() * 2 * np.pi for i in range(C.rotate_times)]
    else:
        angles = C.rotate_angle

    rotated_datas = np.zeros(shape=(1, data.shape[1], data.shape[2]))
    labels = np.zeros(shape = (1,3))
    for i in range(C.rotate_times):

        angle = angles[i]
        rotate_maxtrix = [np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]]),

                          np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]]),

                          np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])]


        if C.rotate_mode == 0:
            return data
        elif C.rotate_mode == 4:

            rotated_pc = np.dot(data[:,:,:3],rotate_maxtrix[np.random.randint(0,3)])
            rotated_datas = np.concatenate((rotated_datas,rotated_pc),axis = 0)
            labels = np.concatenate((labels,label),axis = 0)
        else:
            rotated_pc = np.dot(data[:, :, :3], rotate_maxtrix[C.rotate_mode-1])
            rotated_datas=np.concatenate((rotated_datas,rotated_pc),axis = 0)
            labels = np.concatenate((labels, label), axis=0)

    return rotated_datas[1:,...],labels[1:,...]

def density_block(data,C):
    '''
    input: (B,N,3)
    return density (*,5)
    density = [num_in_radius , mean_distance_in_radius]
    '''

    d_tensor =np.zeros((data.shape[0],data.shape[1],5))
    for i in range(len(data)):
        tree = KDTree(data[i],leaf_size= (data[i].shape[0])//2+10)
        ind,dis = tree.query_radius(data[i],r=C.radius,return_distance = True)
        den = np.array([[len(ind[i]),dis[i].mean()] for i in range(len(ind))])

        d_tensor[i] = np.hstack((data[i],den))
    d_data = d_tensor.reshape(-1,5)
    return d_data

def rotate_block(data,label):
    '''
    input: Block  (*,3) tensor ,
    return a (B,N,3) rotated block tensor
    '''
    B = data.shape[0]//C.num_point
    data = data [:B*C.num_point,...]
    label = label[:B*C.num_point,...]

    data = data.reshape(B, C.num_point, 3)
    r_data = np.zeros(data.shape)
    for i in range(B):
        angle = np.random.uniform() * 2 * np.pi
        rotate_maxtrix = [np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])]
        r_data[i]= np.dot(data[i], rotate_maxtrix).reshape(-1,3)

    return r_data,label

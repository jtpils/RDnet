from tensorflow.keras import backend as K
import numpy as np

class Config:

    def __init__(self):

        self.path = '../Dataset/classes_spare_all_data.h5'


        self.max_iteration = 1000
        self.num_point = 1024
        self.batch_size = 32
        self.allocation = np.array([0.8, 0.1, 0.1])
        self.is_training = True
        self.verbose = 2

        self.rotate_mode = 3  # 5 models:{'None': 0 ,'roll': 1,'pitch':2,'yaw':3 ,'raondom':4}
        self.rotate_times = 5
        self.rotate_angle = []


        self.lr = 1e-5
        self.beta_1 = 0.999
        self.beta_2 = 0.999999
        self.amsgrad = True

        self.radius = 3 # m



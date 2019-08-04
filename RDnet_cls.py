import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取上级目录（当前文件的文件夹目录）
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'RDnet_function'))


import config



print(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',type = int , default = 0, help = 'use GPU or not[default : 0]')
parser.add_argument('--log_dir', default = 'log', help = 'Log dir[default: log]')
parser.add_argument('-p','--path',default = '/Users/hova/PycharmProjects/yhx_rotate_density_net/yhx_rotate_density_net/Dataset/classes_spare_all_data.h5', help = 'path to training data [default: Dateset]')
parser.add_argument('-m','--model',  default  = 'model_A',help = 'model options: model_A, model_B, model_AB [default : model_A+B]')
parser.add_argument('-n','--network',  default  = 'pointnet',help = 'choose a  base network for training [default : pointnet]')

FLAGS = parser.parse_args()
C = config.Config()
C.log_dir = FLAGS.log_dir
C.path = FLAGS.path
C.network = FLAGS.network


if FLAGS.model == 'model_A':
    import model_A as model
    C.model = 'model_A'

if FLAGS.model == 'model_B':
    import model_B as model
    C.model = 'model_B'

if FLAGS.model == 'model_AB':
    import model_AB as model
    C.model = 'model_AB'

MODEL_FILE = os.path.join(BASE_DIR,'RDnet_function',C.network+'.py')
LOG_DIR = C.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE,LOG_DIR))  #bkp of model def
os.system('cp RDnet_cls.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR,'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

#load data
cls_train,cls_val,cls_test,num_classes,classes_map=model.get_data_cls(C)

print('Num train samples {}'.format(len(cls_train[1])))
print('Num val samples {}'.format(len(cls_val[1])))
print('Num test samples {}'.format(len(cls_test[1])))
print('Starting building {} by useing {}...'.format(C.model,C.network))

#build model & compile it
model = model.get_model(C,num_classes)
max_iteration = C.max_iteration

print('Starting training')
for i in range(max_iteration):
    history = model.fit(x = cls_train[0][:1000,...],y = cls_train[1][:1000,...],
                                     batch_size = C.batch_size,
                                     epochs = int(len(cls_train[0])//C.batch_size),
                                     validation_data=(cls_val[0],cls_val[1]))

    total_accuracy = float(sum(history.history['sparse_categorical_accuracy'])/len(history.history['sparse_categorical_accuracy']))
    print('loss_{} is {}'.format(i,history.history['loss']))
    print('total_accuracy',total_accuracy)

    log_string('history dict',history.history)
    log_string('total accyracy over train',total_accuracy)
print('Evaluate on test data')
results = model.evaluate(cls_test[0],cls_test[1],batch_size=128)
print('test loss, test acc:', results)


LOG_FOUT.close()


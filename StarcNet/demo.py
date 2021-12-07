import os
import sys
import csv
import time
import pickle
import numpy as np

start_time = time.time()

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, './src/utils')
sys.path.insert(0, './model')

from data_utils import load_db
from starcnet import Net
import matplotlib.pyplot as plt

from demo_lib import *

print(open('targets.txt', 'r').read())
print(open('frc_fits_links.txt', 'r').read())

print('creating dataset...')
os.system('bash create_dataset.sh')
print('dataset created')

batch_size = 64 # input batch size for testing (default: 64)
data_dir = 'data/' # dataset directory
dataset = 'raw_32x32' # dataset file reference
checkpoint = 'model/starcnet.pth' # trained model
gpu = '' # CUDA visible device (when using a GPU add GPU id (e.g. '0'))
cuda = False # enables CUDA training (when using a GPU change to True)

data_all, _, ids = load_db(os.path.join(data_dir,'test_'+dataset+'.dat'))
mean = np.load(os.path.join(data_dir,'mean.npy'))

data_test = data_all - mean[np.newaxis,:,np.newaxis,np.newaxis] # subtract mean

data = torch.from_numpy(data_test).float()
test_loader = DataLoader(TensorDataset(data), batch_size=batch_size,
                         shuffle=False)

cuda = cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

model = Net() # create model with StarcNet architecture
load_weights(model, cuda, checkpoint) # load trained model parameters

if cuda:
    model.cuda()

# classify all 32x32x5 arrays
predictions, scores = test(test_loader, cuda, model) 

# save scores to 'output/scores.npy' file
np.save(os.path.join('output','scores'), scores) 

print('End of classification | time: %.2fs'%(time.time() - start_time))
print('Objects classified as Class 1: %d'%(len(np.where(predictions == 0)[0])))
print('Objects classified as Class 2: %d'%(len(np.where(predictions == 1)[0])))
print('Objects classified as Class 3: %d'%(len(np.where(predictions == 2)[0])))
print('Objects classified as Class 4: %d'%(len(np.where(predictions == 3)[0])))

p = np.zeros(predictions.shape)
p[predictions>1] = 2
p[predictions<=1] = 1

predictions = p

print('cluster vs noncluster')
print('Objects classified as Class 1: %d'%(len(np.where(predictions == 1)[0])))
print('Objects classified as Class 2: %d'%(len(np.where(predictions == 2)[0])))

from scipy.io import savemat
mdic = {"predictions"   : predictions, 
        "images"        : data.numpy()}
savemat("star_data.mat", mdic)
exit()




print('Optimal Transportation:')
data_subsample = np.random.choice(predictions.shape[0],size=100,replace=False)
data_np = data.numpy()
lambda_v = 10.**np.arange(-5,1,1)
for lambda_ind in range(lambda_v.shape[0]):
    print('lambda: %2.3f'%lambda_v[lambda_ind])
    predictions_OT = {}
    for i in data_subsample:
        print('data_subsample i: %d'%i)
        start_time = time.time()
#        if predictions[i]==1 or predictions[i]==0:
#            for channel in range(0,5):
#                plt.matshow(data_np[i,channel,:,:])
#                plt.title('data point: %d, channel: %d, class: %d'%(i, 
#                             channel, predictions[i]+1))

#                print('channel: %d'%channel)
        prediction_OT, score_OT = test_OT(test_loader, 
                                predictions.shape, scores.shape, 
                                np.array([lambda_v[lambda_ind]]), 
                                predictions, [i], data_np)
#                print('')
        predictions_OT[i] = prediction_OT[i]

        print('time: %.2fs'%(time.time() - start_time))

    prediction_OT = 0*prediction_OT+2
    for k,v in predictions_OT.items():
        prediction_OT[k] = v
    print('Objects classified as Class 1: %d'%(len(np.where(prediction_OT == 1)[0])))
    print('Objects classified as Class 2: %d'%(len(np.where(prediction_OT == 2)[0])))

    predictions_agree = np.where(predictions == prediction_OT)
    print('Class 1 accuracy: %f'%(len(np.where(predictions[predictions_agree] == 1)[0])/predictions.shape[0]))
    print('Class 2 accuracy: %f'%(len(np.where(predictions[predictions_agree] == 2)[0])/predictions.shape[0]))


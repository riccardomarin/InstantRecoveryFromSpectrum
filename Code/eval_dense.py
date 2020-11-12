import tensorflow as tf
import hdf5storage
import numpy as np
import scipy.io as sio
import os
from sklearn.neighbors import *

data = hdf5storage.loadmat('./data/coma_FEM.mat')
remeshed_200 = hdf5storage.loadmat('./data/remeshed_200.mat')
remeshed_500 = hdf5storage.loadmat('./data/remeshed_500.mat')
remeshed_1000 = hdf5storage.loadmat('./data/remeshed_1000.mat')

n = '_1999'

model = './models/pretrained-dense/'

evals = 30

e = data['noeye_evals_FEM3'][:,1:evals+1].astype('float32')       # Eigenvalues of the meshes
pix = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1], 3).astype('float32') # Vertices of the meshes

outliers = np.asarray([6710, 6792, 6980])-1;
test_subj = np.arange(18531,20466)-1;
remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000])-1;

idxs_for_train = [np.int(x) for x in np.arange(0,pix.shape[0],10) if (np.int(x) not in test_subj and np.int(x) not in outliers and np.int(x) not in remeshed)]
idxs_for_test = [x for x in np.arange(0,pix.shape[0]) if x not in idxs_for_train]

sub_set_test_subj = np.hstack((np.arange(0,100,1)));
real_t_subj = test_subj[sub_set_test_subj]

train_images = pix[idxs_for_train, :,:]
train_eigs = e[idxs_for_train]

test_images = pix[idxs_for_test, :, :]
test_eigs = e[idxs_for_test]

#### Nearest Neighbor Baseline
kdt = KDTree(train_eigs, metric='euclidean')
NN_test_subj  = np.squeeze(kdt.query(e[real_t_subj,:], return_distance=False))

err_NN_subj  = np.mean(np.sum((train_images[NN_test_subj,:,:]-pix[real_t_subj,:,:])**2,2))

e_200 = remeshed_200['all_evals_FEM3'][sub_set_test_subj,1:evals+1]
mesh_200 = pix[real_t_subj,:,:]
e_500 = remeshed_500['all_evals_FEM3'][sub_set_test_subj,1:evals+1]
mesh_500 = pix[real_t_subj,:,:]
e_1000 = remeshed_1000['all_evals_FEM3'][sub_set_test_subj,1:evals+1]
mesh_1000 = pix[real_t_subj,:,:]

NN_200  = np.squeeze(kdt.query(e_200, return_distance=False))
NN_500  = np.squeeze(kdt.query(e_500, return_distance=False))
NN_1000  = np.squeeze(kdt.query(e_1000, return_distance=False))

err_NN_subj_200   = np.mean(np.sum((train_images[NN_200,:,:] -mesh_200)**2,2))
err_NN_subj_500   = np.mean(np.sum((train_images[NN_500,:,:] -mesh_500)**2,2))
err_NN_subj_1000  = np.mean(np.sum((train_images[NN_1000,:,:]-mesh_1000)**2,2))

#####
# 
# TEST MODEL

decoder = tf.keras.models.load_model(model + 'decoder'+ n + '.h5')
N1 = tf.keras.models.load_model(model + 'N1'+ n + '.h5')
N2 = tf.keras.models.load_model(model + 'N2'+ n + '.h5')
subj = decoder(N1(e[real_t_subj])).numpy()
subj_200  = decoder(N1(e_200)).numpy()
subj_500  = decoder(N1(e_500)).numpy()
subj_1000 = decoder(N1(e_1000)).numpy()

err_my_subj_200 = np.mean(np.sum((subj_200 -mesh_200)**2,2))
err_my_subj_500 = np.mean(np.sum((subj_500 -mesh_500)**2,2))
err_my_subj_1000 = np.mean(np.sum((subj_1000-mesh_1000)**2,2))
err_my_subj =     np.mean(np.sum((subj         -pix[real_t_subj,:,:])**2,2))

print('         OUR              NN')
print('full: ' + '{:.2e}'.format(err_my_subj) + '          {:.2e}'.format(err_NN_subj) )
print('1000: ' + '{:.2e}'.format(err_my_subj_1000) + '          {:.2e}'.format(err_NN_subj_1000) )
print('500 : ' + '{:.2e}'.format(err_my_subj_500) + '          {:.2e}'.format(err_NN_subj_500) )
print('200 : ' + '{:.2e}'.format(err_my_subj_200) + '          {:.2e}'.format(err_NN_subj_200) )

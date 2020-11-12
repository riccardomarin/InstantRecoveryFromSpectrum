import tensorflow as tf
import hdf5storage
import numpy as np
from sklearn.neighbors import *
import matplotlib.pyplot as plt
import open3d as o3d

tf.config.experimental.set_visible_devices([], 'GPU')
# Load model
n=str(899)
encoder = tf.keras.models.load_model('./models/pretrained-pointnet/encoder_'+n+'.h5')
decoder = tf.keras.models.load_model('./models/pretrained-pointnet/decoder_'+n+'.h5')
N1 = tf.keras.models.load_model('./models/pretrained-pointnet/N1_'+n+'.h5')
N2 = tf.keras.models.load_model('./models/pretrained-pointnet/N2_'+n+'.h5')

def error(s_gt,s_o):
      return np.sum(np.abs(s_gt - s_o)/s_gt)

# Load Data
PI = hdf5storage.loadmat('./data/PI_coma.mat') # Load dataset
PI = PI['PI']-1
data = hdf5storage.loadmat('./data/coma_FEM.mat') # Load dataset
pix1 = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1], 3).astype('float32') # Vertices of the meshes
pix = pix1[:,PI]
outliers = np.asarray([6710, 6792, 6980])-1;
remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000])-1;
e = data['noeye_evals_FEM3'][:,1:30+1].astype('float32')       # Eigenvalues of the meshes
# e = np.asarray([e[i,:] for i in np.arange(0,e.shape[0]) if i not in outliers])

# Selecting training and test dataset
test_subj = np.arange(18531,20465)-1;
idxs_for_train = [np.int(x) for x in np.arange(0,pix.shape[0],10) if (np.int(x) not in test_subj and np.int(x) not in outliers and np.int(x) not in remeshed)]
idxs_for_test = [x for x in np.arange(0,pix.shape[0]) if x not in idxs_for_train and x not in outliers]

# Split in train and test
train_images = pix[idxs_for_train, :,:]
train_eigs = e[idxs_for_train]

# Load test case
dd = hdf5storage.loadmat('./data/spec_esteem.mat')

# Visualize mesh and pointcloud
T = o3d.geometry.TriangleMesh()
T.vertices = o3d.utility.Vector3dVector(dd['mesh'])
T.triangles = o3d.utility.Vector3iVector(dd['f']-1)
T.compute_vertex_normals()

PC = o3d.geometry.PointCloud()
PC.points = o3d.utility.Vector3dVector(dd['pc'])
o3d.visualization.draw_geometries([T, PC])

def error(s_gt,s_o):
      return np.sum(np.abs(s_gt - s_o)/s_gt)
def error_vec(s_gt,s_o):
      return np.sum(np.abs(s_gt - s_o)/s_gt,1)

def normalize(s):
      return s/s[:,0,None]



# Our
subj_evals = N2(encoder(np.reshape(dd['pc'],(1,-1,3)))).numpy()
err_our = error(dd['evals'], subj_evals)

# NN Baseline
kdt = KDTree(encoder(train_images).numpy(), metric='euclidean')
NN1 = np.squeeze(kdt.query(encoder(np.reshape(dd['pc'],(1,-1,3))).numpy(), return_distance=False))
NN1_evals =train_eigs[NN1]
err_NN1 = error(dd['evals'], NN1_evals)

# Plot cumulative error. Sum error in the legend.
plt.plot(np.cumsum(np.abs(dd['evals'] - subj_evals)/dd['evals']))
plt.plot(np.cumsum(np.abs(dd['evals'] - NN1_evals)/dd['evals']))
plt.legend(('Our: ' + "{:.2}".format(err_our),'NN: ' + "{:.2}".format(err_NN1)))
plt.title('Cumulative Error')
plt.show()


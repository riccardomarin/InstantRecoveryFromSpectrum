import tensorflow as tf
from open3d import *
import hdf5storage
import numpy as np

##### Init
tf.random.set_seed(1234) # Set random seed for deterministic experiments

# Parameters: 
params = {'N_units': 30,    #  Latent space dimension
        'L_rate': 0.0001,    #  Learning rate
        'B_size': 512,      #  Batch Size
        'Epochs': 1000       #  Number of traning epochs
        }

BUFFER_SIZE = 60000 # Buffer for batch shuffle

data = hdf5storage.loadmat('../coma_data_om.mat') # Load dataset
pix = data['meshes'] # Vertices of the meshes

train_images = pix.reshape(pix.shape[0], pix.shape[1], 3).astype('float32')

# AutoEncoder build
in_enc=tf.keras.layers.Input((pix.shape[1], 3), name='input')
x = tf.keras.layers.Reshape((pix.shape[1] * 3,), name='res_input')(in_enc)
x = tf.keras.layers.Dense(50 * 3, activation='tanh', name='hidden_en')(x)
out_enc=tf.keras.layers.Dense(np.int64(params['N_units']), activation='tanh', name='latent_space')(x)

in_dec=tf.keras.layers.Dense(50 * 3, activation='tanh', name='latent_space_2')(out_enc)
x=tf.keras.layers.Dense(pix.shape[1] * 3, activation='linear', name='hidden')(in_dec)
out_dec=tf.keras.layers.Reshape((pix.shape[1], 3), name='out_layer')(x)

AE=tf.keras.Model(inputs=in_enc,outputs=out_dec)

# Encoder and Decoder modules, as standalone networks
encoder = tf.keras.Sequential()
encoder.add(AE.layers[0])
encoder.add(AE.layers[1])
encoder.add(AE.layers[2])
encoder.add(AE.layers[3])

encoded_input = tf.keras.layers.Input(shape=(params['N_units'], ))
decoder_layer1 = AE.layers[-3]
decoder_layer2 = AE.layers[-2]
decoder_layer3 = AE.layers[-1]
decoder = tf.keras.Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

generator_optimizer = tf.keras.optimizers.Adam(params['L_rate']) # Optimizer
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(np.int64(params['B_size'])) # Training set batching

##### Training
for epoch in range(params['Epochs']):
    print(epoch)
    for meshes in train_dataset:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = AE(meshes, training=True)
            loss1 = tf.keras.losses.mean_squared_error(meshes, generated)
            gen_loss = loss1

        gradients_of_generator = gen_tape.gradient(gen_loss, AE.trainable_variables)
        temp = generator_optimizer.apply_gradients(zip(gradients_of_generator, AE.trainable_variables))

# Save encoder and decoder disjointly
encoder.save('inference_nt.h5')
decoder.save('generator_nt.h5')

##### Test Experiment
target = train_images[20000,:,:]
in1 = encoder((target).reshape(1,pix.shape[1]*pix.shape[2])).numpy()

recon = decoder(np.reshape(in1,(1,params['N_units']))).numpy().reshape((-1,3))

mesh2 = TriangleMesh()
mesh2.vertices = Vector3dVector( target)
mesh2.triangles = Vector3iVector(np.asarray(data['f'].astype(int))-1)
mesh2.compute_vertex_normals()

mesh3 = TriangleMesh()
mesh3.vertices = Vector3dVector(recon + [0.4, 0, 0])
mesh3.triangles = Vector3iVector(np.asarray(data['f'].astype(int))-1)
mesh3.compute_vertex_normals()
draw_geometries([mesh2,mesh3])
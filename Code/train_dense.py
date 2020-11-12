import tensorflow as tf
from tensorflow.keras import regularizers
import hdf5storage
import numpy as np
import os

##### Init
rand_seed = 1234
tf.random.set_seed(rand_seed) # Set random seed for deterministic experiments
tf.keras.backend.set_floatx('float32')
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

save_every = 400
BUFFER_SIZE = 60000 # Buffer for batch shuffle
#### Network Params
params = {'N_units': 30,            #  Latent space dimension
        'L_rate_AE': 1e-4,          #  Learning rate
        'B_size': 16,               #  Batch Size
        'Epochs': 2010,             #  Max epochs
        'N_evals': 30,              #  Number of eigenvalues
        'W_N1': 1e-4,               #  Weight of Pi loss
        'W_N2': 1e-4,               #  Weight of Rho loss
        'enc_layers': [300, 200],   #  Encoder Layers
        'dec_layers': [200],        #  Decoder Layers
        'AE_activation': 'tanh',    # AE Activation function
        'N1_layers': [80, 160, 320, 640, 320, 160, 80], # Pi Layers
        'N2_layers': [80, 160, 320, 640, 320, 160, 80], # Rho Layers
        'bat_n_all': True          # Batch normalization   
        }

##### Prepare data
data = hdf5storage.loadmat('./data/coma_FEM.mat') # Load dataset
pix = data['meshes_noeye'].reshape(data['meshes_noeye'].shape[0], data['meshes_noeye'].shape[1], 3).astype('float32') # Vertices of the meshes
e = data['noeye_evals_FEM3'][:,1:params['N_evals']+1].astype('float32')       # Eigenvalues of the meshes
outliers = np.asarray([6710, 6792, 6980])-1;
remeshed = np.asarray([820, 1200, 7190, 11700, 12500, 14270, 15000, 16300, 19180, 20000])-1;

# Split in train and test
test_subj = np.arange(18531,20465)-1;
idxs_for_train = [np.int(x) for x in np.arange(0,pix.shape[0],10) if (np.int(x) not in test_subj and np.int(x) not in outliers and np.int(x) not in remeshed)]
idxs_for_test = [x for x in np.arange(0,pix.shape[0]) if x not in idxs_for_train and x not in outliers]

model_name = 'final'
os.mkdir('./models/' + model_name)

#Split in train and test
train_images = pix[idxs_for_train, :,:]
train_eigs = e[idxs_for_train]

test_images = pix[idxs_for_test, :, :]
test_eigs = e[idxs_for_test]

#### BUILD MODELS
def build_AE():   
    # AutoEncoder
    in_enc=tf.keras.layers.Input((pix.shape[1], 3), name='input')
    x = tf.keras.layers.Reshape((pix.shape[1] * 3,), name='res_input')(in_enc)
    for n in np.arange(0, len(params['enc_layers'])):
        x = tf.keras.layers.Dense(params['enc_layers'][n], activation=params['AE_activation'], name='enc_' + str(n),  kernel_regularizer=regularizers.l2(0.01))(x)

    out_en=tf.keras.layers.Dense(np.int64(params['N_units']), activation=params['AE_activation'], name='latent_space',  kernel_regularizer=regularizers.l2(0.01))(x)
    for n in np.arange(0, len(params['dec_layers'])):
        if n == 0:
            x = tf.keras.layers.Dense(params['dec_layers'][n], activation=params['AE_activation'], name='dec_' + str(n),  kernel_regularizer=regularizers.l2(0.01))(out_en)
        else:
            x = tf.keras.layers.Dense(params['dec_layers'][n], activation=params['AE_activation'], name='dec_' + str(n),  kernel_regularizer=regularizers.l2(0.01))(x)

    x=tf.keras.layers.Dense(pix.shape[1] * 3, activation='linear', name='dec_hidden')(x)
    out_dec=tf.keras.layers.Reshape((pix.shape[1], 3), name='dec_out')(x)

    autoenc = tf.keras.Model(inputs=[in_enc], outputs=[out_dec])

    decoder = tf.keras.models.Sequential()
    encoded_input = tf.keras.layers.Input(shape=(params['N_units'], ))
    decoder.add(encoded_input)
    for i in np.arange(0,len(autoenc.layers)):
        if 'dec_' in autoenc.layers[i].name:
            decoder.add(autoenc.layers[i])
    
    #N1 and N2
    in_eig = tf.keras.layers.Input(shape=(params['N_evals'],), name='in_evals' )
    y = tf.keras.layers.BatchNormalization(name='e_to_v_norm')(in_eig)
    for n in np.arange(0, len(params['N1_layers'])):
        y = tf.keras.layers.Dense(params['N1_layers'][n], activation='selu', name='e_to_v_' + str(n),  kernel_regularizer=regularizers.l2(0.01))(y)
        if params['bat_n_all']:
            y = tf.keras.layers.BatchNormalization(name='e_to_v_' + str(n) +'_norm')(y)
    N1 = tf.keras.layers.Dense(params['N_units'], activation='linear', name = 'e_to_v_out')(y)

    y = tf.keras.layers.BatchNormalization(name='v_to_e_norm')(out_en)
    for n in np.arange(0, len(params['N2_layers'])):
        y = tf.keras.layers.Dense(params['N2_layers'][n], activation='selu', name='v_to_e_' + str(n),  kernel_regularizer=regularizers.l2(0.01))(y)
        if params['bat_n_all']:
            y = tf.keras.layers.BatchNormalization(name='v_to_e_' + str(n) +'_norm')(y)
    N2 = tf.keras.layers.Dense(params['N_evals'], activation='linear', name = 'v_to_e_out',  kernel_regularizer=regularizers.l2(0.01))(y)
    
    AE=tf.keras.Model(inputs=[in_enc, in_eig],outputs=[out_dec,N2, N1, out_en])
        
    in_lat = tf.keras.layers.Input(shape=(params['N_units'], ))
    N2_mod = tf.keras.models.Sequential(); N2_mod.add(in_lat)
    for i in np.arange(0,len(AE.layers)):
        if 'v_to_e' in AE.layers[i].name:
            N2_mod.add(AE.layers[i])

    encoder = tf.keras.Model(inputs=[in_enc], outputs=[out_en])
    N1_mod = tf.keras.Model(inputs=[in_eig],outputs=[N1])

    return AE, autoenc, N1_mod, N2_mod, encoder, decoder

AE, autoenc, N1_mod, N2_mod, encoder, decoder = build_AE()


##### Training
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE, seed=rand_seed).batch(np.int64(params['B_size'])) # Training set batching
train_eigen = tf.data.Dataset.from_tensor_slices(train_eigs).shuffle(BUFFER_SIZE, seed=rand_seed).batch(np.int64(params['B_size'])) # Training set batching

generator_optimizer = tf.keras.optimizers.Adam(params['L_rate_AE']) # Optimizer

for epoch in range(0,params['Epochs']):
    for meshes,eig in zip(train_dataset, train_eigen):
        with tf.GradientTape() as gen_tape:
            [g_mesh, g_eval, N1_lat, true_lat] = AE((meshes,eig[:,0:params['N_evals']]), training=True)
            loss_ae = tf.keras.losses.mean_squared_error(meshes, g_mesh)
            loss_N2 = params['W_N2']*tf.keras.losses.mean_squared_error(eig[:,0:params['N_evals']], g_eval)
            loss_N1 = params['W_N1']*tf.keras.losses.mean_squared_error(true_lat, N1_lat)

        gradients_of_generator = gen_tape.gradient([loss_ae, loss_N2, loss_N1],  AE.trainable_variables)
        temp = generator_optimizer.apply_gradients(zip(gradients_of_generator, AE.trainable_variables))
    my = decoder(N1_mod(e[test_subj,:]))
    err_my_subj = np.mean(np.sum((my[0:len(test_subj),:,:]-pix[test_subj,:])**2,2))
    print(str(epoch) + ":ERR_TEST: %10.3e "% (err_my_subj))

    [g_mesh, g_eval, N1_lat, true_lat] = AE((test_images,test_eigs))

    if (epoch+1) % save_every == 0:
        encoder.save('./models/' + model_name + '/encoder_' + str(epoch) + '.h5')
        decoder.save('./models/' + model_name + '/decoder_' + str(epoch)  + '.h5')
        N1_mod.save('./models/' + model_name + '/N1_' + str(epoch)  + '.h5')
        N2_mod.save('./models/'+ model_name + '/N2_' + str(epoch)  + '.h5')

encoder.save('./models/' + model_name + '/encoder.h5')
decoder.save('./models/' + model_name + '/decoder.h5')
N1_mod.save('./models/' + model_name + '/N1.h5')
N2_mod.save('./models/'+ model_name + '/N2.h5')

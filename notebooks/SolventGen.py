import sys
sys.path.insert(0,'/home/mohit/Downloads/code_notebooks/deep_boltzmann')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras import backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import ast

from deep_boltzmann.models import ParticleDimer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet, create_RealNVPNet
from deep_boltzmann.sampling import GaussianPriorMCMC
from deep_boltzmann.sampling.latent_sampling import BiasedModel
from deep_boltzmann.sampling.permutation import HungarianMapper
from deep_boltzmann.util import load_obj, save_obj

from deep_boltzmann.sampling.analysis import free_energy_bootstrap, mean_finite, std_finite

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

trajdict = np.load('output.npz')
traj_closed_train = trajdict['traj_closed_train_hungarian']
traj_open_train = trajdict['traj_open_train_hungarian']
traj_closed_test = trajdict['traj_closed_test_hungarian']
traj_open_test = trajdict['traj_open_test_hungarian']
x = np.vstack([traj_closed_train, traj_open_train])
xval = np.vstack([traj_closed_test, traj_open_test])


MLmodel = Sequential()
MLmodel.add(Dense(128, activation='relu', input_dim=4))

MLmodel.add(Dense(128, activation='relu'))
'''
MLmodel.add(Dense(128, activation='relu'))
MLmodel.add(Dropout(0.7))
'''
om = keras.optimizers.Adam(lr=0.0001)
MLmodel.add(Dense(76))
def energyLoss(y_true, y_pred):
    #end  = (ParticleDimer().energy_tf(y_pred) - ParticleDimer().energy_tf(y_true) )**2
    return K.sqrt(K.mean(K.square(ParticleDimer().energy_tf(y_true) - ParticleDimer().energy_tf(y_pred) )))
    #return mean_squared_error(ParticleDimer().energy(y_true)[:,], ParticleDimer().energy(y_pred)[:,] )

MLmodel.compile(optimizer = om, loss = energyLoss, metrics= [energyLoss])
Losses =MLmodel.fit(x= traj_open_train[5000:55000, 0:4] , y = traj_open_train[5000:55000],epochs = 1000, batch_size = 508)
print(Losses)
for layer in MLmodel.layers():


import numpy as np
from utils.callbacks import LossLogger
import matplotlib.pyplot as plt
import tensorflow as tf

# Alias important packages/modules

tfk = tf.keras
tfkl = tfk.layers

class QuantumMLModels(tfk.Sequential):

    def __init__(self,layers,X_train=None,y_train=None,X_test=None,y_test=None,name="QMLmodel"):
        super().__init__(layers,name)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.history = None

    def stash_data(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train 
        self.X_test = X_test
        self.y_test = y_test
    
    def store_history(self,history):
        self.history = history

    def parity_patch_plot(self,X_sample,y_truth,sample_dims=0):
        """
        Args: 
        
        X_sample 

        y_truth

        """

        Y_predict = self.predict(X_sample)

        if(Y_predict.shape[1] != 1):
            y_predict = Y_predict[:,sample_dims].flatten()
        else:
            y_predict = Y_predict.flatten()

        fig, ax = plt.subplots()

        ax.hist2d(y_predict.flatten(),y_truth[:,sample_dims].flatten(), bins=64, cmap='Blues', alpha=0.9)

        ax.set_xlim(ax.get_ylim())
        ax.set_ylim(ax.get_xlim())

        ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--')
        
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('True Value')
    
    def parity_plot(self,X_sample,y_truth,sample_dims=0):

        Y_predict = self.predict(X_sample)

        if(Y_predict.shape[1] != 1):
            y_predict = Y_predict[:,sample_dims].flatten()
        else:
            y_predict = Y_predict.flatten()

        fig,ax = plt.subplots()

        ax.plot(y_truth[:,sample_dims],y_truth[:,sample_dims],'r-')
        ax.plot(y_predict,y_truth[:,sample_dims],'b.')

        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('True Value')

class RFStackedNetworks()



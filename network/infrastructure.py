
import numpy as np
from utils.callbacks import LossLogger
import matplotlib as plt
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

    def prediction_viewer(self,X_sample,y_truth,sample_dims):
        """
        Args: 
        
        X_sample 

        y_truth

        """

        y_predict = self.predict(X_sample)

        plt.plot(y_truth,y_truth,'-k')
        plt.plot(y_predict,y_truth,'bo')
        plt.xlabel('Prediction')
        plt.ylabel('True Value')




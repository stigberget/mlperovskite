import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tfk = tf.keras

class LossLogger(tfk.callbacks.Callback):

    def __init__(self,batch_nums,log_frequency=2,loss_type='loss',figurative=0):

        super(LossLogger,self).__init__()

        self.loss_type = loss_type
        self.log_frequency = log_frequency
        self.batch_nums = batch_nums
        self.figurative = figurative

        if(figurative):
            self.val_loss_type,axlabel = self._set_loss_type()
            self.floss,self.axloss = plt.subplots(1,1)
            self.axloss.set_xlabel("Epochs")
            self.axloss.set_ylabel(axlabel)
            #self.floss.show()

        self.train_batch_nums = 0
        self.test_batch_nums = 0
        self.min_val_loss = None

        self.epochs = []
        self.loss = []
        self.val_loss = []
        self.train_batch_loss = []
        self.test_batch_loss = []
        self.train_batch_log_intervals = []
        self.test_batch_log_intervals  = []


    def _set_loss_type(self):

        loss_types = ["loss","accuracy"]

        if(self.loss_type == loss_types[0]):
            axlabel = "Loss"
        elif(self.loss_type == loss_types[1]):
            axlabel = "Accuracy"
        else: 
            print("Loss not recognized. Using supplied label name on axis")
            axlabel = self.loss_type
        
        val_loss = "val_" + self.loss_type

        return val_loss,axlabel

            

    def on_epoch_end(self,epoch,logs):
        
        
        self.train_batch_log_intervals = np.linspace(0,epoch+1,self.train_batch_nums,axis=0).reshape((self.train_batch_nums,1))
        self.test_batch_log_intervals = np.linspace(0,epoch+1,self.test_batch_nums,axis=0).reshape((self.test_batch_nums,1))

        self.epochs.append(epoch)
        self.loss.append(logs[self.loss_type])
        self.val_loss.append(logs["val_loss"])

        if(self.figurative):

            if(epoch % self.log_frequency == 0):
                
                # Update loss axis
                self.axloss.plot(self.epochs,self.loss,'-b',self.epochs,self.val_loss,'-r')
                self.axloss.plot(self.train_batch_log_intervals,self.train_batch_loss,'-b',alpha=0.4)
                self.axloss.plot(self.test_batch_log_intervals,self.test_batch_loss,'-r',alpha=0.4)
                self.axloss.legend(["Training Loss","Validation Loss"])    


    def on_train_batch_end(self,batch,logs):
        self.train_batch_nums += 1
        self.train_batch_loss.append(logs[self.loss_type])
    
    def on_test_batch_end(self,batch,logs):
        self.test_batch_nums += 1 
        self.test_batch_loss.append(logs[self.loss_type])
    
        

            

            


import numpy as np
from utils.hyperparameter import HyperParameter as hp
from utils.callbacks import LossLogger

class TuningScheduler:

    def __init__(self,hyperparameters,hypermodel,X_train,y_train,X_val,y_val,callbacks=None):

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.hyperparameters = hyperparameters
        self.hypermodel = hypermodel
        if(callbacks is None):
            self._callbacks = []
        else:
            self._callbacks = callbacks
        
        self.tuned_hyperparameters = {}
        self.print_frequency = 5

    def auto_tune(self,num_rand_configs,num_grid_configs,num_epochs,batch_size,lims=[0.9,1.1],strap_configs=None,figurative=0,verbose=0,log_frequency=1):
        """
        Args:

        num_rand_configs: 
        Int specifying the number of hyperparameter configurations to randomly sample

        num_grid_configs: 
        Dict or int specifying number of grid points that we sample. 
        Dict keys must correspond to names of hyperparameters defined in the passed HyperParameter object

        num_epochs: 
        Number of epochs we wish to run for each hyperparameter configuration

        lims: 
        List (with 2 elements) specifying the adjusted range to sample in the grid sampler following random tuning. 
        Element 0 must be < 1 and Element 1 must be > 1

        strap_configs:
        List containing the names of the hyperparameters that are to be fixed in the grid tuning step. 
        Dict keys must correspond to names of hyperparameters defined in the passed HyperParameter object. Default is None.

        figurative: 
        Plot losses during hyperparameter tuning for each configuration. Permitted values 0 and 1

        verbose:
        Print information to the command line. 
        verbose=0: No printing
        verbose=1: Print hyperparameter config ID during tuning 
        verbose=2: Print losses, hyperparameters, hyperparameter config ID during tuning

        log_frequency: 
        Frequency of updates to the command line or loss figures
        """

        self.tuned_hyperparameters = self.rand_tune(num_rand_configs,num_epochs,batch_size,figurative=figurative,verbose=verbose,log_frequency=log_frequency)
        hyperparameters_copy = self.hyperparameters
        grid_hparams = self.__set_grid_hyperparameters(lims[0],lims[1],num_grid_configs,strap_configs)
        self.hyperparameters = grid_hparams

    def rand_tune(self,num_configs,num_epochs,batch_size,save_best_model=False,figurative=0,verbose=0,log_frequency=1):

        if(verbose > 0):
            print("Running random hyperparameter search...")

        

        val_loss_opt_config = None

        for config in range(num_configs):

            if((config%self.print_frequency) == 0 and verbose > 0):
                print(f"{config} out of {num_configs} sampled")


            for key in self.hyperparameters.sampler.keys():

                self.hyperparameters.values[key] = self.hyperparameters.sampler[key].sample()

            #TODO: check dims of inputs.shape
            batch_nums = (self.X_train.shape[1] + self.X_val.shape[1])/batch_size

    
            loss_logger = LossLogger(batch_nums=batch_nums,log_frequency=log_frequency,figurative=figurative)
            self._callbacks.append(loss_logger)

            
            model = self.hypermodel.build(self.hyperparameters)
            model.fit(epochs=num_epochs,batch_size=batch_size,callbacks=self._callbacks)

            model_loss = np.min(loss_logger.val_loss)

            if(loss_logger.min_val_loss < val_loss_opt_config or val_loss_opt_config is None):
                tuned_hparams = self.hyperparameters.values
                val_loss_opt_config = model_loss
                #TODO: routine for saving best model if save_best_model is TRUE

        if(verbose > 0):
            print("Random hyperparameter search complete.")

        return tuned_hparams


    def grid_tune(self,num_epochs,figurative=0,verbose=0,log_frequency=1):

        if(verbose > 0):
            print("Running grid hyperparameter search...")

        val_loss_opt_config = None

        tuning_sequence,total_num_configs = self.__generate_tuning_sequence()

        # Get number of configs we sample
        for configID in range(total_num_configs):

            for key in self.hyperparameters.keys():
                index = (tuning_sequence[key])[configID]
                self.hyperparameter.values[key] = self.hyperparameters.sampler[key].sample(index)

            tuning_callback = None

            if(self._callbacks is not None):
                self._callbacks.append(tuning_callback)
            else: 
                self._callbacks = [tuning_callback]

    
            loss_logger = LossLogger(batch_nums=self.batch_nums,log_frequency=log_frequency,figurative=figurative)
            self._callbacks.append(loss_logger)

            model = self.hypermodel.build(self.hyperparameters)
            model.fit(epochs=num_epochs,callbacks=self._callbacks)

            model_loss = np.min(loss_logger.val_loss)

            if(loss_logger.min_val_loss < val_loss_opt_config or val_loss_opt_config is None):
                tuned_hparams = self.hyperparameters.values
                val_loss_opt_config = model_loss
                #TODO: routine for saving best model if save_best_model is TRUE

        if(verbose > 0):
            print("Grid based hyperparameter search completed.")
        
        return tuned_hparams
        

    def __set_grid_hyperparameters(self,lfactor,hfactor,num_configs,strap_configs):

        gridhp = hp.HyperParameter()

        configs = {}

        if(type(num_configs) == 'dict'):
            if(len(num_configs) != len(self.tuned_hyperparameters)):
                raise KeyError("Keys of specified number of sample configurations does not match keys of hyperparameters")


        for key in self.hyperparameters.values.keys():

            if(type(num_configs) == 'int'):
                configs["num_configs"] = num_configs  
            else:
                configs["num_configs"] = num_configs[key]
            
            if(key in strap_configs):
                value = self.tuned_hyperparameters.values[key]
                gridhp.fixed(key,value)
                continue
            elif(self.hyperparameters.param_type[key] == 'float'):
                min_val = self.tuned_hyperparameters.values[key]*lfactor
                max_val = self.tuned_hyperparameters.values[key]*hfactor
                gridhp.float(key,min_val,max_val,sampler_type='grid',configs=configs)
            elif(self.hyperparameters.param_type[key] == 'int'):
                min_val = int(self.tuned_hyperparameters.values[key]*lfactor)
                max_val = int(self.tuned_hyperparameters.values[key]*hfactor)
                gridhp.int(key,min_val,max_val,sampler_type='grid',configs=configs)
            elif(self.hyperparameters.param_type[key] == 'boolean'):
                value = self.tuned_hyperparameters.values[key]
                gridhp.fixed(key,value)
            elif(self.hyperparameters.param_type[key] == 'selection'):
                value = self.tuned_hyperparameters.values[key]
                gridhp.fixed(key,value)
        
        return gridhp

    def __generate_tuning_sequence(self):

        indices = []

        for key in self.hyperparameters.keys():
            num_configs = self.hyperparameters.sampler[key].get_size()
            indices.append(np.linspace(1,num_configs,num_configs))


        index_sequence = np.meshgrid(*indices)
        
        tuning_sequence = {}

        for i,key in enumerate(self.hyperparamters.keys()):
            if(i == 0):
                total_num_configs = index_sequence[i].flatten().shape[0]
    
            tuning_sequence[key] = index_sequence[i].flatten()

        return tuning_sequence,total_num_configs




            
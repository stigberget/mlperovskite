

from aifc import Error
from ast import operator
from math import floor
from warnings import WarningMessage
import numpy as np



class HyperParameter:

    def __init__(self,name=None,values=None):

        self.values = {}
        self.sampler =  {}
        self.param_type = {}

        if(name is not None and values is not None):
            self.values[name] = values

    def selection(self,name,selections,sampler_type='default',configs=None):

        self.param_type[name] = 'selection'
        self.sampler[name] = self.__init_selection_sampler(selections,sampler_type,configs)
        
        
    def float(self,name,min_val,max_val,sampler_type='default',configs=None):

        self.param_type[name] = 'float'
        self.sampler[name] = self.__init_float_sampler(min_val,max_val,sampler_type,configs)
        

    def boolean(self,name,default=False,sampler_type='default',configs=None):

        self.param_type[name] = 'boolean'
        self.sampler[name] = self.__init_bool_sampler(default,sampler_type,configs)
        

    def int(self,name,min_val,max_val,sampler_type='default',configs=None):

        self.param_type[name] = 'int'
        self.sampler[name] = self.__init_int_sampler(min_val,max_val,sampler_type,configs)

    def fixed(self,name,value):

        self.param_type[name] = 'fixed'
        self.sampler[name] = self.__init_fixed_sampler(value)
        
    def __init_selection_sampler(self,selections,sampler_type,configs):
        if(sampler_type == 'uniform' or sampler_type == 'default'):
            sampler = SelectUS(selections,configs)
        elif(sampler_type == 'grid'):
            sampler = SelectGS(selections,configs)
        else: 
            emsg = "Sampler type of type " + str(sampler_type) + " for selection HyperParameter is not valid/recognized"
            raise ValueError(emsg)
        return sampler

    def __init_float_sampler(self,min_val,max_val,sampler_type,configs):

        if(sampler_type == 'uniform' or sampler_type == 'default'):
            sampler = FloatUS(min_val,max_val,configs)
        elif(sampler_type == 'lognormal'):
            sampler = FloatLNS(min_val,max_val,configs)
        elif(sampler_type == 'grid'):
            sampler = FloatGS(min_val,max_val,configs)
        elif(sampler_type == 'normal'):
            raise NotImplementedError
        elif(sampler_type == 'constrained'):
            raise NotImplementedError
        else:
            emsg = "Sampler type of type " + str(sampler_type) + " for float HyperParameter is not valid/recognized"
            raise ValueError(emsg) 
        return sampler
    
    def __init_bool_sampler(self,default,sampler_type,configs):
        if(sampler_type == 'uniform' or sampler_type == 'default'):
            sampler = BooleanUS(default,configs)
        else: 
            emsg = "Sampler type of type " + str(sampler_type) + " for boolean HyperParameter is not valid/recognized"
            raise ValueError(emsg) 
        return sampler
    
    def __init_int_sampler(self,min_val,max_val,sampler_type,configs):
        if(sampler_type == 'uniform' or sampler_type == 'default'):
            sampler = IntUS(min_val,max_val,configs)
        elif(sampler_type == 'lognormal'):
            sampler = IntLNS(min_val,max_val,configs)
        elif(sampler_type == 'grid'):
            sampler = IntGS(min_val,max_val,configs)
        else:
            emsg = "Sampler type of type " + str(sampler_type) + " for int HyperParameter is not valid/recognized"
            raise ValueError(emsg) 
        return sampler

    def __init_fixed_sampler(self,value):
        sampler = FixedSampler(value)
        return sampler






class HPSampler:

    def __init__(self,min_val,max_val,configs): 
        
        self.configs = configs
        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng()

    
    
    def check_configs(self,supported_configs):

        if(self.configs is None):
            return
        elif(self.configs is not None and supported_configs is None):
            wmsg = "Sampling configs have been defined but for the given sampler no sampling configs can be provided"
            raise SyntaxWarning(wmsg)
        
        for key in self.configs.keys():
            if key not in supported_configs:
                emsg = "Key of specified sampling configs not recognized, permissible configs are " + str(supported_configs)
                raise TypeError(emsg)

class FixedSampler(HPSampler):

    def __init__(self,value):
        self._value = value
        configs = {"num_configs":1}
        super().__init__(min_val=None,max_val=None,configs=configs)

    def sample(self):
        return self._value

    

class GridSampler(HPSampler):

    def __init__(self,min_val,max_val,configs):

        self.index = -1 
        super().__init__(min_val,max_val,configs)
        
        supported_configs = ["step_size","num_configs"]
        self.check_configs(supported_configs)
    
    def get_size(self):
        return self.configs["num_configs"]
    
    def get_current_index(self):
        return self.index

    def sample(self,index=None):

        if(index is not None):
            self.index = index
        else: 
            self.index += 1 

        if(self.index > (self.configs["num_configs"]-1)):
            raise IndexError("Index exceeds matrix dimensions")
        
        value = self.min_val + (self.index)*self.configs["step_size"]

        assert(value <= self.max.val)

        return value

        


class FloatGS(GridSampler):

    def __init__(self,min_val,max_val,configs):

        min_val = float(min_val)
        max_val = float(max_val)

        configs["step_size"] = (max_val - min_val)/configs["num_configs"]

        super().__init__(min_val,max_val,configs)

        

class IntGS(GridSampler):

    def __init__(self,min_val,max_val,configs):

        min_val = int(min_val)
        max_val = int(max_val)

        if((max_val - min_val)%configs["num_configs"] != 0):
            raise Error("Grid search does not produce step sizes of type int")

        configs["step_size"] = int((max_val - min_val)/configs["num_configs"])

        super().__init__(min_val,max_val,configs)

    def sample(self):
        value = super().sample()
        return int(value)

class SelectGS(IntGS):

    def __init__(self,selection, configs):

        self.selection = selection
        
        if(configs is None):
            configs = {}
            configs["num_configs"] = len(selection)

        super().__init__(min_val=0, max_val=configs["num_configs"], configs=configs)
    
    def sample(self):
        index = super().sample()
        return self.selection["index"]


class LogNormalSampler(HPSampler):

    def __init__(self,min_val,max_val,configs=None):

        

        if(configs is None):
            configs = {}
            configs["mean"] = 0.0
            configs["sigma"] = 1.0
        
        super().__init__(min_val,max_val,configs)

        supported_configs = ["mean","sigma"]

        self.check_configs(supported_configs)
        

    def sample(self):
        iteration = 1
        max_iter = 1e3

        while True:
            realization  = self.rng.lognormal(self.configs["mean"],self.configs["sigma"])
            if(realization < self.min_val or realization > self.max_val):
                continue
            elif(max_iter < iteration):
                raise WarningMessage("Maximum number of iterations reached")
            else:
                break
            iteration += 1


        return realization
    

class FloatLNS(LogNormalSampler):

    def __init__(self,min_val,max_val,configs):

        min_val = float(min_val)
        max_val = float(max_val)

        super().__init__(min_val,max_val,configs)
    
    def sample(self):
        realization = super().sample()
        return float(realization)

    
class IntLNS(LogNormalSampler):

    def __init__(self, min_val, max_val, configs):

        min_val = int(min_val)
        max_val = int(max_val)
        super().__init__(min_val, max_val, configs)

    def sample(self):
        realization = super().sample()
        return int(round(realization))

    
class UniformSampler(HPSampler):

    def __init__(self,min_val,max_val,configs=None,supported_configs=None):

        super().__init__(min_val,max_val,configs)

        self.check_configs(supported_configs)

        
    
    def sample(self):
        realization = self.rng.uniform(self.min_val,self.max_val)
        return realization

class FloatUS(UniformSampler):

    def __init__(self,min_val,max_val,configs=None):
        min_val = float(min_val)
        max_val = float(max_val)
        super().__init__(min_val,max_val,configs)

    def sample(self):
        realization = super().sample()
        return float(realization)

class IntUS(UniformSampler):
    
    def __init__(self, min_val, max_val, configs=None):
        min_val = int(min_val)
        max_val = int(max_val+1)
        super().__init__(min_val, max_val, configs)

    def sample(self):
        realization = super().sample()
        return int(floor(realization))

class SelectUS(UniformSampler):

    def __init__(self,selections,configs=None):

        self.selections = selections

        num_selections = len(selections)
        supported_configs = ["weights"]
        super().__init__(min_val=0,max_val=num_selections,configs=configs,supported_configs=supported_configs)

        # TODO : Implement assertion of length of weights list/array

    def sample(self):

        realization = super().sample()
        
        if(self.configs is None):
            index = int(np.floor(realization))
        else:
            norm_factor = np.sum(self.configs["weights"])
            weights = self.configs["weights"]/norm_factor
            iweights = np.cumsum(weights)
            intervals = iweights*len(self.selections)
            
            dri = realization - intervals
            imin = np.argmin(np.abs(dri))
            if(dri[imin] < 0):
                index = int(imin - 1)
            else:
                index = int(imin)

        return self.selections[index]

    

class BooleanUS(UniformSampler):

    def __init__(self,default=False,configs=None):
        supported_configs = ["skewness"]

        if configs is None:
            configs = {} 
            configs["skewness"] = 0.5
        super().__init__(min_val=0,max_val=1,configs=configs,supported_configs=supported_configs)

        self.default = default

        if(configs["skewness"] > 1 or configs["skewness"] < 0):
            raise ValueError("Skewness values must lie between 0 and 1") 
    
    def sample(self):

        realization = super().sample()

        if(realization < self.configs["skewness"]):
            return self.default
        else:
            return (not self.default)

        
        



        







    

        


        




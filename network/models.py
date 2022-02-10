import tensorflow as tf
from network.infrastructure import QuantumMLModels
import keras_tuner as kt


# Alias important packages

tfk = tf.keras
tfkl = tfk.layers


class HyperTModel:
    def __init__(self,feature_ndims):
        self.feature_ndims = feature_ndims

    def build(self,hparams):

        model = QuantumMLModels(layers=[tfkl.InputLayer(input_shape=self.feature_ndims)])

        lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = hparams.values["learning_rate"],
            decay_steps = hparams.values["decay_steps"],
            decay_rate = hparams.values["decay_rate"]
            )
        
        ADAM = tfk.optimizers.Adam(
            learning_rate = lr_schedule,
            beta_1 = hparams.values["beta_1"],
            beta_2 = hparams.values["beta_2"],
            clipvalue = hparams.values["clipvalue"]
            )
        initializer = tfk.initializers.HeUniform()
        model.add(tfkl.Dense(units=12,activation='elu',kernel_initializer=initializer))
        model.add(tfkl.Dense(units=12,activation='relu',kernel_initializer=initializer))
        model.add(tfkl.Dense(units=12,activation=hparams.values["activation_1"],kernel_initializer=initializer))
        if(hparams.values["layer_1"]): 
            model.add(tfkl.Dense(units=12,activation='relu',kernel_initializer=initializer))
        if(hparams.values["layer_2"]):
            model.add(tfkl.Dense(units=12,activation=hparams.values["activation_2"],kernel_initializer=initializer))

        model.add(tfkl.Dense(units=5,activation='relu',kernel_initializer=initializer))
        model.compile(optimizer=ADAM,loss=tfk.losses.mean_squared_error)

        return model


class HyperModel:

    def __init__(self,feature_ndims):
        self.feature_ndims_ = feature_ndims

    def build_model(self,hp):
        model = QuantumMLModels(layers=tfkl.InputLayer(input_shape=(self.feature_ndims_,)))
        lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = hp.Float(name="learning_rate",min_value=1e-3,max_value=7.5e-2),
            decay_steps = hp.Int(name="decay_steps",min_value=80,max_value=120),
            decay_rate = hp.Float(name="decay_rate",min_value=0.88,max_value=0.98)
            )
        ADAM = tfk.optimizers.Adam(
            learning_rate = lr_schedule,
            beta_1 = hp.Float(name="beta_1",min_value=0.87,max_value=0.93),
            beta_2 = hp.Float(name="beta_2",min_value=0.988,max_value=0.999),
            clipvalue = hp.Float(name="clipvalue",min_value=0.08,max_value=4)
            )
        
        network_type = hp.Choice(name="network_depth",values=[1,2,3,4])
        layer_units = hp.Int(name="layer_units",min_value=8,max_value=20)
        initializer = tfk.initializers.HeUniform()
        model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
        model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
        model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))

        loss = tfk.losses.mean_absolute_error

        if(network_type >= 1):
            model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
        if(network_type >= 2):
            model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
        if(network_type >= 3):
            model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
        if(network_type >= 4):
            model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))

        model.add(tfkl.Dense(units=1,activation=None,kernel_initializer=initializer))
        model.compile(optimizer=ADAM,loss=loss)

        return model
                
def bandgap_model(feature_ndims):
    model = QuantumMLModels(layers=tfkl.InputLayer(input_shape=(feature_ndims,)))

    initializer = tfk.initializers.HeUniform()

    layer_units = 12

    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 0.01,
            decay_steps = 115,
            decay_rate = 0.922
            )
    
    ADAM = tfk.optimizers.Adam(
            learning_rate = lr_schedule,
            beta_1 = 0.927,
            beta_2 = 0.995,
            clipvalue = 4
            )
    
    mae = tfk.losses.mean_absolute_error

    model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=5,activation=None,kernel_initializer=initializer))

    model.compile(optimizer=ADAM,loss=mae)

    return model

def fermi_level_model(feature_ndims):
    model = QuantumMLModels(layers=tfkl.InputLayer(input_shape=(feature_ndims,)))

    initializer = tfk.initializers.HeUniform()

    layer_units = 12

    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = 0.01,
            decay_steps = 100,
            decay_rate = 0.9
            )
    
    ADAM = tfk.optimizers.Adam(
            learning_rate = lr_schedule,
            #beta_1 = 0.927,
            #beta_2 = 0.995,
            clipvalue = 0.8
            )
    
    mae = tfk.losses.mean_absolute_error

    model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    model.add(tfkl.BatchNormalization())
    #model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    #model.add(tfkl.BatchNormalization())
    model.add(tfkl.Dense(units=layer_units,activation='elu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=layer_units,activation='relu',kernel_initializer=initializer))
    model.add(tfkl.Dense(units=1,activation=None,kernel_initializer=initializer))

    model.compile(optimizer=ADAM,loss=mae)

    return model




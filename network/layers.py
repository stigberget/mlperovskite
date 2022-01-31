import tensorflow as tf
from network.infrastructure import QuantumMLModels


# Alias important packages

tfk = tf.keras
tfkl = tfk.layers

class HyperModel:
    def __init__(self,feature_ndims):
        self.feature_ndims = feature_ndims

    def build(self,hparams):

        model = QuantumMLModels()

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

        model.add(tfkl.Input(shape=self.feature_ndims))
        model.add(tfkl.Dense(units=10,activation='elu'))
        model.add(tfkl.Dense(units=10,activation='relu'))
        model.add(tfkl.Dense(units=10,activation=hparams.values["activation_1"]))

        model.compile(optimizer=ADAM,loss='rmse')


        



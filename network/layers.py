import tensorflow as tf

# Alias important packages
tfk = tf.keras
tfkl = tfk.layers



class GCNLayer(tfkl.Layer):

    def __init__(self,activation=None,**kwargs):

        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self,input_shape):

        node_shape,adj_shape = input_shape

        self.w = self.add_weight(shape=(node_shape[2],node_shape[2]),name='w')


    def call(self,inputs):

        # split input into nodes, adj
        nodes, adj = inputs
        # compute degree
        degree = tf.reduce_sum(adj, axis=-1)
        # GCN equation
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)

        return out,adj

class GRLayer(tf.keras.layers.Layer):
    """A GNN layer that computes average over all node features"""

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction




        




        



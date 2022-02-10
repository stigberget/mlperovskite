import matplotlib.pyplot as plt
import numpy as np

def parity_plot(model,X,y_truth,labels=None,title=None):

    if(labels is None):
        labels = ["Predicted Value","True Value"]
    
    fig,ax = plt.subplots()

    y_predict = model.predict(X)

    
    ax.plot(y_predict,y_truth,'b.')

    ax.set_xlim(ax.get_ylim())
    ax.set_ylim(ax.get_xlim())

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') 

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.grid(True)

    if(title is not None):
        ax.set_title(title)


def patch_parity_plot(model,X,y_truth,labels=None):

    if(labels is None):
        labels = ["Predicted Value","True Value"]
    
    y_predict = model.predict(X)

    fig, ax = plt.subplots()

    ax.hist2d(y_predict,y_truth,bins=64,cmap='Blues',alpha=1)

    ax.set_xlim(ax.get_ylim())
    ax.set_ylim(ax.get_xlim())

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'r--') 

    ax.grid(True)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])


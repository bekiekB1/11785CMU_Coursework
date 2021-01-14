"""Problem 3 - Training on MNIST"""
import numpy as np

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)
from mytorch.nn.linear import Linear
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.loss import *
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor
from mytorch.nn.activations import *
# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    mnist_model = Sequential(Linear(784,20),ReLU(),Linear(20,10))
    #mnist_model = Sequential(Linear(784,20),BatchNorm1d(20),ReLU(),Linear(20,10))
    creterion = CrossEntropyLoss()
    mdl_optimizer = SGD(mnist_model.parameters(), momentum=0.9,lr=0.1)
    


    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(mnist_model,mdl_optimizer,creterion,train_x, train_y, val_x, val_y)
    
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    # TODO: Implement me! (Pseudocode on writeup)
    model.train()
    np_samples = train_x.shape[0]
    for epoch in range(num_epochs):
        indx_shuffle = np.random.permutation(train_x.shape[0])
        train_x, train_y = train_x[indx_shuffle], train_y[indx_shuffle]
        #batches = get_batch(train_x,train_y)
        batches = list(zip(np.array_split(train_x,np_samples//100),np.array_split(train_y,np_samples//100)))
        for i, (batch_data, batch_labels) in enumerate(batches):
            #if(i*BATCH_SIZE>=np_samples):
            #    break
            optimizer.zero_grad()
            out = model(Tensor(train_x))
            loss = criterion(out,Tensor(train_y))
            loss.backward()
            optimizer.step()
            if(i%100==0 and i!=0):
                accuracy = validate(model,val_x,val_y)
                val_accuracies.append(accuracy)
                model.train()  
        print(f'Epoch:{epoch+1} \t Validation AC: {val_accuracies[-1]}')
    return val_accuracies


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval() 
    num_samples = val_x.shape[0]
    #val_batches = get_batch(val_x,val_y)
    batches = list(zip(np.array_split(val_x,num_samples//100),np.array_split(val_y,num_samples//100)))
    num_correct = 0
    for i,(batch_data, batch_labels) in enumerate(batches):
        #if(i*BATCH_SIZE>=num_samples):
        #        break 
        out = model(Tensor(batch_data))
        batch_preds = np.argmax(out.data,axis=1)
        #print('\nPrediction on Batch:',batch_preds[:10])
        #print(f'\n Bach True label: {batch_labels}')
        num_correct += (batch_preds == batch_labels).sum()
    accuracy = (num_correct / len(val_x) *100)
    return accuracy

'''
#Generator for efficiency and Memory
def get_batch(data, labels, batch_size = 100):
    i = 0
    while True:
        if i*batch_size >= len(labels):
            i = 0
            idx = np.random.permutation(len(labels))
            data, labels = data[idx], labels[idx]
            continue
        else:
            X = data[i*batch_size:(i+1)*batch_size,:]
            y = labels[i*batch_size:(i+1)*batch_size]
            i += 1
            yield X,y
'''
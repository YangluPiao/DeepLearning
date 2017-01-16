from __future__ import print_function
import numpy as np
import numpy
import glob
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import matplotlib.pyplot as plt
import theano.tensor.nnet as nnet


from dropout import DropoutHiddenLayer
from utils import shared_dataset, load_data
from theano.tensor.nnet import conv2d
from figures import showpic
from figures import Dropout
from nns import LeNetConvPoolLayer, LogisticRegression, HiddenLayer, train_nn


def MY_CNN(fil, learning_rate=0.1, n_epochs=128,
                    nkerns=[64, 128, 256], batch_size=20):
    print('...downloading data',file=fil)
    pb = 4
    rng = np.random.RandomState(23455)
    datasets = load_data(pb=pb)
    
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    
    #showdata = load_data(pb=1, theano_shared=False)
    #show_x,show_y=showdata[2][0:16]
    #showpic(pb=pb, dataset=show_x)
    #showpic(pb=4.1, dataset=show_x)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
     # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    print('... building the model',file=fil)

    layer0_input = x.reshape((batch_size, 3, 32,32)) 
    #with border_mode='half', dimension of output = dimension of input/2. 
    layer0 = LeNetConvPoolLayer(
        rng=rng,
        input = layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        pool_enabled=True
        
    )#output: 64*32*32
    '''
    layer1 = LeNetConvPoolLayer(
        rng=rng,
        input = layer0.output,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        border_mode='half',
        pool_enabled=True
        
    )#output: 128*16*16
    
    layer2 = LeNetConvPoolLayer(
        rng=rng,
        input = layer1.output,
        image_shape=(batch_size, nkerns[1], 16, 16),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        border_mode='half',
        pool_enabled=True
        
    )#output: 256*8*8
    
    upsampling1 = T.nnet.abstract_conv.bilinear_upsampling(
        input=layer2.output,
        ratio=2,
        batch_size=batch_size,
        num_input_channels=nkerns[2],
    )
    #output: 256*16*16
    
    layer3 = LeNetConvPoolLayer(
        rng=rng,
        input = upsampling1,
        image_shape=(batch_size, nkerns[2], 16, 16),
        filter_shape=(nkerns[1], nkerns[2], 3, 3),
        pool_enabled=False
        
    )#output: 128*16*16
    
    added1=layer3.output+layer1.output#128*16*16
    #out1 = added1.flatten(2)
    
    upsampling2 = T.nnet.abstract_conv.bilinear_upsampling(
        input=added1,
        ratio=2,
        batch_size=batch_size,
        num_input_channels=nkerns[1],
    )#output:128*32*32
    
    layer4 = LeNetConvPoolLayer(
        rng=rng,
        input = upsampling2,
        image_shape=(batch_size, nkerns[1], 32, 32),
        filter_shape=(nkerns[0], nkerns[1], 3, 3),
        pool_enabled=False
        
    )#output: 64*32*32
    
    added2 = layer4.output+layer0.output
    
    layer5 = LeNetConvPoolLayer(
        rng=rng,
        input = added2,
        image_shape=(batch_size, nkerns[0], 32, 32),
        filter_shape=(3, nkerns[0], 3, 3),
        pool_enabled=False
        
    )
    '''
    flattened=layer0.output.flatten(2)#final output : 3*32*32=3072
    
    layer6=HiddenLayer(
        rng=rng,
        input=flattened,
        n_in=14400,
        n_out=512,
        activation=T.tanh
    )
    layer7 = LogisticRegression(input=layer6.output, n_in=512, n_out=10) 
    
    
    cost = layer7.negative_log_likelihood(y)
    
    #MSE cost function
    #cost = T.mean((layer7.y_pred - y) ** 2)
    
    test_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    validate_model = theano.function(
        [index],
        layer7.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )
    #params = layer7.params+ layer6.params+ layer5.params + layer4.params+ layer3.params+ layer2.params+ layer1.params+ layer0.params
    params = layer0.params+layer7.params + layer6.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    #Add momentum
    momentum =theano.shared(numpy.cast[theano.config.floatX](0.5), name='momentum')
    
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))    
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )
    '''
    output_pic = theano.function(
        [index],
        flattened,
        givens={ 
            x: test_set_x[:index]
       }
    )
    '''
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############

    print('... training')
    print('... training',file=fil)
    train_nn(fil=fil,train_model=train_model, validate_model=validate_model, test_model=test_model,
        n_train_batches=n_train_batches, n_valid_batches=n_valid_batches, n_test_batches=n_test_batches, 
        n_epochs=n_epochs, pic_out=False, verbose = True)
    

    
    
    
if __name__ == '__main__':
    K=open('CNN.txt','w')
    MY_CNN(K)
    K.close()
    
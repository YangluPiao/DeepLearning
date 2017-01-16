"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof.  Zoran Kostic

"""
import os
import sys
import numpy as np
import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T
import figures as test

def shared_dataset(pb,data_xy, borrow=True,train=False):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    '''
    if pb == 4:
        l_x=len(data_x)
        for i in range(l_x):
            data_x[i] = test.Dropout(data_x[i])
    '''
    if train==True:
        l_x=len(data_x)
        l_y=len(data_y)
        if pb == 2.1:
            for i in range(int(l_x/20)):
                dd=test.translate(test.unflatten(data_x[i]))
                data_x=numpy.vstack((data_x,test.flatten(dd)))
                if(i%200==0):
                    d=len(data_x)-l_x
                    print("number of augmented input data: %d" %d)
                data_y=numpy.append(data_y,data_y[i])
                if(i%200==0):
                    d=len(data_y)-l_y
                    print("number of augmented output data: %d" %d)
                
        elif pb== 2.2:
            for i in range(int(l_x/20)):
                dd=test.rotation(test.unflatten(data_x[i]))
                data_x=numpy.vstack((data_x,test.flatten(dd)))
                if(i%200==0):
                    d=len(data_x)-l_x
                    print("number of augmented input data: %d" %d)
                data_y=numpy.append(data_y,data_y[i])
                if(i%200==0):
                    d=len(data_y)-l_y
                    print("number of augmented output data: %d" %d)
        elif pb == 2.3:
            index=(numpy.random.random(int(l_x/20))*(int(l_x/20))).astype(int)
            for i in index:
                dd=test.flip(test.unflatten(data_x[i]))
                data_x=numpy.vstack((data_x,test.flatten(dd)))
                if(i%200==0):
                    d=len(data_x)-l_x
                    print("number of augmented input data: %d" %d)
                data_y=numpy.append(data_y,data_y[i])
                if(i%200==0):
                    d=len(data_y)-l_y
                    print("number of augmented output data: %d" %d)
        elif pb == 2.4:
            index_gaus=(numpy.random.random(int(l_x/40))*(int(l_x/40))).astype(int)
            index_uni=(numpy.random.random(int(l_x/40))*(int(l_x/40))).astype(int)
            for i in index_gaus:
                dd=test.add_noise(test.unflatten(data_x[i]))
                data_x=numpy.vstack((data_x,test.flatten(dd)))
                if(i%100==0):
                    d=len(data_x)-l_x
                    print("number of augmented input data: %d" %d)
                data_y=numpy.append(data_y,data_y[i])
                if(i%100==0):
                    d=len(data_y)-l_y
                    print("number of augmented output data: %d" %d)
            for j in index_uni:
                dd=test.add_noise(test.unflatten(data_x[j]),typ='Uniform')
                data_x=numpy.vstack((data_x,test.flatten(dd)))
                if(j%100==0):
                    d=len(data_x)-l_x
                    print("number of augmented input data: %d" %d)
                data_y=numpy.append(data_y,data_y[j])
                if(j%100==0):
                    d=len(data_y)-l_y
                    print("number of augmented output data: %d" %d)
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(pb,ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(pb=pb,data_xy=test_set)
        valid_set_x, valid_set_y = shared_dataset(pb=pb,data_xy=valid_set)
        train_set_x, train_set_y = shared_dataset(pb=pb,data_xy=train_set,train=True)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

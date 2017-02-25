from __future__ import division
import os
import timeit
import inspect
import sys
import numpy as np
import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from nn import myMLP, train_nn, LogisticRegression
sys.setrecursionlimit(15000)


def create_additioinal_label(x):
    y = numpy.zeros(x.shape)
    for i in range(x.shape[1]):
        if(i==0):
            y[:,i] = numpy.mod(x[:,i],2)
        else:
            y[:,i] = numpy.mod(numpy.sum(x[:,0:(i+1)], axis=1),2) 
    return y.astype(int)

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """


    X = numpy.random.randint(2, size=(num, nbit))

    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y


#TODO: implement RNN class to learn parity function
class RNN(object):
    def __init__(self, nh, nc, cs):
        # parameters of the model
        self.wx = theano.shared(name='wx',value=0.2 * numpy.random.uniform(-1.0, 1.0,(cs, nh)).astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.w = theano.shared(name='w',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.b = theano.shared(name='b',value=numpy.zeros(nc,dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',value=numpy.zeros(nh,dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sequence
        x = T.matrix()
        y_sequence = T.ivector('y_sequence')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sequence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sequence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sequence_nll = -T.mean(T.log(p_y_given_x_sequence)
                               [T.arange(x.shape[0]), y_sequence])

        sequence_gradients = T.grad(sequence_nll, self.params)

        sequence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sequence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
        self.sequence_train = theano.function(inputs=[x, y_sequence, lr],
                                              outputs=sequence_nll,
                                              updates=sequence_updates,
                                              allow_input_downcast=True)

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sequence_train(words, labels, learning_rate)
#TODO: implement LSTM class to learn parity function
class LSTM(object):
    #Idea is from http://apaszke.github.io/lstm-explained.html
    def __init__(self, nh, nc, cs):

        # parameters of the model
        self.wxi = theano.shared(name='wxi',value=0.2 * numpy.random.uniform(-1.0, 1.0,(cs, nh)).astype(theano.config.floatX))
        self.wxf = theano.shared(name='wxf',value=0.2 * numpy.random.uniform(-1.0, 1.0,(cs, nh)).astype(theano.config.floatX))
        self.wxo = theano.shared(name='wxo',value=0.2 * numpy.random.uniform(-1.0, 1.0,(cs, nh)).astype(theano.config.floatX))
        self.wxc = theano.shared(name='wxc',value=0.2 * numpy.random.uniform(-1.0, 1.0,(cs, nh)).astype(theano.config.floatX))
        self.wi = theano.shared(name='wi',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.wf = theano.shared(name='wf',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.wc = theano.shared(name='wc',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.bi = theano.shared(name='bi',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.bo = theano.shared(name='bo',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.bcin = theano.shared(name='bcin',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.c0 = theano.shared(name='c0',value=numpy.zeros(nh,dtype=theano.config.floatX))
        self.b = theano.shared(name='b',value=numpy.zeros(nc,dtype=theano.config.floatX))
        self.w = theano.shared(name='w',value=0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX))

        # bundle
               
        self.params = [self.wxi, self.wxf,self.wxo,self.wxc,self.wi, self.wf,  self.wo, self.wc,
                       self.bi, self.bf, self.bo, self.bcin, self.h0,self.c0, self.w, self.b]
        
        x = T.matrix()
        y_sequence = T.ivector('y_sequence')  # labels

        def recurrence(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wxi)+ T.dot(h_tm1, self.wi) + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wxf)+ T.dot(h_tm1, self.wf) + self.bf)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wxo)+ T.dot(h_tm1, self.wo) + self.bo)
            c_in_t = T.tanh(T.dot(x_t, self.wxc)+ T.dot(h_tm1, self.wc) + self.bcin)
            c_t = f_t*c_tm1+i_t*c_in_t
            h_t = o_t*T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t,c_t,s_t]

        [h,c,s],_ = theano.scan(fn=recurrence,
                              sequences=x,
                              outputs_info=[self.h0, self.c0,None],
                              #n_steps=self.input.shape[0]
                              )

        p_y_given_x_sequence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sequence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sequence_nll = -T.mean(T.log(p_y_given_x_sequence)
                               [T.arange(x.shape[0]), y_sequence])

        sequence_gradients = T.grad(sequence_nll, self.params)

        sequence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sequence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
        self.sequence_train = theano.function(inputs=[x, y_sequence, lr],
                                              outputs=sequence_nll,
                                              updates=sequence_updates,
                                              allow_input_downcast=True)

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sequence_train(words, labels, learning_rate)
#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit):
    #f=open('./problem_b/shallow_mlp_8bit.txt','w')
    #f=open('./problem_b/shallow_mlp_12bit.txt','w')
    f=open('./problem_b/deep_mlp_8bit.txt','w')
    #f=open('./problem_b/deep_mlp_12bit.txt','w')
    batch_size=24
    #n_hidden=24
    n_hidden=(24,24,24,24)
    learning_rate=0.08
    L1_reg = 0.0
    L2_reg = 0.0
    n_epochs=300
    n_hiddenLayers=4
    # generate datasets
    train_set = gen_parity_pair(n_bit, 2000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
        
        
    #training_enabled = T.iscalar('training_enabled')
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    #print('... building the model', file=f)
    
    rng = np.random.RandomState(23455)
    
    layers_input = x.reshape((batch_size,n_bit))
    layers=myMLP(rng, input=layers_input, n_in=n_bit, n_hidden=n_hidden, n_out=2, n_hiddenLayers=n_hiddenLayers)
    

    test_model = theano.function(
        [index],
        layers.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            #training_enabled: numpy.cast['int32'](0)
        }
    )
    
    validate_model = theano.function(
        [index],
        layers.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            #training_enabled: numpy.cast['int32'](0)
        }
    )
    
    
    
    cost =  layers.negative_log_likelihood(y) +  layers.L1*L1_reg + layers.L2_sqr*L2_reg 
    params = layers.params    
    grads = [T.grad(cost, param) for param in params]
    
    
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            #training_enabled: numpy.cast['int32'](1)
        }
    )

    ###############
    # TRAIN MODEL #
    ###############

    print('... training')
    #print('... training',file=f)
    train_nn(train_model=train_model, validate_model=validate_model, test_model=test_model,
            n_train_batches=n_train_batches, n_valid_batches= n_valid_batches,
             n_test_batches=n_test_batches, n_epochs=n_epochs, fil=f)
    f.close()
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(n_bit,fil):
    n_bit=n_bit
    #n_hidden = 10
    n_hidden = 100
    n_epochs=1000
    learning_rate=0.5
    n_win=7
    verbose=True
    f=fil
    print('... loading the dataset')
    print>>f,'... loading the dataset'
    # generate datasets
    train_set_x,train_set_y = gen_parity_pair(n_bit, 1000)
    valid_set_x,valid_set_y = gen_parity_pair(n_bit, 500)
    test_set_x,test_set_y  = gen_parity_pair(n_bit, 100)

    numpy.random.seed(100)
    #We need additional labels
    
    train_set_y = create_additioinal_label(train_set_x)

    valid_set_y = create_additioinal_label(valid_set_x)

    test_set_y = create_additioinal_label(test_set_x)

    n_out = 2


    
    print('... building the model')
    print>>f,'... building the model'
    rnn = RNN(
        nh=n_hidden,
        nc=n_out,
        cs=n_win,)
    
    start_time = timeit.default_timer()
    # train with early stopping on validation set
    print('... training')
    print>>f,'... training'
    best_perform = numpy.inf

    for e in range(n_epochs):

        print('epoch:%d->' % e)
        print>> f, ('epoch:->%d' % e)
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            rnn.train(x, y, n_win, learning_rate)

        pred_train = np.asarray([rnn.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in train_set_x ])
        pred_valid = np.asarray([rnn.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in valid_set_x ])
        pred_test = np.asarray([rnn.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in test_set_x ])


        #Mean Square Error
        res_train = np.mean(np.asarray((train_set_y[:, n_bit-1]-pred_train[:, n_bit-1])**2))
        res_valid = np.mean(np.asarray((valid_set_y[:, n_bit-1]-pred_valid[:, n_bit-1])**2))
        res_test = np.mean(np.asarray((test_set_y[:, n_bit-1] - pred_test[:, n_bit-1])**2))

        print('cost(mse): %f' % res_train)
        print>> f, ('cost(mse): %f' % res_train)
        
        if res_valid < best_perform:

            best_perform = res_valid

            if verbose:
                print('NEW BEST: epoch %i, valid error %f %%, best test error %f %%' %(e,res_valid*100.,res_test*100.))
                print>>f,('NEW BEST: epoch %i, valid error %f %%, best test error %f %%' %(e,res_valid*100.,res_test*100.))
            valid_error, test_error = res_valid, res_test
            best_epoch = e
        else:
            print('')
            print>>f,''

        # learning rate decay if no improvement in 10 epochs
        if abs(best_epoch-e) >= 10:
            learning_rate *= 0.5

        if learning_rate < 1e-5:
            break
    
    print('BEST RESULT: epoch %i, valid error %.4f %%, best test error %.4f %%'%( best_epoch,valid_error*100.,test_error*100.))
    print>>f,('BEST RESULT: epoch %i, valid error %.4f %%, best test error %.4f %%'%( best_epoch,valid_error*100.,test_error*100.))
    end_time = timeit.default_timer()
    print((' ran for %.2fm' % ((end_time - start_time) / 60.)))
    print>>f,((' ran for %.2fm' % ((end_time - start_time) / 60.)))
    f.close()
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(n_bit,fil):
    n_bit=n_bit
    n_hidden = 1
    n_epochs=1000
    #For 8 bit:
    #learning_rate=0.15
    #For 12 bit:
    learning_rate=0.5
    n_win=7
    verbose=True
    f=fil
    print('... loading the dataset')
    print>>f,'... loading the dataset'
    # generate datasets
    train_set_x,train_set_y = gen_parity_pair(n_bit, 1000)
    valid_set_x,valid_set_y = gen_parity_pair(n_bit, 500)
    test_set_x,test_set_y  = gen_parity_pair(n_bit, 100)

    numpy.random.seed(100)
    #We need additional labels
    
    train_set_y = create_additioinal_label(train_set_x)

    valid_set_y = create_additioinal_label(valid_set_x)

    test_set_y = create_additioinal_label(test_set_x)

    n_out = 2

    
    print('... building the model')
    print>>f,'... building the model'
    lstm = LSTM(
        nh=n_hidden,
        nc=n_out,
        cs=n_win,)
    
    start_time = timeit.default_timer()
    # train with early stopping on validation set
    print('... training')
    print>>f,'... training'
    best_perform = numpy.inf

    for e in range(n_epochs):
        
        print('epoch:%d->' %e)
        print>>f,('epoch:->%d' %e)
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            lstm.train(x, y, n_win, learning_rate)

        pred_train = np.asarray([lstm.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in train_set_x ])
        pred_valid = np.asarray([lstm.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in valid_set_x ])
        pred_test = np.asarray([lstm.classify(numpy.asarray( contextwin(x, n_win)).astype('int32')) for x in test_set_x ])


        #Mean Square Error
        res_train = np.mean(np.asarray((train_set_y[:, n_bit-1]-pred_train[:, n_bit-1])**2))
        res_valid = np.mean(np.asarray((valid_set_y[:, n_bit-1]-pred_valid[:, n_bit-1])**2))
        res_test = np.mean(np.asarray((test_set_y[:, n_bit-1] - pred_test[:, n_bit-1])**2))

        print('cost(mse): %f' %res_train)
        print>>f,('cost(mse): %f' %res_train)
        
        if res_valid < best_perform:

            best_perform = res_valid

            if verbose:
                print('NEW BEST: epoch %i, valid error %.4f %%, best test error %.4f %%' %(e,res_valid*100.,res_test*100.))
                print>>f,('NEW BEST: epoch %i, valid error %.4f %%, best test error %.4f %%' %(e,res_valid*100.,res_test*100.))
            valid_error, test_error = res_valid, res_test
            best_epoch = e
        else:
            print('')
            print>>f,''

        # learning rate decay if no improvement in 10 epochs
        if abs(best_epoch-e) >= 10:
            learning_rate *= 0.5

        if learning_rate < 1e-5:
            break
    
    print('BEST RESULT: epoch %i, valid error %.4f %%, best test error %.4f %%'%( best_epoch,valid_error*100.,test_error*100.))
    print>>f,('BEST RESULT: epoch %i, valid error %.4f %%, best test error %.4f %%'%( best_epoch,valid_error*100.,test_error*100.))
    end_time = timeit.default_timer()
    print((' ran for %.2fm' % ((end_time - start_time) / 60.)))
    print>>f,((' ran for %.2fm' % ((end_time - start_time) / 60.)))
    f.close()
    
if __name__ == '__main__':
    test_rnn_parity(n_bit=12,fil=open('./rnn_12bit.txt','w'))
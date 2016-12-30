import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg
from glob import glob

from PIL import Image
from numpy import array

from numpy import *
import sys
import theano
import theano.tensor as T
rng=np.random

theano.config.floatX = 'float32'
theano.config.exception_verbosity='high'
sys.setrecursionlimit(15000)

def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction
        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)
        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image
        ax: the axis on which the image will be plotted
    '''
    H, W = X_mn.shape
    re_im = np.dot(D, c).reshape(H, W) + X_mn
    plt.imshow(re_im, cmap=plt.gray())
def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs
    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
    im_num: Integer
        index of the image to visualize
    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plt.subplot(3, 3, i * 3 + j + 1)
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('./output2/2_im{0}.png'.format(im_num))
    plt.close(f)
    
def plot_top_16(D, sz, imname):
    out=D[:,:16]
    out=np.asarray(out)
    pic=[]
    for i in range(16):
        pic.append(out[:,i].reshape((sz,sz)))
    pic=np.asarray(pic)
  
    f,a=plt.subplots(4,4)
    for i in range(0,4):
        for j in range(0,4):
            plt.axes(a[i,j])
            plt.imshow(pic[4*i+j],cmap='Greys')

    f.savefig(imname.format(sz))
    
    plt.close(f)

def main():
    print("Loading data...")
    pic=[]
    for infile in sorted(glob('./Fei_256/*.jpg')):
        im=Image.open(infile)
        out = im.convert("L") 
        out=np.asarray(out)
        out=out.reshape((256*256,))
        pic.append(out)
    Ims=np.asarray(pic)
    
    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    print("Building model...")


    H = 256
    W = 256
    D = np.empty(shape=(H * W, 16), dtype='float32')# matrix D to store all d_i
    Lambda = np.empty(shape=(16,), dtype='float32')# cost function
    TS = 500# Do this many iterations at most
    Stop = 10 ** -8# Stop when d_i is smaller than this value

    # Declare Theano symbolic variables
    X_Theano = T.matrix("X_Theano")
    d_i = T.vector("d_i")
    L = T.vector("L")
    d = T.matrix("d")
    acc = T.scalar("acc")

    for i in range(0, 16):
        diff_c = 2
        diff_p = 0
        t = 0
        acc = 0  

        y = np.float32(np.random.rand(H * W))
        D[:, i] = (y / np.linalg.norm(y))

        if i > 0:
            for j in range(0, i):
                acc += L[j] * T.pow(T.dot(d_i.T, d[:, j]), 2)

        Opt = - T.dot(T.dot(X_Theano, d_i).T, T.dot(X_Theano, d_i)) + acc
        gd = T.grad(Opt, d_i)
        f = theano.function([X_Theano, d_i, L, d], gd, on_unused_input='ignore')

        while (t == 0) or (t <= TS and abs(diff_c - diff_p) >= Stop):

            diff_p = diff_c
            y = D[:, i] - 0.01 * f(X, D[:, i], Lambda, D)
            D[:, i] = y / np.linalg.norm(y)

            diff_c = np.dot(np.dot(X, D[:, i]).T, np.dot(X, D[:, i]))
            t += 1
            Lambda[i] = diff_c

        print(Lambda[i])

    c = np.dot(D.T, X.T)
    
    for i in range(0, 200, 10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])
    plot_top_16(D, 256, 'output2/top16_256.png')
    
if __name__ == '__main__':
    main()
    
    
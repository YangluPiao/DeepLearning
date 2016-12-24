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

import theano
import theano.tensor as T
rng=np.random

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    n = int(256)
    c_im = c[:num_coeffs, im_num* (im_num + 1)]
    D_im = D[:, :num_coeffs]
    co = np.dot(D_im, c_im)

    X_tmp = []
    for i in range(0, 1):
        tp = co[i]
        tp = tp + X_mean
        tp = np.asarray(tp)
        X_tmp.append(tp)
    X_tmp = np.asarray(X_tmp)

    row = []
    big_row = []
    matrix = []

    for i in range(0, 1):
        for k in range(0, n ** 2, n):
            for j in range(0, 1):
                row.extend(X_tmp[j + i][k:k + n])
            row = np.asarray(row)
            big_row.append(row)
            row = []
        if i == 0:
            matrix = big_row
            big_row = []
        else:
            big_row = np.asarray(big_row)
            matrix = np.vstack([matrix, big_row])
            big_row = []
    matrix = np.asarray(matrix)
    X_recon_img = Image.fromarray(matrix)
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    f, axarr = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i, j])
            plt.imshow(reconstructed_image(D, c, num_coeff_array[i * 3 + j], X_mean, im_num))

    f.savefig('./output2/2_{0}_im{1}.png'.format(1, im_num))
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
        acc = 0  # Needs to be outside as A_i is before the while loop, Theano just changes d_i everytime

        y = np.float32(np.random.rand(H * W))
        D[:, i] = (y / np.linalg.norm(y))

        # Construct Theano function
        if i > 0:
            for j in range(0, i):
                acc += L[j] * T.pow(T.dot(d_i.T, d[:, j]), 2)

        Opt = - T.dot(T.dot(X_Theano, d_i).T, T.dot(X_Theano, d_i)) + acc
        gd = T.grad(Opt, d_i)
        f = theano.function([X_Theano, d_i, L, d], gd, on_unused_input='ignore')

        while (t == 0) or (t <= TS and abs(diff_c - diff_p) >= Stop):

            # Theano
            diff_p = diff_c
            y = D[:, i] - 0.01 * f(X, D[:, i], Lambda, D)
            D[:, i] = y / np.linalg.norm(y)
            # print D[:, i].shape
            diff_c = np.dot(np.dot(X, D[:, i]).T, np.dot(X, D[:, i]))
            t += 1
            # if abs(diff_c - diff_p) < 500:
            #     print diff_c, diff_p, t
            Lambda[i] = diff_c

        print(Lambda[i])

    c = np.dot(D.T, X.T)

    for i in range(0, 200, 10):
        plot_reconstructions(D, c,[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mn.reshape((256, 256)),i)

    plot_top_16(D, 256, 'output2/top16_256.png')
if __name__ == '__main__':
    main()
    
    
import os
from glob import glob
import re
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy.linalg as linalg



from PIL import Image
from numpy import array

# Reconstruct images.
def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
        Plots nine PCA reconstructions of a particular image using number
        of components specified by num_coeffs
        Parameters
        ---------------
        c: np.ndarray
            a n x m matrix  representing the coefficients of all the image blocks.
            n represents the maximum dimension of the PCA space.
            m is (number of images x n_blocks**2)
        D: np.ndarray
            an N x n matrix representing the basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)
        im_num: Integer
            index of the image to visualize
        X_mn: np.ndarray
            a matrix representing the mean block.
        num_coeffs: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    n = int(256/n_blocks)
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    co=np.dot(D_im,c_im)

    X_tmp=[]
    for i in range(0,n_blocks**2):
        tp=co[:,i]
        tp=tp+X_mean
        tp=np.asarray(tp)
        X_tmp.append(tp)
    X_tmp=np.asarray(X_tmp)
    
    row=[]
    big_row=[]
    matrix=[]
    
    for i in range (0,n_blocks):
        for k in range(0,n**2,n):
            for j in range(0,n_blocks):
                row.extend(X_tmp[j+n_blocks*i][k:k+n])
            row=np.asarray(row)
            big_row.append(row)
            row=[]
        if i == 0:
            matrix=big_row
            big_row=[]
        else:
            big_row=np.asarray(big_row)
            matrix=np.vstack([matrix,big_row])
            big_row=[]
    matrix=np.asarray(matrix)
    X_recon_img=Image.fromarray(matrix)
    return X_recon_img

# Plot reconstructed images.
def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
        Plots the top 16 components from the basis matrix D.
        Each basis vector represents an image block of shape (sz, sz)
        Parameters
        -------------
        D: np.ndarray
            N x n matrix representing the basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)
            n represents the maximum dimension of the PCA space (assumed to be atleast 16)
        sz: Integer
            The height and width of each block
        imname: string
            name of file where image will be saved.
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num))
            
    f.savefig('./output1/1_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)

#Plot 16 images once.
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
    szs = [8,32,64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        pic=[]
        for infile in sorted(glob('./Fei_256/*.jpg')):
            im=Image.open(infile)
            out = im.convert("L")    
            pic.append(array(out))
        IMs=np.asarray(pic)
        X=[]
        X_r=[]
        X_m=[]
        

        n_blocks=int(256/sz)
        for infile in IMs:
            for i in range(0,256,sz):
                for j in range(0,256,sz):
                    X_r.append(infile[i:i+sz,j:j+sz])
                    X_m.append(X_r)
                    X_r=[]
            X_m=np.asarray(X_m).reshape((n_blocks**2,sz**2))
            X.append(X_m)
            X_m=[]
            
                    
        X=np.asarray(X).reshape((200*n_blocks**2,sz**2))
        
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
        [value,D]=linalg.eigh(np.dot(X.T,X))# Normalize X for calculating eigenvalue and eigenvector.

        idx=np.argsort(value)# Returns the indices that would sort an array.
        idx=idx[::-1]# Reverse the sorted array.
        D=D[:,idx]
        value=value[idx]
                
        D=np.asarray(D)
            
        c = np.dot(D.T, X.T)
        
        print("Finished calculating c and D")
        for i in range(0, 200, 10):
            print("im_num is :",i)
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean, n_blocks=int(256/sz), im_num=i)
       
        plot_top_16(D, sz, imname='./output1/1_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
    
    
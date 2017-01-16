from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from hw3_utils import load_data
def translate(pic):
    pix=np.random.random()*8-4
    pic_trans=np.roll(pic,int(pix),axis=0)
    pic_trans=np.roll(pic_trans,int(pix),axis=1)
    return pic_trans

def rotation(pic):
    angle=np.random.random()*8-4
    im = Image.fromarray((np.uint8(pic*255)))
    return im.rotate(int(angle))

def flip(pic):
    im = Image.fromarray((np.uint8(pic*255)))
    return im.transpose(Image.FLIP_LEFT_RIGHT)

def add_noise(pic,typ='Gaussian'):
    if typ == 'Gaussian':
        pic = pic + np.reshape(np.random.normal(0,0.1,pic.size),(3,32,32)).transpose(1,2,0)
    elif typ == 'Uniform':
        pic = pic + np.reshape(np.random.uniform(-0.1,0.1,pic.size),(3,32,32)).transpose(1,2,0)
    return pic

def unflatten(array):
    return np.reshape(np.asarray(array),(3,32,32)).transpose(1,2,0)

def flatten(pic):
    return np.asarray(pic).transpose(2,0,1).ravel()

def Dropout(pic):
    index=np.random.random(int(3072*0.3))*3072
    index=index.astype(int)
    for i in index:
        pic[i]=0
    return pic

def showpic(pb,dataset,typ='Gaussian'):
    f,a=plt.subplots(4,4)
    for i in range(0,4):
        for j in range(0,4):
            pic_modified = unflatten(dataset[4*i+j])
            if pb == 2.1:
                pp = translate(pic_modified)
            elif pb == 2.2:
                pp = rotation(pic_modified)
            elif pb == 2.3:
                pp = flip(pic_modified)
            elif pb == 2.4:
                pp = add_noise(pic_modified,typ)
            elif pb == 4.1:
                pp = Dropout(flatten(pic_modified))
                pp = unflatten(pp)
            else :
                pp = pic_modified
            plt.axes(a[i,j])
            plt.imshow(pp)
    if pb < 3:
        f.savefig('./problem2/hw3_pb{0}.png'.format(pb))
        plt.close(f)
    elif pb == 3:
        f.savefig('./problem3/hw3_pb{0}.png'.format(pb))
        plt.close(f)
    elif pb == 4:
        f.savefig('./problem4/hw3_pb{0}_groud_truth.png'.format(pb))
        plt.close(f)
    elif pb == 4.1:
        f.savefig('./problem4/hw3_pb{0}_corrupted.png'.format(pb))
        plt.close(f)
    elif pb == 4.2:
        f.savefig('./problem4/hw3_pb{0}_restored.png'.format(pb))
        plt.close(f)
'''        
datasets=load_data(pb=4,theano_shared=False)
x,y=datasets[2]
showpic(pb=4,dataset=x)
'''
        

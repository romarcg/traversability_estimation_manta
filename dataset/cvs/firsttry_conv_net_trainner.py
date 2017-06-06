#%% 1. Simple example: full pipeline on an image

import matplotlib.pyplot as plt
import pathlib
import skimage
import skimage.transform
import skimage.viewer
import pandas as pd
import numpy as np
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import time
patchsize=32



ddir=pathlib.Path(".")

cdirs={0:ddir/"c0",
       1:ddir/"c1",
       2:ddir/"c2"}


#%% How does one image look?
im = skimage.io.imread(list(cdirs[0].glob("*"))[0])
viewer=skimage.viewer.ImageViewer(im)
viewer.show()

#%% Load images

imagesize=500

dataset=[]
for cn,cdir in cdirs.items():
    for f in cdir.glob("*"):
        im=skimage.io.imread(f)
        h,w=im.shape[0:2]
        sz=min(im.shape[0:2])
        im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:]
        im=skimage.transform.resize(im,(imagesize,imagesize))
        plt.imshow(im)
        plt.show()
        dataset.append({"label":cn,
                        "image":im})

dataset=pd.DataFrame(dataset)

dataset_te=dataset.sample(50)
dataset_tr=dataset.loc[dataset.index.difference(dataset_te.index)]


#%% This is the set of our training images
viewer=skimage.viewer.CollectionViewer(list(dataset_tr["image"]))
viewer.show()


#%% This is the set of our testing images
viewer=skimage.viewer.CollectionViewer(list(dataset_te["image"]))
viewer.show()



#%% Define our neural network architecture
model = Sequential()
model.add(Convolution2D(5, 3, 3,
                        border_mode='valid',
                        input_shape=(patchsize,patchsize,3)))
model.add(Activation('relu'))
model.add(Convolution2D(5, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(5, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer="adadelta",
              metrics=['accuracy'])

#%% Define how to build the datasets

if(False):
    def transform(im,sz):
        imr = skimage.transform.resize(im, (sz,sz))
        return imr
else:
    def transform(im,sz):
        tf1 = skimage.transform.SimilarityTransform(scale=1/im.shape[0])
        tf2 = skimage.transform.SimilarityTransform(translation=[-0.5, -0.5])
        tf3 = skimage.transform.SimilarityTransform(rotation=np.deg2rad(np.random.rand()*360))
        tf4 = skimage.transform.SimilarityTransform(scale=2**0.5+np.random.rand()*0.2)
        tf5 = skimage.transform.SimilarityTransform(translation=[0.5, 0.5]+(np.random.rand((2))*0.2-0.1))
        tf6 = skimage.transform.SimilarityTransform(scale=patchsize)
        imr = skimage.transform.warp(im, (tf1+(tf2+(tf3+(tf4+(tf5+tf6))))).inverse,output_shape=(sz,sz),mode="edge")
        return imr

def sample(df,sz):
    r=df.sample(n=1)
    l=r["label"].iloc[0]
    im=r["image"].iloc[0]
    im=transform(im,sz)
    return im,l

def mktrte(df,N,sz):
    X = []
    y = []
    for i in range(N):
        im,l=sample(df,sz)
        X.append(im)
        y.append(l)
    X=np.array(X).astype('float32')
    y=np.array(y)
    y=np_utils.to_categorical(y,3)
    return X,y


def generator(df,batch_size,sz):
    while True:
        X,y = mktrte(df,batch_size,sz)
        yield (X,y)        
        
#%% Visualize many variations for a single training image
viewer=skimage.viewer.CollectionViewer(mktrte(dataset_tr.iloc[[0]],100,patchsize)[0])
viewer.show()

#%% Visualize the training set
viewer=skimage.viewer.CollectionViewer(mktrte(dataset_tr,100,patchsize)[0])
viewer.show()

#%%  Train the neural network
(X_te,y_te)=mktrte(dataset_te,1000,patchsize)
batch_size = 100
history=model.fit_generator(
                    generator(dataset_tr,batch_size,patchsize),
                    samples_per_epoch=1000, 
                    nb_epoch=500, 
                    verbose=1,
                    validation_data=(X_te,y_te),
                    callbacks=[keras.callbacks.TensorBoard(log_dir='./logs/'+time.strftime("%Y%m%d%H%M%S"), histogram_freq=0, write_graph=False, write_images=False)])

#%%

final=[]
for f in (ddir/"final").glob("*"):
    im=skimage.io.imread(f)
    h,w=im.shape[0:2]
    sz=min(im.shape[0:2])
    im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:]
    im=skimage.transform.resize(im,(imagesize,imagesize))
    plt.imshow(im)
    plt.show()
    final.append(im)

#%%
for im in dataset_te["image"]:
    fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    axs[0].imshow(im)
    imt=transform(im,patchsize);
    axs[1].imshow(imt)
    outs=model.predict(np.array([imt]))
    axs[2].bar(left=range(3),height=outs[0])
    axs[2].set_ylim([0,1])
    print(outs)

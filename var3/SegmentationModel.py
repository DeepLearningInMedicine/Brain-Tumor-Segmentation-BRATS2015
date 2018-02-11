import numpy as np
import random
import json
import h5py
from patch_library import PatchLibrary
from glob import glob
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from skimage.segmentation import mark_boundaries
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import classification_report
from keras.models import Sequential, model_from_json, Model
from keras.layers import concatenate, Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.initializers import constant
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.legacy.layers import MaxoutDense

class SegmentationModel(object):
    
    def __init__(self, n_epoch = 10, n_chan = 4, batch_size = 16, loaded_model = False, architecture = 'single', w_reg = 0.01, n_filters = [64, 128, 128, 128], k_dims = [7, 5, 5, 3], activation = 'relu'):
        self.n_epoch = n_epoch
        self.n_chan = n_chan
        self.batch_size = batch_size
        self.architecture = architecture
        self.loaded_model = loaded_model
        self.w_reg = w_reg
        self.n_filters = n_filters
        self.k_dims = k_dims
        self.activation = activation
        #self.model_comp = self.compile_model()
        self.model_comp = self.build_Pereira(33, 33, 4, 5)
        #self.model_comp = self.compile_model_two()
        #self.model_comp = self.u_net()
        
    def compile_model_two(self, classes=5, alp = 0.333):
        
        patch = Input(shape = (4, 33, 33))
        
        conv1 = Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (4, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(patch)
        conv1act = LeakyReLU(alpha=alp)(conv1)
        
        conv2 = Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(conv1act)
        conv2act = LeakyReLU(alpha=alp)(conv2)
        
        conv3 = Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(conv2act)
        conv3act = LeakyReLU(alpha=alp)(conv3)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv3act)
        
        conv4 = Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(pool1)
        conv4act = LeakyReLU(alpha=alp)(conv4)
        conv5 = Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(conv4act)
        conv5act = LeakyReLU(alpha=alp)(conv5)
        conv6 = Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) )(conv5act)
        conv6act = LeakyReLU(alpha=alp)(conv6)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv6act)
        
        flat = Flatten()(pool2)
        drflat = Dropout(0.1)(flat)
        dense1 = Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1))(drflat)
        dense1act = LeakyReLU(alp)(dense1)
        drdense1 = Dropout(0.1)(dense1act)
        dense2 =  Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1))(drdense1)
        drdense2 = Dropout(0.1)(dense2)
        
        small_patch = Input(shape = (4, 5, 5))
        flat_spatch = Flatten()(small_patch)
        one_more = MaxoutDense(128, nb_feature=5)(flat_spatch)
        dr_one_more = Dropout(0.1)(one_more)
        
        final_input =  concatenate([drdense2, dr_one_more])
        dense3 = Dense(classes, kernel_initializer = 'glorot_normal', bias_initializer = constant(0.1))(final_input)
        predictions = Activation('softmax')(dense3)
        
        model = Model(inputs = [patch, small_patch], outputs = predictions)
        model.compile(loss='categorical_crossentropy',metrics =['accuracy'], optimizer='adam')
        return model
    
    
    def u_net(self):
        K.set_image_dim_ordering('th')
        inputs = Input(shape = (4, 208, 208))
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 1)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        
        
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 1)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 1)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 1)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(5, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv11 = Conv2D(5, 1, activation = 'sigmoid')(conv9)
        
        model = Model(input = inputs, output = conv11)
        model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model



        
    
    def compile_model(self):
        
        single = Sequential()
        
        single.add(Convolution2D(self.n_filters[0], (self.k_dims[0], self.k_dims[0]), border_mode='valid', W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg), input_shape=(self.n_chan,33,33), dim_ordering='th'))
        single.add(Activation(self.activation))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(self.n_filters[1], (self.k_dims[1], self.k_dims[1]), activation=self.activation, border_mode='valid', W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(self.n_filters[2], (self.k_dims[2], self.k_dims[2]), activation=self.activation, border_mode='valid', W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(BatchNormalization(mode=0, axis=1))
        single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
        single.add(Dropout(0.5))
        single.add(Convolution2D(self.n_filters[3], (self.k_dims[3], self.k_dims[3]), activation=self.activation, border_mode='valid', W_regularizer=l1_l2(l1=self.w_reg, l2=self.w_reg)))
        single.add(Dropout(0.25))

        single.add(Flatten())
        single.add(Dense(5))
        single.add(Activation('softmax'))

        adam = Adam()
        single.compile(loss='categorical_crossentropy',metrics =['accuracy'], optimizer='adam')
        print ('Done.')
        return single
    
    def build_Pereira(self, w, h, d, classes, weightsPath = None, alp = 0.333, dropout = 0.1):
        '''INPUT:
                INPUT WIDTH, HEIGHT, DEPTH, NUMBER OF OUTPUT CLASSES, PRELOADED WEIGHTS, PARAMETER FOR LEAKYReLU, DROPOUT PROBABILITY
           OUTPUT:
                TRAINED CNN ARCHITECTURE
                '''
        K.set_image_dim_ordering('th')
        model = Sequential()
        
                
        #first set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (d, h, w), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 33, 33), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #second set of CONV => CONV => CONV => LReLU => MAXPOOL
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (64, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha=alp))
        model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", data_format='channels_first', input_shape = (128, 16, 16), kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1) ))
        model.add(LeakyReLU(alpha = alp))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        #Fully connected layers
        
        # FC => LReLU => FC => LReLU
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        model.add(Dense(256, kernel_initializer = 'glorot_normal', bias_initializer=constant(0.1)))
        model.add(LeakyReLU(alp))
        model.add(Dropout(0.1))
        
        
        
        
        # FC => SOFTMAX
        model.add(Dense(classes, kernel_initializer = 'glorot_normal', bias_initializer = constant(0.1)))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',metrics =['accuracy'], optimizer='adam')
        
        
        return model
    

    
    def fit_model(self, X_train, X5, y_train, X5_train = None, save = True):
        
        Y_train = np_utils.to_categorical(y_train, 5)

        shuffle = zip(X_train, Y_train)
        np.random.shuffle(shuffle)
        X_train = np.array([shuffle[i][0] for i in xrange(len(shuffle))])
        #X5 = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        Y_train = np.array([shuffle[i][1] for i in xrange(len(shuffle))])
        
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        
        
        
        checkpointer = ModelCheckpoint(filepath="npy/bm_{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)
        
        #class_weight = {0 : 1.,1 : 20., 2 : 6, 3 : 15., 4 : 20.} 
        self.model_comp.fit(X_train, Y_train, batch_size=self.batch_size, epochs=15, validation_split=0.1, verbose=1, callbacks=[checkpointer])
        
        return None
    
    
    


       
        
    
        
        
                          
        
        
        
 
    
    
   
    
    
    


    
if __name__ == '__main__':
    
    #train_data = glob('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/Norm_PNG/**')
    #print(train_data)
    
    #patches = PatchLibrary((33,33), train_data, 20000)
    X = np.load('data/X.npy')
    X5 = np.load('data/x.npy')
    y = np.load('data/y.npy')
    #model = load_model('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/bm_04-0.94.hdf5')
    
    #predict
                                                      

    model = SegmentationModel()
    model.fit_model(X, X5, y)
    #model.save_model('media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training')
    

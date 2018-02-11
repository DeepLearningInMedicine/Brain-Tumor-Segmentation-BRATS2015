import numpy as np
import random
import os
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import progressbar
from sklearn.feature_extraction.image import extract_patches_2d

progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])

np.random.seed(5)


class PatchLibrary(object):
    def __init__(self, patch_size, train_data, num_samples):
        self.patch_size = patch_size
        self.train_data = train_data
        self.num_samples = num_samples
        self.h = self.patch_size[0]
        self.w = self.patch_size[1]
        
    def find_patches_unet(self):
        h, w = self.patch_size[0], self.patch_size[1]
        
        per_class = self.num_samples/ 5
        patches, labels = [], []
        
        for i in xrange(5):
            print 'Finding patches of class {}...'.format(i)
            
            ct = 0
            while ct<per_class:
                im_path = random.choice(self.train_data)
                fn = os.path.basename(im_path)
                label = io.imread('npy/Labels/' + fn[:-4] + 'L.png')
            #print("Reached here")
                while len(np.argwhere(label == i)) < 10:
                #print("Stuck here")
                    im_path = random.choice(self.train_data)
                    fn = os.path.basename(im_path)
                    label = io.imread('npy/Labels/' + fn[:-4] + 'L.png')
                
                img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
                
                img = img[:, 16:-16, 16:-16]
                label = label[16:-16, 16:-16]
                    
                patch_gt = []
                for j in xrange(5):
                    temp_label = (label ==j)
                    patch_gt.append(temp_label)
                patch_gt = np.array(patch_gt)
                
                
                
                patches.append(img)
                labels.append(patch_gt)
                ct = ct+1
                
                print (float(ct)/per_class) *100
        
        
        return np.array(patches), np.array(labels)
                    
                    
                    
                    
            #small_patch = np.array([i[p_six[0]:p_six[1], p_six[2]:p_six[3]] for i in img])
            #print("This too clear")
            
            
           
                
            
            
    def find_patches(self, class_num, num_patches):
        h, w = self.patch_size[0], self.patch_size[1]
        patches, labels = [], np.full(num_patches, class_num, 'float')
        small_patches = []
        
        print 'Finding patches of class {}...'.format(class_num)
       
        
        
        progress_another = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])
        ct = 0
        while ct<num_patches:
            im_path = random.choice(self.train_data)
            fn = os.path.basename(im_path)
            label = io.imread('npy/Labels/' + fn[:-4] + 'L.png')
            #print("Reached here")
            while len(np.argwhere(label == class_num)) < 10:
                #print("Stuck here")
                im_path = random.choice(self.train_data)
                fn = os.path.basename(im_path)
                label = io.imread('npy/Labels/' + fn[:-4] + 'L.png')
                
            img = io.imread(im_path).reshape(5, 240, 240)[:-1].astype('float')
            p = random.choice(np.argwhere(label == class_num))
            p_ix = (p[0]-(h/2), p[0]+((h+1)/2), p[1]-(w/2), p[1]+((w+1)/2))
            p_six = (p[0]-(5/2), p[0]+((5+1)/2), p[1]-(5/2), p[1]+((5+1)/2))
            #print("This clear")
            patch = np.array([i[p_ix[0]:p_ix[1], p_ix[2]:p_ix[3]] for i in img])
            small_patch = np.array([i[p_six[0]:p_six[1], p_six[2]:p_six[3]] for i in img])
            #print("This too clear")
            
            if patch.shape != (4, h, w) or len(np.unique(patch)) == 1: #So that patch is not entirely background
                continue
                
            patches.append(patch)
            small_patches.append(small_patch)
            ct = ct+1
            print (float(ct)/num_patches) *100
        p = patches
            
        for img_ix in xrange(len(p)):
            for slice in xrange(len(p[img_ix])):
                if np.max(p[img_ix][slice]) != 0:
                    p[img_ix][slice] /= np.max(p[img_ix][slice])
              
            
            
            
        return np.array(p),np.array(small_patches), labels

            
                
            
        
        
    def make_training_patches(self, entropy = False, balanced_classes = True, classes = [0, 1, 2, 3, 4]):
        if balanced_classes:
            per_class = self.num_samples/ len(classes)
            #per_class = [
            patches,s_patches, labels, labels_Unet = [],[], [], []
            progress.currval = 0
            
            for i in progress(xrange(len(classes))):
                p,q,l = self.find_patches(classes[i], per_class)
                
                for img_ix in xrange(len(p)):
                    for slice in xrange(len(p[img_ix])):
                        if np.max(p[img_ix][slice]) != 0:
                            p[img_ix][slice] /= np.max(p[img_ix][slice])
                        if np.max(q[img_ix][slice]) != 0:
                            q[img_ix][slice] /= np.max(q[img_ix][slice])
                        
                patches.append(p)
                s_patches.append(q)
                
                print(len(patches))
                labels.append(l)
                
            print(patches[0].shape)
            return np.array(patches).reshape(self.num_samples, 4, self.h, self.w), np.array(s_patches).reshape(self.num_samples, 4, 5, 5), np.array(labels).reshape(self.num_samples)
        else:
            print "Use balanced classes, random won't work."

if __name__ == '__main__':
    train_data = glob('npy/Norm_PNG/**')
    patches = PatchLibrary((32,32), train_data, 1000)
    #X,x,y = patches.make_training_patches()
    X, y = patches.find_patches_unet()
    y = y.astype(int)
    
    #X_o, x_o, y_o = patches.make_training_patches()
    np.save('npy/X_unet_full.npy', X)
    #np.save('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/x.npy', x_o)
    #np.save('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/X.npy', X_o)
    np.save('npy/y_unet_full.npy', y)
    #np.save('/media/hrituraj/New Volume/BRATS2015_Training/BRATS2015_Training/y.npy', y_o)
    #train_data = glob('/home/hrituraj/BRATS2015_Training/BRATS2015_Training/Norm_PNG/**')
    #Let_us_make_patches = PatchLibrary((33,33),train_data, 10000 )              
    #Let_us_make_patches.make_training_images()

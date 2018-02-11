import numpy as np
import subprocess
import random
import progressbar
from glob import glob
from skimage import io
import SimpleITK as sitk
import os


np.random.seed(5)#for reproducability

progress = progressbar.ProgressBar(widgets = [progressbar.Bar('*','[',']'), progressbar.Percentage(), ''])

class BrainPipeline(object):
    
    def __init__(self, path):
        self.path = path
        self.modes = ['flair','t1', 't1c', 't2', 'gt']
        self.slices_by_mode, n = self.read_scans()
        self.slices_by_slice = n
        self.normed_slices = self.n4_normalize()
        self.normed_slices = self.norm_slices()
        
        
        
    def read_scans(self):
        print ("Loading Scans...")
        
        slices_by_mode = np.zeros([5, 155, 240, 240])
        slices_by_slice = np.zeros([155, 5, 240, 240])
        
        flair = glob(self.path + '/*Flair*/*.mha')
        t2 = glob(self.path + '/*_T2*/*.mha')
        gt = glob(self.path + '/*more*/*.mha')
        t1s = glob(self.path + '/*T1*/*.mha')
        t1 = [scan for scan in t1s]

        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]
        
        for scan_idx in range(5):
            slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
            
        for mode_idx in range(slices_by_mode.shape[0]):
            for slice_idx in range(slices_by_mode.shape[1]):
                slices_by_slice[slice_idx][mode_idx] = slices_by_mode[mode_idx][slice_idx]
        return slices_by_mode, slices_by_slice
        
    def n4_normalize(self):
        print ('n4bias correction slices...')
        normed_slices = np.zeros((155, 5, 240, 240))
        
        for slice_ix in range(155):
            
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in range(4):
                if(mode_ix==1 or mode_ix==2):
                    normed_slices[slice_ix][mode_ix] =  self._n4_on_image(self.slices_by_slice[slice_ix][mode_ix])
                else:
                    normed_slices[slice_ix][mode_ix] = self.slices_by_slice[slice_ix][mode_ix]
                    
                
        print ('Done.')
        return normed_slices
    
    def _n4_on_image(self, image):
        # print(type(img))
        img = sitk.GetImageFromArray(image)
        img = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
        corrected_img = sitk.GetArrayFromImage(corrected_img)
        return corrected_img
            
    def norm_slices(self):
        print ("Normalizing slices....")
        normed_slices = np.zeros([155, 5, 240, 240])
            
        for slice_idx in range(155):
            normed_slices[slice_idx][-1] = self.normed_slices[slice_idx][-1]
                
            for mode_idx in range(4):
                normed_slices[slice_idx][mode_idx] = self.normalize(self.normed_slices[slice_idx][mode_idx])
                
        print ('Done')
        return normed_slices
        
        
    def normalize(self, slice):
        #b,t = np.percentile(slice, (0.5,0.95))
        #slice = np.clip(slice, b, t)
            
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)
        
        
    def save_patient(self, reg_norm_n4, patient_num):
            
        print ('Saving scans for patient {}'.format(patient_num))
            
        progress.currval = 0
            
        if reg_norm_n4 == 'norm':
            for slice_ix in range(155):
            
                strip = self.normed_slices[slice_ix].reshape(1200, 240)
          
                if np.max(strip)!=0:
                    strip /= np.max(strip)
                if np.min(strip) <=-1:
                    strip /= abs(np.min(strip))
                
                
                
                    
                
                io.imsave('npy/Norm_PNG/{}_{}.png'.format(patient_num,
                          slice_ix), strip)
                    
        elif reg_norm_n4 == 'reg':
            for slice_ix in progress(range(155)):
                strip = self.slices_by_slice[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                io.imsave('npy/Training_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
        else:
            for slice_ix in progress(range(155)):
                strip = self.normed_slices[slice_ix].reshape(1200, 240)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                if np.min(strip) <= -1:
                    strip /= abs(np.min(strip))
                
                io.imsave('npy/n4_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)

                    
def save_patient_slices(patients, types):
    for patient_num, path in enumerate(patients):
        
        if(patient_num >18):
        
            a = BrainPipeline(path)
            a.save_patient(types, patient_num)
            print ("Number of Patients scanned {}/{}".format(patient_num, len(patients)))
    
                        
def save_labels(labels):
    '''
    INPUT list 'fns': filepaths to all labels
    '''
    print(len(labels))
    progress.currval = 0
    for label_idx in progress(range(len(labels))):
        slices = io.imread(labels[label_idx], plugin = 'simpleitk')
        for slice_idx in range(len(slices)):
            io.imsave('npy/Labels/{}_{}L.png'.format(label_idx, slice_idx), slices[slice_idx])
  
            
            
if __name__ == '__main__':
    #labels = glob('/media/hrituraj/New Volume1/BRATS2015_Training/BRATS2015_Training/HGG/**/*more*/**.mha')
    #save_labels(labels)
    patients = glob('/mnt/hgfs/Medical-Data/BRATS-2015/BRATS2015_Training/HGG/**')
    save_patient_slices(patients, 'norm')
    
   
    


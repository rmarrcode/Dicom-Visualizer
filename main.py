import numpy as np 
import os
import copy
from math import *
import matplotlib.pyplot as plt
from functools import reduce
import pydicom
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
from scipy.linalg import norm
import scipy.ndimage
from ipywidgets.widgets import * 
import ipywidgets as widgets
import plotly
from plotly.graph_objs import *
import chart_studio.plotly as py
from GK import * 

#takes a pathname opens dicom files 
def load_scan(path):
    #get all slices as pydicom array
    slices = [pydicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    #finding slice_thickness from all slices
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -   
                          slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation) 
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
    
#pixels -> hu
def get_pixels_hu(scans):
    #getting pixel values
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)    
    image[image == -2000] = 0
    #linear conversion pixels
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)    
    counts = counts[vals != bg]
    vals = vals[vals != bg]    
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

#produces a binary mask for a lung
#change constant
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8)+1
    labels = measure.label(binary_image)
 
    # Pick the pixel in the very corner to determine which label is air
    # Improvement: Pick multiple background labels from around the patient 
    # More resistant to “trays” on which the patient lays cutting the air around the person in half 
    background_label = labels[0,0,0]
 
    # Fill the air around the person
    # air becomes 2 ?
    binary_image[background_label == labels] = 2

 
    # Method of filling the lung structures (that is superior to 
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
 
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def main():
    path =  '/home/ryan/Desktop/DCM/chest'
    patient_dicom = load_scan(path)
    patient_pixels = get_pixels_hu(patient_dicom)

    #plt.imshow(patient_pixels[326], cmap=plt.cm.bone)
    #plt.show()

    segmented_lungs = segment_lung_mask(patient_pixels, fill_lung_structures=False)    
    segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)
        
    internal_structures = segmented_lungs_fill - segmented_lungs
    copied_pixels = copy.deepcopy(patient_pixels)
    for i, mask in enumerate(segmented_lungs_fill): 
        #find where not lungs are found
        get_high_vals = mask == 0
        copied_pixels[i][get_high_vals] = 0
    seg_lung_pixels = copied_pixels

    #plt.imshow(seg_lung_pixels[0], cmap=plt.cm.bone)
    #plt.imshow(seg_lung_pixels[15], cmap=plt.cm.bone)
    #plt.show()

    selected_slices = seg_lung_pixels[0]
    gk_clustered_imgs = np.array([gk_segment(x) for x in selected_slices])   
    plt.imshow(gk_clustered_imgs[0], cmap=plt.cm.bone)

if __name__ == "__main__":
    main()
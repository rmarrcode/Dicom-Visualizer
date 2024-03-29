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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import numpy

#takes a pathname opens dicom files 
def load_scan(path):
    #get all slices as pydicom array
    slices = [pydicom.dcmread(path + '/' + s, force = True) for s in               
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

def plot_3d(image):
    p = image.transpose(2,1,0)
    verts, faces, _, _ = measure.marching_cubes_lewiner(p)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=.7)
    face_color = [0.75, 0.45, 0.45]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.show()

#produces a binary mask for a lung
#change constant
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    # -700
    binary_image = np.array(image >= -700, dtype=np.int8)+1 
    labels = measure.label(binary_image)
 
    # Pick the pixel in the very corner to determine which label is air
    background_label = labels[0,0,0]
 
    # Fill the air around the person
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

#sort elements based on name in directory 
def mask_arr(fname):
    onlyfiles = [f for f in os.listdir(fname)]
    ans = np.ndarray((len(onlyfiles), 512, 512))
    i = 0
    for img in (sorted(onlyfiles)):
        im = Image.open(fname+img)
        imarray = numpy.array(im)
        ans[i] = imarray
        i = i+1
    return ans

def show_all(slices):
    count = 0
    for slice in slices:
        plt.figure(count)
        plt.imshow(slice)
        count = count + 1
    plt.show()

def orMask (a,b):
    l = len(a)
    ans = np.ndarray((l, 512, 512))
    for i in range(l):
        for j in range(512):
            for k in range(512):
                ans[i][j][k] = (a[i][j][k] ^ b[i][j][k])
    return ans


def main():
    path =  'Lung_Scan'

    patient_dicom = load_scan(path)
    patient_pixels = get_pixels_hu(patient_dicom)
    segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)
    
    plot_3d(segmented_lungs_fill)

if __name__ == "__main__":
    main()

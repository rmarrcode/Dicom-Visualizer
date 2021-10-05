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

def main():
    path = '/home/ryan/Desktop/A'
    patient_dicom = load_scan(path)
    patient_pixels = get_pixels_hu(patient_dicom)

    plt.figure(1)
    plt.imshow(patient_pixels[1], cmap=plt.cm.bone)
    plt.imshow(patient_pixels[10], cmap=plt.cm.bone)
    plt.show()

    
if __name__ == "__main__":
    main()
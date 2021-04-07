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


def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -   
                          slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation) 
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
    
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


path =  '/home/ryan/Desktop/DCM/chest'
patient_dicom = load_scan(path)
patient_pixels = get_pixels_hu(patient_dicom)

plt.imshow(patient_pixels[326], cmap=plt.cm.bone)
plt.show()

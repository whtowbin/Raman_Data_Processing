#%%
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import numpy as np
#%%

calibrate = io.imread("/Users/wtowbin/Projects/hyper-raman/Dev Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")

# List of image Transformations to apply to full dataset e.g., median filter, normalization, cropping, etc.
image_transforms = []

# %%
plt.plot(median_calibrate.argmax(axis=0), marker = ".", linestyle = 'none')
plt.ylim(219,226)

# %%
fig, ax = plt.subplots()
ax.imshow(median_calibrate)
plt.ylim(200, 250)


# %%
"""
Notes on how to process data:

- Median Filter to help identify outliers

- Normalize each image against average Laser Power to help with power fluctuations
- Need a nornalization method for across the laser line in each image, Looks like the line shape changes with laser power, see if it chagnes with depth. 

"""
def fit_gaussian(image):
    # Fit 1D Gaussian array along each column in the image
    gaussian_fits = []

def calibrate_raman(image):


# %%

#%%
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
#%%

calibrate = io.imread("/Users/wtowbin/Projects/hyper-raman/Dev_Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")

# Running List of image Transformations to apply to full dataset e.g., median filter, normalization, cropping, etc.
image_transforms = []

#%%
median_calibrate = skfilter.median(calibrate)

# %%
plt.plot(median_calibrate.argmax(axis=0), marker = ".", linestyle = 'none')
plt.ylim(219,226)



# %%


fig, ax = plt.subplots()
ax.imshow(median_calibrate)
#plt.ylim(200, 250)

#%%

peak_positions = median_calibrate.argmax(axis = 0)
peak_maxs = median_calibrate.max(axis = 0)
p

peak_widths = np.apply_along_axis(signal.peak_widths, axis = 0, median_calibrate, kwargs={peaks:[]})# This wont work becasue I cant specify the hight for each column. I need a new funtion that can do all of this in 1 step
peak_sigmas = peak_widths/ 2.355 # conversiton from full width half max FWHM/ 2sqrt(2*ln(2))



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

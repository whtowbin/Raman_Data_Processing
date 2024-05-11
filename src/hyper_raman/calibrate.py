#%%
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import optimize
from copy import deepcopy




#%%

calibrate = io.imread("/Users/wtowbin/Projects/hyper-raman/Dev_Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")


def get_metadata(directory):
    # function to read metadata file for image calibration
    pass

# Running List of image Transformations to apply to full dataset e.g., median filter, normalization, cropping, etc.
# I can potentially have another list of operations performed on calibration images
image_transform_log = []

def reset_transform_log(transform_log = image_transform_log):
    transform_log.clear()

# Maybe have each entry in the form (function,{**kwargs})
# Then I can have another function that applies the log sequentially
# I will need to have it act on an image object or stack.  

def log_transforms(func, transfrom_log = image_transform_log):
    def wrapper(*args, **kwargs): # args should just be image
        result = func(*args, **kwargs)
        transfrom_log.append((func,kwargs,{"shape":result.shape})) # leave out args since it should just be the image file
        return result
    return wrapper

# I could also have another function that keeps track of the array shape after each trans
#%%
median_calibrate = skfilter.median(calibrate)

# # %%
# plt.plot(median_calibrate.argmax(axis=0), marker = ".", linestyle = 'none')
# plt.ylim(219,226)


# fig, ax = plt.subplots()
# ax.imshow(median_calibrate)
# #plt.ylim(200, 250)

#%%


def gauss(x, mu, sd, A=1):

    """
    Pulled from PyIRoGlass

    Return a Gaussian peak for fitting

    Parameters:
        x (numeric): The wavenumbers of interest.
        mu (numeric): The center of the peak.
        sd (numeric): The standard deviation (or width of peak).
        A (numeric, optional): The amplitude. Defaults to 1.

    Returns:
        G (np.ndarray): The Gaussian fit.

    """

    G = A * np.exp(-((x - mu) ** 2) / (2 * sd**2))

    return G
    
def fit_spectrum_peak(array):
    """Function for fitting a gaussian peak to an array of 1D Raman unsaturated spectra

    Args:
        array (numpy array): 1D numpy array of single unsaturated raman peak
    
    Returns:
        gaussian_max (numpy array): peak position with subpixel accuracy
    """
    indices = np.arange(len(array))
    peak_position = array.argmax(axis = 0)
    peak_max = array.max(axis = 0)
    peak_width = signal.peak_widths(array,[peak_position])[0][0]
    peak_sigma = peak_width/ 2.355 # conversiton from full width half max FWHM/ 2sqrt(2*ln(2))
    try:
        popt, pcov = optimize.curve_fit(f=gauss,xdata=indices,ydata=array,p0=(peak_position,peak_sigma,peak_max))
    except:
        print("An exception occured during guassian fitting on array")
        popt = np.array([np.nan,np.nan,np.nan])
    return popt[0]

#mu, sd, A
def test_fit_spectrum(image):
    indices = index_array(image)
    mid_idx = int(np.round(image.shape[1]/2))
    array = median_calibrate[:,mid_idx]
    peak_pos = fit_spectrum_peak(array, indices)
    assert peak_pos is not np.nan
    return peak_pos


def nm_to_wn(nm, laser_nm = 532):
   return 1e7* (1/laser_nm - 1/nm)

def wn_to_nm(wn, laser_nm= 532):
    return 1/(1/laser_nm - wn/10**7)

def pixel_to_nm(peak_pos_nm, peak_pos_pixel, indices, nm_per_pixel = 0.147750):
    return peak_pos_nm + nm_per_pixel * (indices - peak_pos_pixel)

def calibrate_image_wn(array ,peak_pos_wn=1332, laser_nm = 532, nm_per_pixel = 0.147750):
    """Function for fitting a gaussian peak to an array of 1D Raman unsaturated spectra

    Args:
        array (numpy array): 1D numpy array of single unsaturated raman peak
    
    Returns:
        gaussian_max (numpy array): peak position with subpixel accuracy
    """
    indices = np.arange(len(array))
    peak_position_init = array.argmax(axis = 0)
    peak_max = array.max(axis = 0)
    peak_width = signal.peak_widths(array,[peak_position_init])[0][0]
    peak_sigma = peak_width/ 2.355 # conversiton from full width half max FWHM/ 2sqrt(2*ln(2))
    try:
        popt, pcov = optimize.curve_fit(f=gauss,xdata=indices,ydata=array,p0=(peak_position_init,peak_sigma,peak_max))
    except:
        print("An exception occured during guassian fitting on array")
        popt = np.array([np.nan,np.nan,np.nan])

    peak_pos_pixel =  popt[0] #popt is the best-fit guassian params in the form (pos,sigma,amplitude)
    peak_pos_nm = wn_to_nm(peak_pos_wn, laser_nm)
    array_nm = pixel_to_nm(peak_pos_nm, peak_pos_pixel, indices, nm_per_pixel)
    array_wn = nm_to_wn(array_nm, laser_nm)
    return array_wn


#%%
def nm_to_wn(nm, laser_nm = 532):
   return 1e7* (1/laser_nm - 1/nm)

def wn_to_nm(wn, laser_nm= 532):
    return 1/(1/laser_nm - wn/10**7)

def pixel_to_nm(peak_pos_nm, peak_pos_pixel, indices, nm_per_pixel = 0.147750):
    return peak_pos_nm + nm_per_pixel * (indices - peak_pos_pixel)
    

# Replace Defaults with image metadata pulled when stack is read in
def calibrate_image(image, peak_pos_wn=1332, laser_nm = 532, nm_per_pixel = 0.147750):
    #peak_positions = np.apply_along_axis(fit_spectrum_peak, axis = 0,arr = image) 
    peak_positions = np.apply_along_axis(calibrate_image_wn, axis = 0,arr = image) 

    return peak_positions

def replace_bad(array_1D, threshold = 0.01):
    # this function replaces values in an array that are above or below the median by a threshold
    array_1D = deepcopy(array_1D)
    median = np.nanmedian(array_1D)
    pos_thresh, neg_thresh = 1 + threshold, 1 - threshold
    array_1D[(array_1D > median*pos_thresh) | (array_1D < median * neg_thresh)] = np.nan
    return array_1D
#np.argwhere(array_1D == np.nan) # find array location

@log_transforms
def crop_central_columns(image, crop_percent= .1,):
    crop_low = int(image.shape[1]*crop_percent)
    crop_high = int(image.shape[1]* (1-crop_percent))
    return image[:,crop_low:crop_high]

def padding_or_cropping_function(wn_array):
    # This shoud make a repeatable padding or cropping function the can be applied to each image in the same way that whe wavenumber array is to make wavenumber spacing aligned
    pass

#%%
wn_calibration_array = calibrate_image(median_calibrate)

#%%
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_calibration_array[-2, :], marker = ".")
ax.set_ylim((3560,3580))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")


# %%
"""
Notes on how to process data:

- Median Filter to help identify outliers

- Normalize each image against average Laser Power to help with power fluctuations
- Need a nornalization method for across the laser line in each image, Looks like the line shape changes with laser power, see if it chagnes with depth. 

"""





# %%

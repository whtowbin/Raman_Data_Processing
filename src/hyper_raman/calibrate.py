#%%
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import optimize
from scipy import interpolate
from copy import deepcopy
from numba import jit

#%%
calibrate = io.imread("/Users/wtowbin/Projects/hyper-raman/Dev_Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")

def get_metadata(directory):
    # function to read metadata file for image calibration
    pass

# Running List of image Transformations to apply to full dataset e.g., median filter, normalization, cropping, etc.
image_transform_log = []
# I can potentially have another list of operations performed on calibration images

def reset_transform_log(transform_log = image_transform_log):
    transform_log.clear()

# I should consider adding a way to retaining specifc outputs like dark and bright corrections
def log_transforms(func, transfrom_log = image_transform_log):
    def wrapper(*args, **kwargs): # args should just be image
        result = func(*args, **kwargs)
        transfrom_log.append((func,kwargs,{"shape":result.shape})) # leave out args since it should just be the image file
        return result
    return wrapper
#%%

def round_half(number):
    return np.round(number*2)/2
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

def calibrate_array_wn(array ,peak_pos_wn=1332.0, laser_nm = 532, nm_per_pixel = 0.147750):
    """Function for fitting a gaussian peak to an array of 1D Raman unsaturated spectra and returning an array in wavenumber (cm^-1)
    The defaults are for the primary diamond Raman peak at 1332 cm^-1
    Args:
        array (_type_): 1D numpy array of single unsaturated raman peak
        peak_pos_wn (float, optional): Reference Peak Location. Defaults to 1332.0 for diamond.
        laser_nm (int, optional): Excitation laser wavelength. Defaults to 532nm.
        nm_per_pixel (float, optional): Linear spacing in nanometeres for each pixel. Defaults to 0.147750nm.

    Returns:
        array: 1D array whose values correspond to the wavenumber coordinates for the input array. 
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

# Replace Defaults with image metadata pulled when stack is read in
def calibrate_image(image, peak_pos_wn=1332, laser_nm = 532, nm_per_pixel = 0.147750):
    peak_positions = np.apply_along_axis(calibrate_array_wn, axis = 0,arr = image) 
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
def crop_central_columns(image, crop_percent= .05):
    crop_low = int(image.shape[1]*crop_percent)
    crop_high = int(image.shape[1]* (1-crop_percent))
    return image[:,crop_low:crop_high]



# I am not exactly sure how to do this interpolation and apply it to the images 
# I think i need to crop the Data and then Interpolate it to the new array. 
# It might make the most sense to do this in nanometers or pixel units then switch to wavenumber
# I think that the way it should go is to apply it to each row, func = interpolate(Detector X, Column_Data)
# func (output array)
def interpolate_images(calib_image_wn, image, min=None, max= None):
    interp_spacing = np.round((np.nanmin(calib_image_wn[-1,:] - calib_image_wn[-2,:])),1)
    if min == None:
        min = round_half(np.nanmax(calib_image_wn[0,:]))
    if max == None:
        max = round_half(number=np.nanmin(calib_image_wn[-1,:]))
    print(f"min:{min},  max:{max}, step:{interp_spacing}")
    interp_array_wn = np.arange(min,max,interp_spacing)
    
    def interp_func(image_stack, interp_array = interp_array_wn):
        x = image_stack[:,:,0]
        y = image_stack[:,:,1]
        return interpolate.CubicSpline(x,y, interp_array)
    
    stacked = np.dstack([calib_image_wn,image])
    return np.apply_along_axis(interp_func, axis = 0,arr = stacked)

    # I need to iterate over both arrays at the same time. 
    

def padding_or_cropping_function(wn_array):
    # This shoud make a repeatable padding or cropping function the can be applied to each image in the same way that whe wavenumber array is to make wavenumber spacing aligned
    pass


#%%

#%%
wn_calibration_array = calibrate_image(median_calibrate)
#%%

cropped_wn = crop_central_columns(wn_calibration_array)

cropped_image = crop_central_columns(median_calibrate)

#interpolate_images(cropped_wn, cropped_image)



#%%d
#I could maybe use numba to speed this up but I would probbaly need to switch to numpy's linear interp or numba fast iterp package 

def interp_iterate(calib_image_wn, image, min=None, max= None):
    interp_spacing = np.round((np.nanmin(calib_image_wn[-1,:] - calib_image_wn[-2,:])),1)
    if min == None:
        min = round_half(np.nanmax(calib_image_wn[0,:]))
    if max == None:
        max = round_half(number=np.nanmin(calib_image_wn[-1,:]))
    interp_array_wn = np.arange(min,max,interp_spacing)

    results = []
    
    for idx in range(calib_image_wn.shape[1]):
        x = calib_image_wn[:,idx]
        y = image[:,idx]

        interp_fn = interpolate.Akima1DInterpolator(x,y)
        results.append(interp_fn(interp_array_wn))
    return np.array(results)
#%%

test_interp = interp_iterate(cropped_wn, cropped_image)
#%%
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_calibration_array[-2, :], marker = ".")
ax.set_ylim((3560,3580))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")

#%%
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_calibration_array[222, :], marker = ".")
ax.set_ylim((1315,1335))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")

#%%
plt.imshow(wn_calibration_array)


#%%
# make array from max to min and map data onto it

# %%
"""
Notes on how to process data:

- Median Filter to help identify outliers

- Normalize each image against average Laser Power to help with power fluctuations
- Need a nornalization method for across the laser line in each image, Looks like the line shape changes with laser power, see if it chagnes with depth. 

"""





# %%

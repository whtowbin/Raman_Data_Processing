#%%
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import optimize




#%%

calibrate = io.imread("/Users/wtowbin/Projects/hyper-raman/Dev_Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")

# Running List of image Transformations to apply to full dataset e.g., median filter, normalization, cropping, etc.
image_transforms = []

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
    

def index_array(image, axis = 0):
    return np.arange(image.shape[axis])

def fit_spectrum_peak(array):#, indices):
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

def calibrate_image(image, peak_pos=1332):
    indices = index_array(image)
    # apply_along_axis is failing with error: TypeError: unhashable type: 'numpy.ndarray'
    peak_positions = np.apply_along_axis(fit_spectrum_peak, axis = 0,arr = image,) #**{indices:[indices]})
    return peak_positions


#%%
def my_func(a):
    """Average first and last element of a 1-D array"""
    return (a[0] + a[-1]) * 0.5

result =np.apply_along_axis(np.mean, 0, median_calibrate)
plt.plot(result)
#%%
peak_pos = calibrate_image(median_calibrate)

#%%
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(peak_pos)
ax.set_ylim((221,225))
ax.set_xlabel("Column")
ax.set_ylabel("Peak Position")

# %%
"""
Notes on how to process data:

- Median Filter to help identify outliers

- Normalize each image against average Laser Power to help with power fluctuations
- Need a nornalization method for across the laser line in each image, Looks like the line shape changes with laser power, see if it chagnes with depth. 

"""





# %%

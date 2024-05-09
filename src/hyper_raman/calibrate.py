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


peak_widths = np.apply_along_axis(signal.peak_widths, axis = 0, median_calibrate, kwargs={peaks:[]})# This wont work becasue I cant specify the height for each column. I need a new funtion that can do all of this in 1 step
peak_sigmas = peak_widths/ 2.355 # conversiton from full width half max FWHM/ 2sqrt(2*ln(2))

# Consider masking array with Nans for areas where laser signal is unstable


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
    

def index_array(image, axis = 0)
    return np.arange(image.shape[axis])

def fit_spectrum_peak(array, length):
    """Function for fitting a gaussian peak to an array of 1D Raman unsaturated spectra

    Args:
        array (numpy array): 1D numpy array of single unsaturated raman peak
    
    Returns:
        gaussian_max (numpy array): peak position with subpixel accuracy
    """

    peak_position = array.argmax(axis = 0)
    peak_max = array.max(axis = 0)
    peak_widths = signal.peak_widths(array,peak_position)
    peak_sigma = peak_width/ 2.355 # conversiton from full width half max FWHM/ 2sqrt(2*ln(2))
    try:
        popt, pcov = optimize.curve_fit(f=gauss,xdata=indices,ydata=array,p0=(peak_position,peak_sigma,peak_max))
    except:
        pass
    return popt
#mu, sd, A
def calibrate(image, peak_pos):
    indices = index_array(image)


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

#%%
import calibrate
from skimage import io
import skimage.filters as skfilter
import matplotlib.pyplot as plt
import tifffile
import numpy as np
#%%
image_transform_log = calibrate.start_logging()

image = io.imread(fname="/Users/wtowbin/Projects/hyper-raman/Dev_Data/Unsaturated Raman/Unsaturated Raman 10x_0000.tif")

median_calibrate = skfilter.median(image)


# %%
plt.plot(median_calibrate.argmax(axis=0), marker = ".", linestyle = 'none')
plt.ylim(219,226)

fig, ax = plt.subplots()
ax.imshow(median_calibrate)
#plt.ylim(200, 250)

# %%#%%
wn_calibration_array = calibrate.calibrate_image(median_calibrate)
#%%

cropped_wn = calibrate.crop_central_columns(wn_calibration_array)

cropped_image = calibrate.crop_central_columns(median_calibrate)



#interpolate_images(cropped_wn, cropped_image)

#%%
# %%timeit
wn_1D = calibrate.interpolate_image(cropped_wn, cropped_image, return_wn=True)
test_interp = calibrate.interpolate_image(cropped_wn, cropped_image)


# Processing would loop through the stack and apply the dark correction,  cropping, and interpolation for each image then save it. 
#%%

fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_1D,test_interp[:,10:300])
ax.set_ylim((0,1000))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")
#%%
calibrated_avg = test_interp.mean(axis=1)
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_1D,calibrated_avg)
ax.set_ylim((0,1000))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")

#%%
uncalibrated_avg = cropped_image.mean(axis=1)
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(uncalibrated_avg)
ax.set_ylim((0,1000))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")

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
# Plot Noise in array
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(wn_calibration_array[222, :], marker = ".")
ax.set_ylim((1328.5,1329))
ax.set_xlim((1050,1115))
ax.set_xlabel("Column")
ax.set_ylabel("Pixel row in wn")
#%%
plt.imshow(wn_calibration_array)

#%%


#io.imsave("Interpolated_image_test.tif", test_interp)
# %%
imout16bit = np.round(test_interp).astype('uint16')

tifffile.imwrite("Interpolated_image_test_custom.tif", imout16bit)
# %%

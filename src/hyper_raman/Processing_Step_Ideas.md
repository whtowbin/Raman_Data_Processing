# Processing Step List

## Calibration Image
1) Calibrate Spectrometer image based on Diamond Peak Position 
2) Interpolate Data to Regular Grid
3) For each spectrometer Image in a folder do this to create an image
4) Then Repeat to stack each image into a final 

## How to read through original files 
Here is the general format:
Directory
  - Metadata.txt
  - Slices
    - Directory Names: Position# microns
      - Images possibly multiple if averages 
  
I should process the Calibration and Dark images the same way with this.


## Questions to Work Through
- What format should I put the functions in?
  - Currently it is in a series of functions and not classes. This will require a script to compose 
  - I am a bit against writing classes since it has extra overhead, but it would be nice to have a wrapper classs  to process the maps in xarray more easily. 
    - i.e. integrate, peak select, baseline,  

- Xarray map objects functions
- map.select_wn(wn_low, wn_high)
- map.baseline(p, lam) # Start with a wrapper for simple baseline function
- map.intergrate(wn_low, wn_high)
- map.select_peak(peak_pos)
- map.crop() # I should access this both by index and by spatial position


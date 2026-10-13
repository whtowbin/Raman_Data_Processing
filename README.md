# Hyper-Raman: Raman spectrometer image processing

Tools for calibrating raw spectrometer camera images from a line-scanning Raman microscope, the first step toward hyperspectral Raman maps.

## What it does

`src/hyper_raman/calibrate.py`:

- Fits a Gaussian to the diamond Raman peak (1332 cm-1) down each column of a calibration image and converts pixel positions to wavelength and wavenumber for a 532 nm laser, giving a wavenumber for every pixel (`calibrate_image`).
- Interpolates an image onto a regular wavenumber grid from that calibration (`interpolate_image`) and saves the result as a 16-bit TIFF (`save_TIFF`).
- Records transforms applied to an image with a decorator (`log_transforms`, so far used for cropping) so the same steps can be repeated on a full dataset.

`MetaDataReader_Draft.py` reads scan metadata and positions from file names; `test_calibrate.py` is a development script that runs the steps on local data not included in the repository.

## Install

```bash
git clone https://github.com/whtowbin/Raman_Data_Processing
cd Raman_Data_Processing
uv sync            # or: pip install -e .
```

Not on PyPI.

## Status

This is a working research repository, not a finished package. Some files are exploratory analyses or earlier versions kept for reference, and the layout will change as I clean it up. Bad-pixel, dark and intensity corrections and stacking a scan into a map are not written yet (see the TODO notes at the end of `calibrate.py` and `Processing_Step_Ideas.md`).

## License

The code is under the MIT License (see LICENSE.txt).

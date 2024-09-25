#%%
import pandas as pd
from pathlib import Path


#%%
Path_name = "/Users/wtowbin/Projects/hyper-raman/Dev_Data/unsat raman/Metadata.txt"

def read_metadata(Path):
    """_summary_

    Args:
        Path (_type_): _description_
    """
    metadata_df = pd.read_csv(Path,
                delimiter="\t",
                index_col=0,
                names=["values", "none"])

    metadata_df = metadata_df.iloc[:,0:1]
    metadata_dict = metadata_df.to_dict()["values"]

    Step_XY = metadata_dict["Step Size (microns)"]
    Step_Z = metadata_dict["Step Size (Z)(Microns)"]
    nm_per_pixel = metadata_dict["nm/pixel"]
    Illumination_Source = metadata_dict['Illumination Source']
    try:
        laser_nm = float(metadata['Illumination_Source'].split()[0])
    except:
        print(f"Could not determine laser wavelength from metadata. Input given:{Illumination_Source}")
        input = input("Please Input the laser wavelength used for this data?:")
    return {"Step_XY":Step_XY, "Step_Z": Step_Z, "nm_per_pixel": nm_per_pixel, 'Illumination_Source': Illumination_Source, "laser_nm":laser_nm }

# %%
#%%

def parse_directory(dir):
    # This function should run through data collected by Daniel's spectrometers and determine paths to relevant files. 
    # Basic Structure:
    #   - Metadata.txt
    #   - /Slices
    #       - /Micron Slices 
    #           - Spectrometer Files
    #   Alternative Files Currently in another folder 
    #   - Dark Image Corrections
    #   - Unsaturated Raman Corrections. 
    #   - Potentially Bright Corrections 

    # I will need to loop over the micro slice

    dir_path = Path(dir)
    #dir_path.glob()
    # data 

def list_directories(dir):
    p = Path(dir)
    return [x for x in p.iterdir() if x.is_dir()]
    
def list_files(dir):
    p = Path(dir)
    p.glob("*.tiff")
    return [x for x in p.iterdir() if x.is_file()]


#%%
dir = "/Users/wtowbin/Projects/hyper-raman/Dev_Data/unsat raman"


test_image_dir = "/Users/wtowbin/Projects/hyper-raman/Dev_Data/Test_Data"
#%%

metadata = read_metadata(Path_name)

if __name__ == "__main__":
    read_metadata(Path_name)
# %%

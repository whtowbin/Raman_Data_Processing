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
    # read csv with Pandas and load it into a dict to access the values 
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
        laser_nm = float(Illumination_Source.split()[0])
    except Exception as error:
        print("An error occurred:", type(error).__name__, "–", error)
        print(f"Could not determine laser wavelength from metadata. Input given:{Illumination_Source}")
        laser_nm = input("Please Input the laser wavelength used for this data?:")
    return {"Step_XY":Step_XY, "Step_Z": Step_Z, "nm_per_pixel": nm_per_pixel, 'Illumination_Source': Illumination_Source, "laser_nm":laser_nm}

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
    
def list_files(dir, filetype = "*.tiff"):
    p = Path(dir)
    return list(sorted(p.glob(filetype)))
    #return [x for x in p.iterdir() if x.is_file()]

def parse_position_from_name(path_object):
    # Assumes files are in the format of "####.## microns.tiff" 
    return float(path_object.name.split(" ")[0])

#  



# Test section
#%% Test section
dir = "/Users/wtowbin/Projects/hyper-raman/Dev_Data/unsat raman"

test_image_dir = "/Users/wtowbin/Projects/hyper-raman/Dev_Data/Test_Data"
#%%

metadata = read_metadata(Path_name)

Dir_tree = list_directories(dir)

img_tree = list_files(test_image_dir)



#%%
if __name__ == "__main__":
    read_metadata(Path_name)
# %%

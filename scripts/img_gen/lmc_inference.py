from omegaconf import OmegaConf
import os
import torch
from pathlib import Path
import tifffile
import xmltodict
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from ldm.data import image_processing
from einops import rearrange
import cv2
from skimage import exposure

#d = '/home/trangle/Desktop/submission/test'
#INPUT_PATH = f"{d}/input"
#OUTPUT_PATH = f"{d}/output"
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
if not os.path.isdir(os.path.join(OUTPUT_PATH,"images")): os.mkdir(os.path.join(OUTPUT_PATH,"images"))

############# HELPER FUNCTIONS #################

def read_image(location):
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray()    # Extract image data
        metadata   = tif.ome_metadata # Get the existing metadata in a DICT
    return image_data, metadata

def save_image(*, location, array,metadata):
    #Save each predicted images with the required metadata
    print(" --> save "+str(location))
    pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
    physical_size_x = float(pixels["@PhysicalSizeX"])
    physical_size_y = float(pixels["@PhysicalSizeY"])
    tifffile.imwrite(location,
                     array,
                     description=metadata.encode(),
                     resolution=(physical_size_x, physical_size_y),
                     metadata=pixels,
                     tile=(128, 128),
                     )
    
# For each submission, 36 algorithm jobs will be created from the 36 valid items (=Phase 1 test image)
# -> No batch processing, prepping image one by one
location_mapping = {"Tubulin": 0, "Actin": 1, "Mitochondria": 2}
def one_hot_encode_locations(locations, location_mapping):
    loc_labels = list(map(lambda n: location_mapping[n] if n in location_mapping else -1, str(locations).split(',')))
    # create one-hot encoding for the labels
    locations_encoding = np.zeros((max(location_mapping.values()) + 1, ), dtype=np.float32)
    locations_encoding[loc_labels] = 1
    return locations_encoding

def prepare_input1(bf, transforms = None):
    bf = cv2.resize(bf, (256,256), interpolation = cv2.INTER_LINEAR)
    imarray = np.stack([bf,bf,bf], axis=2)
    assert image_processing.is_between_0_255(imarray)
    imarray = image_processing.convert_to_minus1_1(imarray)
    imarray = torch.from_numpy(imarray).unsqueeze(0)
    imarray = rearrange(imarray, 'b h w c -> b c h w').contiguous()
    assert imarray.shape == (1,3,256,256)
    return imarray

def rescale_2_98(arr):
    p2, p98 = np.percentile(arr, (2, 99.8))
    arr = exposure.rescale_intensity(arr, in_range=(p2, p98), out_range=(0, 255)).astype(np.uint8)
    return arr

def load_model1(device, ckpt_path='./checkpoints/BF_to_Nucleus.ckpt'):
    # Model1
    config1 = OmegaConf.load('./configs/BF_to_org.yaml')
    checkpoint1 = ckpt_path
    model = instantiate_from_config(config1['model'])
    model.load_state_dict(torch.load(checkpoint1, map_location="cpu")["state_dict"], strict=False)
    model = model.to(device)
    return model.eval()

def predict_with_vqgan(x, model):
    h, _, _ = model.encode(x)
    ypred = model.decode(h)
    ypred = torch.clip(ypred, min=-1, max=1)
    ypred = torch.permute(ypred, (0, 2, 3, 1)).to('cpu').detach().numpy()
    ypred = (ypred + 1) / 2 * 65535 # -> [0, 65535]
    return ypred.astype(np.uint16).squeeze()

def main():
    # Loading models
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    m1 = load_model1(device)

    #List the input files
    transmitted_light_path = os.path.join(INPUT_PATH , "images","organelles-transmitted-light-ome-tiff")
    for input_file_name in os.listdir(transmitted_light_path):
        if input_file_name.endswith(".tiff"):
            print(" --> Predict " + os.path.join(transmitted_light_path,input_file_name))
            bf, metadata = read_image(os.path.join(transmitted_light_path,input_file_name))
            if bf.max() == 0:
                print('image empty')
                for organelle in ["Nucleus","Mitochondria", "Actin", "Tubulin"]:
                    output_organelle_path = os.path.join(OUTPUT_PATH, "images", organelle.lower() + "-fluorescence-ome-tiff")
                    os.makedirs(output_organelle_path, exist_ok=True)
                    save_image(location=os.path.join(output_organelle_path,os.path.basename(input_file_name)), array=bf, metadata=metadata)
            else:
                bf = rescale_2_98(bf)
                print('BF input: ', bf.dtype, bf.shape, bf.min(), bf.max())
                bf_batch = prepare_input1(bf)
                bf_batch = torch.tensor(bf_batch).to(device)
                for organelle in ["Nucleus","Mitochondria", "Actin", "Tubulin"]:
                    m = load_model1(device, ckpt_path = f"./checkpoints/BF_to_{organelle}.ckpt")
                    pred_batch = predict_with_vqgan(bf_batch, m)
                    pred = pred_batch.mean(axis=2).astype(np.uint16)
                    pred = cv2.resize(pred, (bf.shape[1],bf.shape[0]), interpolation = cv2.INTER_LINEAR) # upscale
                    #print('Output:', nucleus.shape, nucleus.min(), nucleus.max(), nucleus.dtype)
                    output_organelle_path = os.path.join(OUTPUT_PATH, "images", organelle.lower() + "-fluorescence-ome-tiff")
                    os.makedirs(output_organelle_path, exist_ok=True)
                    save_image(location=os.path.join(output_organelle_path,os.path.basename(input_file_name)), array=pred, metadata=metadata)


if __name__ == "__main__":
    raise SystemExit(main())
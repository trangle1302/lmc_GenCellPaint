import numpy as np
from PIL import Image
from skimage import exposure
import torch
import tifffile

def is_float(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    elif torch.is_tensor(x):
        return torch.is_floating_point(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_integer(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif torch.is_tensor(x):
        return not torch.is_floating_point(x) and not torch.is_complex(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_between_minus1_1(x):
    correct_range = x.min() >= -1 and x.min() < 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_1(x):
    correct_range = x.min() >= 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_255(x):
    correct_range = x.min() >= 0 and x.max() <= 255
    if correct_range:
        assert is_integer(x)
        return True
    else:
        return False


def convert_to_uint8(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.uint8)
    elif torch.is_tensor(x):
        return x.to(torch.uint8)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def convert_to_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif torch.is_tensor(x):
        return x.to(torch.float32)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def convert_to_0_1(x):
    if is_between_minus1_1(x):
        y = (x + 1) / 2
    elif is_between_0_1(x):
        y = x
    elif is_between_0_255(x):
        y = x / 255
        return convert_to_float32(x)
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_float32(y)


def convert_to_0_255(x):
    if is_between_minus1_1(x):
        y = ((x + 1) / 2 * 255)
    elif is_between_0_1(x):
        y = x * 255
    elif is_between_0_255(x):
        y = x
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_uint8(y)


def convert_to_minus1_1(x):
    if is_between_minus1_1(x):
        y = x
    elif is_between_0_1(x):
        y = x * 2 - 1
    elif is_between_0_255(x):
        y = x / 255 * 2 - 1
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')
    return convert_to_float32(y)


def get_bbox_from_mask(mask, bbox_label):
    # Identify the positions of the object in the mask
    positions = np.where(mask == bbox_label)

    if len(positions[0]) > 0 and len(positions[1]) > 0: # The object is in the mask

        # Calculate bounding box
        top = np.min(positions[0])
        left = np.min(positions[1])
        bottom = np.max(positions[0])
        right = np.max(positions[1])
    else:
        top = left = bottom = right = 0
    return top, left, bottom, right


def crop_around_object(img, mask, bbox_label, size):
    """
    Crop the image around the object specified by bbox_label in the mask.

    Parameters:
    - img: numpy array of the image (height, width, 3).
    - mask: numpy array of the mask (same size as img), integers.
    - bbox_label: Integer, label in the mask for the object of interest.
    - size: Integer, side length of the square crop.

    Returns:
    - Cropped (and possibly padded) image around the object.
    """
    
    top, left, bottom, right = get_bbox_from_mask(mask, bbox_label)

    if size > 0:
        # Calculate center of the bounding box
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2

        h, w, c = img.shape
        half_size = size // 2
        
        top = max(0, center_y - half_size)
        left = max(0, center_x - half_size)
        bottom = min(h, center_y + half_size)
        right = min(w, center_x + half_size)

    # Cropping
    cropped_img = img[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]

    if size > 0:
        # Padding
        pad_top = abs(min(0, center_y - half_size))
        pad_bottom = abs(h - max(h, center_y + half_size))
        pad_left = abs(min(0, center_x - half_size))
        pad_right = abs(w - max(w, center_x + half_size))
        cropped_img = np.pad(cropped_img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
        cropped_mask = np.pad(cropped_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

        assert cropped_img.shape == (size, size, c)
        assert cropped_mask.shape == (size, size)

    return cropped_img, cropped_mask

    
def load_jump(image_ids,rescale=True):
    data_dir = "/scratch/groups/emmalu/JUMP/processed_tiled"
    # JUMP channels
    # 1: Mito, 2: AGP, 3: NucleoliRNA, 4: ER, 5: Nucleus, 6: BF
    imgs = []
    for image_path in image_ids:
        imgarray = np.array(Image.open(f'{data_dir}/{image_path}'))
        imgs.append(imgarray)
    for i in range(3-len(imgs)):  # ch=1 don't work for normal conv set up, assert ch=3
        imgs.append(imgarray)
    full_res_image = np.stack(imgs, axis=2)
    #combine channels into single multichannel image
          
    if rescale:
        p2, p99 = np.percentile(full_res_image, (2, 99.8))
        full_res_image = exposure.rescale_intensity(full_res_image, in_range=(p2, p99), out_range=(0, 255)).astype(np.uint8)
    else:
        full_res_image = (full_res_image/256).astype(np.uint8)
    assert is_between_0_255(full_res_image)
    return full_res_image

def load_ometiff_image(image_id, chs, rescale=True):
    """
    Load ome.tiff image
    args:
        image_id: str, format f'{study}/{Image_id}'
        chs: str
    """
    data_dir = "/scratch/groups/emmalu/lightmycell/Images"
    
    imgs = []
    for ch in chs:
        #print('Reading : ', f'{data_dir}/{image_id}_{ch}.ome.tiff')
        imgarray = tifffile.imread(f'{data_dir}/{image_id}_{ch}.ome.tiff')
        imgs.append(imgarray)
    for i in range(3-len(imgs)):  # ch=1 don't work for normal conv set up, assert ch=3
        imgs.append(imgarray)
    full_res_image = np.stack(imgs, axis=2)
            
    if rescale:
        p2, p99 = np.percentile(full_res_image, (2, 99.8))
        full_res_image = exposure.rescale_intensity(full_res_image, in_range=(p2, p99), out_range=(0, 255)).astype(np.uint8)
    else:
        full_res_image = (full_res_image/256).astype(np.uint8)
    assert is_between_0_255(full_res_image)
    return full_res_image 

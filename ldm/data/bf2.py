import cv2
import albumentations
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from ldm.data import image_processing
#from ldm.data.serialize import TorchSerializedList
import torch.nn as nn
from einops import rearrange
from ldm.util import instantiate_from_config
import torch
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import random
import numpy as np

location_mapping = {"Tubulin": 0, "Actin": 1, "Mitochondria": 2}
def one_hot_encode_locations(locations, location_mapping):
    loc_labels = list(map(lambda n: location_mapping[n] if n in location_mapping else -1, str(locations).split(',')))
    # create one-hot encoding for the labels
    locations_encoding = np.zeros((max(location_mapping.values()) + 1, ), dtype=np.float32)
    locations_encoding[loc_labels] = 1
    return locations_encoding

class BFPaint:

    def __init__(self, group='train', path_to_metadata=None, input_channels=None, output_channels=None, is_ldm=False, size=512, scale_factor=1, flip_and_rotate=True, return_info=False, refine=False):

        self.is_ldm = is_ldm # False: AE, True: use inputs as ref-image, outputs as stage_1
        #Define channels
        self.input_channels = input_channels
        if output_channels is None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels
        
        self.metadata = pd.read_csv(path_to_metadata)
        if refine:
            self.metadata = self.metadata[self.metadata.datasource=='lmc']
            # weights of ['Actin','Tubulin','Mitochondria'] = [24.98765432,  3.03903904,  0.3800939 ] 
            ids_a = self.metadata[self.metadata.Ch=='Actin'].Image_id.tolist()
            df_a = self.metadata.loc[np.repeat(self.metadata[self.metadata.Image_id.isin(ids_a)].index.values, 64)]
            ids_t = self.metadata[self.metadata.Ch=='Tubulin'].Image_id.tolist()
            df_t = self.metadata.loc[np.repeat(self.metadata[self.metadata.Image_id.isin(ids_t)].index.values, 7)]
            self.metadata = pd.concat([self.metadata, df_a, df_t])
        
        self.metadata['io'] = [('ref' if f in ['BF','Nucleus'] else 'org') for f in self.metadata.Type ]
        if (self.input_channels == 'org') & (self.input_channels == self.output_channels): # phase 1 org
            self.metadata = self.metadata[self.metadata.io == 'org']
        elif self.output_channels == 'org': # phase 2 ldm
            imgs_keep = set(self.metadata.Image_id)
            for ch in set(self.input_channels): # BF, Nucleus
                imgs_k = self.metadata.groupby('Image_id').agg({'Type': set })
                imgs_k['Type'] = [(set(list(t)[0].split(',')) if len(t)==1 else t) for t in imgs_k.Type]
                imgs_k = imgs_k.iloc[[(ch in f) for f in imgs_k.Type]]
                # Check if this image_id also contains 1 of the organelle channels
                imgs_k = imgs_k.iloc[[len({'Mitochondria','Tubulin','Actin'}.intersection(f))>0 for f in imgs_k.Type]]
                imgs_keep = imgs_keep.intersection(imgs_k.index.tolist())# imgs_keep.update(imgs_k.index.tolist())
                
            self.metadata = self.metadata[self.metadata.Image_id.isin(imgs_keep)].drop_duplicates()
        else: # phase 1 ref, or bf -> nu
            imgs_keep = set(self.metadata.Image_id)
            for ch in set(self.input_channels + self.output_channels):
                imgs_k = self.metadata.groupby('Image_id').agg({'Type': set })
                imgs_k['Type'] = [(set(list(t)[0].split(',')) if len(t)==1 else t) for t in imgs_k.Type]
                imgs_k = imgs_k.iloc[[(ch in f) for f in imgs_k.Type]]
                imgs_keep = imgs_keep.intersection(imgs_k.index.tolist())# imgs_keep.update(imgs_k.index.tolist())
                
            self.metadata = self.metadata[self.metadata.Image_id.isin(imgs_keep)].drop_duplicates()
            
        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)   
        self.indices = self.metadata[(self.metadata.split==group)].index
        self.image_ids = self.metadata[(self.metadata.split==group)].Image_id
    
        self.return_info = return_info

        self.final_size = int(size*scale_factor)
        self.transforms = [albumentations.geometric.resize.Resize(height=self.final_size, width=self.final_size, interpolation=cv2.INTER_LINEAR)]
        self.flip_and_rotate = flip_and_rotate
        if self.flip_and_rotate:
            self.transforms.extend([albumentations.RandomRotate90(p=1.0),
                            albumentations.HorizontalFlip(p=0.5)])
        self.data_augmentation = albumentations.Compose(self.transforms)

        print(f"Dataset group: {group}, length: {len(self.indices)}, in channels: {self.input_channels},  output channels: {self.output_channels}")

    def __len__(self):
        return len(self.image_ids) # len(set(self.image_ids)) * 3
        #return 16 # testing #len(self.image_ids)


    def __getitem__(self, i):
        sample = {}
        #get image
        img_index = self.indices[i]
        info = self.metadata.iloc[img_index].to_dict()
        datasource = info["datasource"]
        img_id = info["Image_id"]
        if (self.input_channels == 'org') & (self.input_channels == self.output_channels): # AE orgs
            Ch = info['Ch']
            if datasource == 'lmc':
                image_id = "/".join([info["Study"], img_id])
                imarray = image_processing.load_ometiff_image(image_id, [Ch, Ch, Ch], rescale=True)
                targetarray = image_processing.load_ometiff_image(image_id, [Ch, Ch, Ch], rescale=True)
                (image_width, image_height) = imarray.shape[:2]
            elif datasource == 'HPA':
                imarray = image_processing.load_intensity_rescaled_image(img_id)
                mapping_d = {'Actin': 1, 'Nucleus': 2, 'Tubulin': 0, 'ER': 3, 'Mitochondria':1}
                in_chs = [mapping_d[Ch] for _ in range(3)]
                targetarray = imarray[:, :, in_chs]
                imarray = imarray[:, :, in_chs]
                (image_width, image_height) = imarray.shape[:2]
            elif datasource == 'jump':
                imarray = image_processing.load_jump([info["Id"]])
                targetarray = imarray.copy()
                (image_width, image_height) = imarray.shape[:2]
                
        elif self.output_channels == 'org': # ldm
            # assert datasource == 'lmc'
            img_df = self.metadata[self.metadata.Image_id == img_id] # all channels and stack for this image_id
            in_chs = []
            #print(self.input_channels)
            for ch in self.input_channels:
                if ch != 'BF':
                    in_chs.append(ch if datasource == 'lmc' else img_df[img_df.Ch==ch].Id.values[0])
                else: # sample 1 BF zstack
                    if datasource=='lmc':
                        chosen_bf = img_df[img_df.Type=='BF'].Id.sample(1).values[0].replace(img_id,'')[1:].replace('.ome.tiff','')
                    else:
                        chosen_bf = img_df[img_df.Type=='BF'].Id.sample(1).values[0]
                    in_chs.append(chosen_bf)
                    
            possible_ch = set(img_df.Type.unique()).difference(['BF','Nucleus'])
            ch = random.choice(list(possible_ch))
            info['Ch'] = ch # update organnelle for guide latent diffusion
            #print(img_df, f'Possible ch {possible_ch}')
            info['BF_mode'] = img_df[img_df.Type=='BF'].Ch.values[0]
            if datasource == 'lmc':
                image_id = "/".join([info["Study"], img_id])
                imarray = image_processing.load_ometiff_image(image_id, in_chs, rescale=True)
                targetarray = image_processing.load_ometiff_image(image_id, [ch,ch,ch], rescale=True)
            elif datasource == 'jump':
                imarray = image_processing.load_jump(in_chs)
                print(img_df[img_df.Ch==ch])
                out_chs = [img_df[img_df.Ch==ch].Id.values[0]]
                #print(out_chs)    
                targetarray = image_processing.load_jump(out_chs)
            else:
                print('datasource not implemented')
            
            image_width, image_height = imarray.shape[:2]
        else:
            if datasource == 'lmc':
                img_df = self.metadata[self.metadata.Image_id == img_id]
                img_df = img_df[img_df.Type=='BF']
                in_chs = []
                for ch in self.input_channels:
                    if ch != 'BF':
                        in_chs.append(ch)
                    else:
                        in_chs.append(img_df.Id.sample(1).values[0].replace(img_id,'')[1:].replace('.ome.tiff',''))
                #print(self.input_channels, self.output_channels)
                if self.input_channels == self.output_channels:
                    out_chs = in_chs
                else:
                    out_chs = []
                    for ch in self.output_channels:
                        if ch != 'BF':
                            out_chs.append(ch)
                        else:
                            #print(img_df.Id.sample(1).values)   
                            out_chs.append(img_df.Id.sample(1).values[0].replace(img_id,'')[1:].replace('.ome.tiff',''))
                            
                image_id = "/".join([info["Study"], img_id])
                imarray = image_processing.load_ometiff_image(image_id, in_chs, rescale=True)
                targetarray = image_processing.load_ometiff_image(image_id, out_chs, rescale=True)
                #print(imarray.shape, targetarray.shape)
                image_width, image_height = imarray.shape[:2]
                #print(in_chs, out_chs) 
                
            elif datasource == "HPA":
                imarray = image_processing.load_intensity_rescaled_image(img_id)
                mapping_d = {'Actin': 1, 'Nucleus': 2, 'Tubulin': 0, 'ER': 3}
                #print(f'{img_id} loaded: {imarray.shape}')
                in_chs = [mapping_d[ch] for ch in self.input_channels]
                for i in range(3-len(in_chs)):
                    in_chs.append(in_chs[0])
                out_chs = [mapping_d[ch] for ch in self.output_channels]
                for i in range(3-len(out_chs)):
                    out_chs.append(out_chs[0])
                #print(in_chs, out_chs)
                targetarray = imarray[:, :, out_chs]
                imarray = imarray[:, :, in_chs]
                (image_width, image_height) = imarray.shape[:2]
            elif datasource == 'jump':
                img_df = self.metadata[self.metadata.Image_id == img_id]
                in_chs = []
                for ch in self.input_channels:
                    if ch != 'BF':
                        in_chs.append(img_df[img_df.Ch==ch].Id.values[0])
                    else:
                        in_chs.append(img_df[img_df.Type=='BF'].Id.sample(1).values[0])
                if self.input_channels == self.output_channels:
                    out_chs = in_chs
                else:
                    out_chs = []
                    for ch in self.output_channels:
                        if ch != 'BF':
                            out_chs.append(img_df[img_df.Ch==ch].Id.values[0])
                        else:
                            out_chs.append(img_df[img_df.Type=='BF'].Id.sample(1).values[0])
                
                imarray = image_processing.load_jump(in_chs)
                targetarray = image_processing.load_jump(out_chs)
                (image_width, image_height) = imarray.shape[:2]
                
                print('JUMP:', in_chs, out_chs)
            else:
                print('datasource not implemented')
        
        assert imarray.shape == (image_width, image_height, 3)
        assert targetarray.shape == (image_width, image_height, 3)
        assert image_processing.is_between_0_255(imarray)
        
        transformed = self.data_augmentation(image=imarray, mask=targetarray)
        imarray = transformed["image"]
        targetarray = transformed["mask"]
        
        #print(img_id, imarray.shape, imarray.max(), imarray.dtype, '>>>>>', targetarray.shape, targetarray.max(), targetarray.dtype)
        imarray = image_processing.convert_to_minus1_1(imarray)
        targetarray = image_processing.convert_to_minus1_1(targetarray)
        
        assert imarray.shape == (self.final_size, self.final_size, 3)
        assert targetarray.shape == (self.final_size, self.final_size, 3)
        
        if self.is_ldm:
            sample.update({"image": targetarray, 
                           "ref-image": imarray, 
                           "location_classes": one_hot_encode_locations(info["Ch"], location_mapping),
                           "BF_mode": info["BF_mode"]})
            print(sample['BF_mode']) 
        else:
            sample.update({"image": imarray, "ref-image": targetarray, "location_classes": one_hot_encode_locations(info["Ch"], location_mapping)}) 
        if self.return_info:
            sample["info"] = info

        return sample

class BFClassEmbedder(nn.Module):
    def __init__(self, include_location=False, include_ref_image=True, include_bf_mode=False, use_loc_embedding=True, image_embedding_model=None):
        super().__init__()
        self.include_location = include_location
        self.include_ref_image = include_ref_image
        self.include_bf_mode = include_bf_mode
        self.use_loc_embedding = use_loc_embedding
        if self.use_loc_embedding:
            self.loc_embedding = nn.Sequential(
                nn.Linear(len(location_mapping.keys()), 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
        
        if image_embedding_model:
            assert not isinstance(image_embedding_model, dict)
            self.image_embedding_model = instantiate_from_config(image_embedding_model)

    def forward(self, batch, key=None):
        conditions = dict()
        embed = []
        if self.include_bf_mode:
            embed.append(batch["BF_mode"])
        #if "seq_embed" in batch:
        #    embed.append(batch["seq_embed"])
        if self.include_location:
            if self.use_loc_embedding:
                embeder = self.loc_embedding.to(batch["location_classes"].device)
                embed.append(embeder(batch["location_classes"]))
                #print('Embeding locations: ', batch["location_classes"].shape,  batch["location_classes"] ,embeder(batch["location_classes"]))
            else:
                embed.append(batch["location_classes"])
                #print(batch["location_classes"])
            
        if "densent_avg" in batch:
            embed.append(batch["densent_avg"])
        if embed:
            conditions["c_crossattn"] = embed

        if self.include_ref_image:
            image = batch["ref-image"]
            assert image.shape[3] == 3
            image = rearrange(image, 'b h w c -> b c h w').contiguous()
            with torch.no_grad():
                img_embed = self.image_embedding_model.encode(image)
            if torch.any(torch.isnan(img_embed)):
                raise Exception("NAN values encountered in the image embedding")
            conditions["c_concat"] = [img_embed]
        return conditions

    def decode(self, c):
        assert self.include_ref_image
        condition_row = c['c_concat']
        assert len(condition_row) == 1
        with torch.no_grad():
            condition_rec_row = [self.image_embedding_model.decode(cond) for cond in condition_row]
        n_imgs_per_row = len(condition_rec_row)
        condition_rec_row = torch.stack(condition_rec_row)  # n_log_step, n_row, C, H, W
        condition_grid = rearrange(condition_rec_row, 'n b c h w -> b n c h w')
        condition_grid = rearrange(condition_grid, 'b n c h w -> (b n) c h w')
        condition_grid = make_grid(condition_grid, nrow=n_imgs_per_row)
        return condition_grid
        

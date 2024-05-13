from GPUtil import showUtilization as gpu_usage

from omegaconf import OmegaConf
import argparse, os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.evaluation import metrics3


############# HELPER FUNCTIONS #################
def predict_with_vqgan(x, model):
    h = model.encode(x) 
    ypred = model.decode(h) 
    return ypred

def normalize_array(array):
  return (array + 1)/2


def save_imgs(original_arrays, recon_arrays, imgdir, num_exs):
  #TO DO: Need to chang num_exs bc imgs will be saved in batches
  if len(original_arrays) == len(recon_arrays): #recons from autoencoder
    for i in range(1, num_exs+1):
      oringal_array = normalize_array(original_arrays[i])
      recon_array = normalize_array(recon_arrays[i])
      original_img = Image.fromarray(oringal_array).convert('RGB')
      recon_img = Image.fromarray(recon_array).convert('RGB')
      original_img.save(f"{imgdir}/real_{str(i)}.png")
      recon_img.save(f"{imgdir}/fake_{str(i)}.png")
  else: #samples from ldm
    assert len(recon_arrays) % len(original_arrays), "wrong number of samples "
    #TO DO save smampled imgs
  return
#################################################### 

def main(opt):
  config = OmegaConf.load(opt.config_path)
  checkpoint = opt.checkpoint
  savedir = opt.savedir
  mask = opt.mask

  #Constructing savedir names
  model_name = checkpoint.split("/checkpoints")[0].split("/")[-1]
  savedir = f"{savedir}/{model_name}"
  imgdir = f"{savedir}/images"
  
  os.makedirs(imgdir, exist_ok=True) 
  
  #Get Dataloader
  data_config = config['data']
  data = instantiate_from_config(data_config)
  data.prepare_data()
  data.setup()


  #Get Model
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = instantiate_from_config(config['model'])
  model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
  model = model.to(device)
  image_evaluator = metrics3.ImageEvaluator(device=device)
  if "ldm" in model_name:
          sampler = DDIMSampler(model)
  model.eval()



  batch_size = 16

  mses = [] #mean squared errors
  maes = [] #mean absolute errors
  ssims =[] #structural similarity index measure
  ious = [] #intersection over union
  pccs = [] #pearson correlation coefficient
  edists = [] #euclidean distances
  cdists = [] #cosine distances
  cell_areas = []

  with torch.no_grad():
    with model.ema_scope():
      for i in range(0, len(data.datasets["test"]), batch_size): #loop through number of batches
        print("Loop step: " + str(i))
        #get batch
        lim = min([i+batch_size, len(data.datasets["test"])])
        batch = [data.datasets["test"][j] for j in range(i, lim)]

        #Reformat batch to dict
        collated_batch = dict()
        for k in batch[0].keys():
          collated_batch[k] = [x[k] for x in batch]
          if isinstance(batch[0][k], (np.ndarray, np.generic)):
            collated_batch[k] = torch.tensor(collated_batch[k]).to(device)

        
        recons = predict_with_vqgan(torch.permute(collated_batch['image'], (0, 3, 1, 2)), model)

        if mask:
          mse_per_chan, mae_per_chan, ssim_per_chan, iou_per_chan, edist_per_chan, pcc_per_chan, cdist_per_chan, cell_area = image_evaluator.calc_metrics(samples=recons, targets=torch.permute(collated_batch['image'], (0, 3, 1, 2)), masks=collated_batch['cell-mask'])
        else:
          mse_per_chan, mae_per_chan, ssim_per_chan, iou_per_chan, edist_per_chan, pcc_per_chan, cdist_per_chan, __ = image_evaluator.calc_metrics(samples=recons, targets=torch.permute(collated_batch['image'], (0, 3, 1, 2)))
        mses.append(mse_per_chan)
        maes.append(mae_per_chan)
        ssims.append(ssim_per_chan)
        ious.append(iou_per_chan)
        pccs.append(pcc_per_chan)
        edists.append(edist_per_chan)
        cdists.append(cdist_per_chan)
        if mask:
          cell_areas.append(cell_area)


  mses = np.concatenate(mses, axis=0)
  maes = np.concatenate(maes, axis=0)
  ssims = np.concatenate(ssims, axis=0)
  ious = np.concatenate(ious, axis=0)
  pccs = np.concatenate(pccs, axis=0)
  edists = np.concatenate(edists, axis=0)
  cdists = np.concatenate(cdists, axis=0)    
  data = np.concatenate([mses, maes, ssims, ious, pccs, edists, cdists], axis=1)

  mses_cols = ["MSE chan " + str(i) for i in range(mses.shape[1])]
  mae_cols = ["MAE chan " + str(i) for i in range(mses.shape[1])]
  ssim_cols = ["SSIM chan " + str(i) for i in range(ssims.shape[1])]
  iou_cols = ["IOUs chan " + str(i) for i in range(ious.shape[1])]
  pcc_cols = ["PCC chan " + str(i) for i in range(pccs.shape[1])]
  edists_cols = ["Euclidean chan " + str(i) for i in range(edists.shape[1])]
  cdists_cols = ["Cosine chan " + str(i) for i in range(cdists.shape[1])]
  cols = mses_cols + mae_cols + ssim_cols + iou_cols + pcc_cols + edists_cols + cdists_cols
  
  output = pd.DataFrame(data, columns=cols)

  if mask:
    cell_areas = np.concatenate(cell_areas, axis=0)  
    output["Cell Area"] = cell_areas
  
  if mask:
    output.to_csv(f"data/basic/{model_name}.csv", index=False)
  else:
     output.to_csv(f"data/basic/{model_name}_unmasked.csv", index=False)

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Reconstruct test imgs with autoencoder or sample images from ldm. Example command: python analysis/reconstruct.py --config=configs/autoencoder/jump_autoencoder__r45__fov512.yaml --checkpoint=/scratch/users/zwefers/stable-diffusion/logs/2024-01-31T07-58-47_jump_autoencoder__r45__fov512/checkpoints/last.ckpt --savedir=/scratch/groups/emmalu/JUMP_HPA_validation/ --num_exs=100")
  parser.add_argument(
    "--config_path",
    type=str,
    nargs="?",
    help="the model config",
    )
  parser.add_argument(
    "--checkpoint",
    type=str,
    nargs="?",
    help="the model checkpoint",
  )
  parser.add_argument(
    "--mask",
    action="store_true"
  )
  parser.add_argument(
    "--savedir",
    type=str,
    default="/scratch/groups/emmalu/JUMP_HPA_validation/",
    nargs="?",
    help="where to save npy file",
  )
  parser.add_argument(
    "--num_exs",
    type=int,
    default=100,
    nargs="?",
    help="number of example images to save",
  )
  parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="unconditional guidance scale",
  )
  parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
  )

  opt = parser.parse_args()
  main(opt)
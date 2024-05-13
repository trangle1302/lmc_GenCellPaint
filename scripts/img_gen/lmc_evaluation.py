from omegaconf import OmegaConf
import argparse, os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, os.getcwd())
from ldm.util import instantiate_from_config
from ldm.evaluation import metrics3

############# HELPER FUNCTIONS #################
def predict_with_vqgan(x, model):
    h, _, _ = model.encode(x)
    ypred = model.decode(h)
    return ypred

def normalize_array(array):
  return (array + 1)/2


def save_imgs(original_arrays, recon_arrays, imgdir, num_exs):
  print(f'Real: {original_arrays.shape}, fake: {recon_arrays.shape}')
  #TO DO: Need to chang num_exs bc imgs will be saved in batches
  if len(original_arrays) == len(recon_arrays):
    for i in range(1, num_exs+1):
      oringal_array = normalize_array(original_arrays[i])
      recon_array = normalize_array(recon_arrays[i])
      original_img = Image.fromarray(oringal_array).convert('RGB')
      recon_img = Image.fromarray(recon_array).convert('RGB')
      original_img.save(f"{imgdir}/real_{str(i)}.png")
      recon_img.save(f"{imgdir}/fake_{str(i)}.png")
  elif original_arrays.shape[1] == 5: # 5ch
    pass
  else:
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
  imgdir = f"{savedir}/{model_name}/images"
  print(f'Saving to {imgdir}')
  os.makedirs(imgdir, exist_ok=True) 
  
  #Get Dataloader
  data = instantiate_from_config(config.data)
  data.prepare_data()
  data.setup()
  
  #Get Model
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = instantiate_from_config(config.model)
  model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
  model = model.to(device)
  image_evaluator = metrics3.ImageEvaluator(device=device)
  model.eval()



  batch_size = 20

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
      for i in range(0, len(data.datasets["validation"]), batch_size): #loop through number of batches
        lim = min([i+batch_size, len(data.datasets["validation"])])
        batch = [data.datasets["validation"][j] for j in range(i, lim)]

        #Reformat batch to dict
        collated_batch = dict()
        for k in batch[0].keys():
          collated_batch[k] = [x[k] for x in batch]
          if isinstance(batch[0][k], (np.ndarray, np.generic)):
            collated_batch[k] = torch.tensor(collated_batch[k]).to(device)

        
        recons = predict_with_vqgan(torch.permute(collated_batch['image'], (0, 3, 1, 2)), model)
        targets = torch.permute(collated_batch['ref-image'], (0, 3, 1, 2))
        
        if mask:
          raise NotImplementedError("find and import cell mask")
          # mse_per_chan, mae_per_chan, ssim_per_chan, iou_per_chan, edist_per_chan, pcc_per_chan, cdist_per_chan, cell_area = image_evaluator.calc_metrics(samples=recons, targets=targets, masks=collated_batch['cell-mask'])
        else:
          mse_per_chan, ssim_per_chan, mae_per_chan, pcc_per_chan, cos_sim, edist_per_chan, iou_per_chan = image_evaluator.calc_metrics(samples=recons, targets=targets)
        cdist_per_chan = torch.ones(cos_sim.shape) - cos_sim
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

  mses_cols = ["MSE chan " + str(i) for i in range(1, mses.shape[1]+1)]
  mae_cols = ["MAE chan " + str(i) for i in range(1, mses.shape[1]+1)]
  ssim_cols = ["SSIM chan " + str(i) for i in range(1, ssims.shape[1]+1)]
  iou_cols = ["IOUs chan " + str(i) for i in range(1, ious.shape[1]+1)]
  pcc_cols = ["PCC chan " + str(i) for i in range(1, pccs.shape[1]+1)]
  edists_cols = ["Euclidean chan " + str(i) for i in range(1, edists.shape[1]+1)]
  cdists_cols = ["Cosine chan " + str(i) for i in range(1, cdists.shape[1]+1)]
  cols = mses_cols + mae_cols + ssim_cols + iou_cols + pcc_cols + edists_cols + cdists_cols
  
  output = pd.DataFrame(data, columns=cols)
  print(output)
  if mask:
    cell_areas = np.concatenate(cell_areas, axis=0)  
    output["Cell Area"] = cell_areas
  
  if mask:
    output.to_csv(f"{savedir}/{model_name}.csv", index=False)
  else:
     output.to_csv(f"{savedir}/{model_name}_unmasked.csv", index=False)

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Reconstruct test imgs with vqgans and calculate metrics. Example command: python scripts/img_gen/lmc_evaluation.py --config=configs/bf_orgs_cellpaint.yaml --checkpoint=/scratch/users/tle1302/stable-diffusion/logs/2024-05-12T13-52-56_bf1_orgs_cellpaint/checkpoints/last.ckpt --savedir=/scratch/groups/emmalu/JUMP/vqgan_output --num_exs=100")
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
    default="/scratch/groups/emmalu/JUMP/vqgan_output",
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
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
  )

  opt = parser.parse_args()
  main(opt)
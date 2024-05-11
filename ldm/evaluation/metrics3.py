import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from skimage.filters import threshold_otsu
import cv2

from ldm.data import image_processing

class ImageEvaluator:
    def __init__(self, device):
        # self.fid = FrechetInceptionDistance(normalize=True)
        self.n = 1

    def clf_images(self, resized_cropped_images):
        outputs = self.loc_clf({'image': resized_cropped_images})
        logits = outputs['logits']
        probs = torch.sigmoid(logits)
        probs = probs.to('cpu').detach().numpy()
        feats = outputs['feature_vector']
        return probs, feats

    def calc_metrics(self, samples, targets, masks=None):
        '''The value range of the inputs should be between -1 and 1.'''
        bs, resolution = targets.size(0), targets.size(2) #bs=batch_size
        # assert targets.size() == (bs, 3, resolution, resolution)
        # assert samples.size() == (bs, 3, resolution, resolution)
        assert image_processing.is_between_minus1_1(targets)
        #note samples don't come out of model perfect between [-1,1], they exceed this range slightly
        samples = torch.clip(samples, min=-1, max=1) #is this necessary? --> yes
        targets = torch.clip(targets, min=-1, max=1)

        #normalize
        targets = (targets + 1) / 2 * 255# [0, 255]
        samples = (samples + 1) / 2 * 255# [0, 255]
        
        # Calculate MSE and SSIM
        if masks is not None:
            transform = torchvision.transforms.Resize((resolution, resolution), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            masks = transform(masks)
            masks = torch.tensor(masks > 0, dtype=torch.int8) #binarize cell mask
            assert masks.size() == (bs,resolution, resolution)
        # print(samples.shape, targets.shape)
        mses = F.mse_loss(samples, targets, reduction='none')
        maes = (samples - targets).abs()
        ssims = torchmetrics.functional.image.ssim.ssim(samples, targets, reduction='none')
        vsamples = samples - samples.mean(dim=[2,3])[:, :, None, None]
        vtargets = targets - targets.mean(dim=[2,3])[:, :, None, None]
        covar = torch.mul(vtargets, vsamples)
        cos_numerator = samples * targets
        samples_sqrd = samples * samples
        targets_sqrd = targets * targets
        cell_area = None
        mses_for_edist = mses.clone()
            
        if masks is not None:

            mse_per_chan = [0 for i in range(bs)]
            mae_per_chan = [0 for i in range(bs)]
            ssim_per_chan = [0 for i in range(bs)]
            cell_area = [0 for i in range(bs)]
            for i in range(bs): #apply cell mask

                #count number of pixels part of cell, will only take mean/sum over these pixels
                num_pixels = torch.count_nonzero(masks[i]).to('cpu').detach()
                cell_area[i] = num_pixels
                num_pixels_ssim = torch.count_nonzero(masks[i][5:251, 5:251])

                mse_per_chan[i] = (torch.sum(torch.mul(mses[i], masks[i].repeat(3, 1, 1)), dim=[1,2])/num_pixels).to('cpu').detach().numpy()
                mae_per_chan[i] = (torch.sum(torch.mul(maes[i], masks[i].repeat(3, 1, 1)), dim=[1,2])/num_pixels).to('cpu').detach().numpy()
                ssim_per_chan[i] = (torch.sum(torch.mul(ssims[i], masks[i][5:251, 5:251].repeat(3, 1, 1)), dim=[1,2])/num_pixels_ssim).to('cpu').detach().numpy()

                vsamples[i] = torch.mul(vsamples[i], masks[i].repeat(3, 1, 1))
                vtargets[i] = torch.mul(vtargets[i], masks[i].repeat(3, 1, 1))
                covar[i] = torch.mul(covar[i], masks[i].repeat(3, 1, 1))
                cos_numerator[i] = torch.mul(cos_numerator[i], masks[i].repeat(3, 1, 1))
                samples_sqrd[i] = torch.mul(samples_sqrd[i], masks[i].repeat(3, 1, 1))
                targets_sqrd[i] = torch.mul(targets_sqrd[i], masks[i].repeat(3, 1, 1))
            
            #converting output to tensors
            mse_per_chan = np.array(mse_per_chan)
            mae_per_chan = np.array(mae_per_chan)
            ssim_per_chan = np.array(ssim_per_chan)

        else: #Not masking cells --> take mean over all pixels
            mse_per_chan = mses.mean(dim=[2,3]).to('cpu').detach().numpy()
            mae_per_chan = maes.mean(dim=[2,3]).to('cpu').detach().numpy()
            ssim_per_chan = ssims.mean(dim=[2,3]).to('cpu').detach().numpy()
       
    
        #Euclidean Distance
        edists_per_chan = torch.sqrt(torch.sum(mses_for_edist, dim=[2,3])).to('cpu').detach().numpy()

        #pearson correlation coefficient
        covar = torch.sum(covar, dim=[2,3])
        sample_var = torch.sum(torch.square(vsamples), dim=[2,3])
        target_var = torch.sum(torch.square(vtargets), dim=[2,3])
        pcc_per_chan = torch.div(covar, torch.sqrt(sample_var*target_var)).to('cpu').detach().numpy()

        #cosine difference = 1 - cosine similiarity
        cos_numerator = torch.sum(cos_numerator, dim=[2,3])
        samples_sqrd = torch.sum(samples_sqrd, dim=[2,3]) 
        targets_sqrd = torch.sum(targets_sqrd, dim=[2,3]) 
        cos_sim = torch.div(cos_numerator, torch.sqrt(samples_sqrd*targets_sqrd)).to('cpu').detach().numpy()
        #cos_per_chan = torch.ones(cos_sim.shape) - cos_sim

        # Calculate intersection over union
        #thresh = 0.5 # changed to otsu
        n_ch = ssim_per_chan.shape[1]
        iou_per_chan = np.zeros(ssim_per_chan.shape) #[bs, n_ch]
        for i in range(bs):
            #get otsu thresholds for each channel
            thresholds = []
            for j in range(n_ch):
                channel_img = samples[i][j].clone().to('cpu').detach().numpy()
                thresh = threshold_otsu(channel_img)
                thresholds.append(torch.ones_like(samples[i][j].repeat(1, 1, 1))*thresh)
            thresholds = torch.cat(thresholds, dim=0).to("cuda")
            if masks is not None:
                sampled_binary = torch.tensor(torch.mul(samples[i], masks[i].repeat(n_ch, 1, 1)) > thresholds, dtype=torch.int8)
                target_binary = torch.tensor(torch.mul(targets[i], masks[i].repeat(n_ch, 1, 1)) > thresholds, dtype=torch.int8)
            else:
                sampled_binary = torch.tensor(samples[i] > thresholds, dtype=torch.int8)
                target_binary = torch.tensor(targets[i] > thresholds, dtype=torch.int8)
            intersection_per_chan = torch.sum(torch.logical_and(sampled_binary, target_binary), dim=(1, 2))
            union_per_chan = torch.sum(torch.logical_or(sampled_binary, target_binary), dim=(1, 2))
            iou_per_chan[i] = torch.div(intersection_per_chan, union_per_chan).to('cpu').detach().numpy()
        #print('Metric shapes: ', mse_per_chan.shape, ssim_per_chan.shape, cos_sim.shape, edists_per_chan.shape, iou_per_chan.shape)
        return mse_per_chan, ssim_per_chan, mae_per_chan, pcc_per_chan, cos_sim, edists_per_chan, iou_per_chan

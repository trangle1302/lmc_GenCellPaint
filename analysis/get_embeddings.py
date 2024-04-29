from omegaconf import OmegaConf
import os
import torch
import tifffile
import numpy as np
from einops import rearrange
import cv2
from skimage import exposure
import pandas as pd
import umap
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
from ldm.data import image_processing


def read_image(location):
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray()  # Extract image data
        metadata = tif.ome_metadata  # Get the existing metadata in a DICT
    return image_data, metadata


def rescale_2_98(arr, l_thres=2, h_thres=99.8):
    p2, p98 = np.percentile(arr, (l_thres, h_thres))
    arr = exposure.rescale_intensity(
        arr, in_range=(p2, p98), out_range=(0, 255)
    ).astype(np.uint8)
    return arr


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def prepare_input1(img_paths):
    imarray = []
    for img_path in img_paths:
        arr, _ = read_image(img_path)
        arr = rescale_2_98(arr)
        arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_LINEAR)
        arr = np.stack([arr, arr, arr], axis=2)
        assert image_processing.is_between_0_255(arr)
        arr = image_processing.convert_to_minus1_1(arr)
        imarray.append(arr)
    imarray = np.stack(imarray, axis=0)
    print(imarray.shape)
    imarray = torch.from_numpy(imarray)
    imarray = rearrange(imarray, "b h w c -> b c h w").contiguous()
    assert imarray.shape == (len(img_paths), 3, 256, 256)
    return imarray


def run_umap(feats, n_neighbors=15, n_components=3, min_dist=0.1, metric="euclidean"):
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
    )
    embedded_points = umap_model.fit_transform(feats)
    return embedded_points


def plot_umap(embedded_points, labels, plot_mean=True, save_path="tmp.png"):
    unique_labels = np.unique(labels)
    # Generate colors for each label
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "yellow",
        "cyan",
        "magenta",
        "lime",
        "pink",
        "teal",
        "lavender",
        "brown",
        "beige",
        "maroon",
        "orchid",
        "olive",
        "coral",
        "navy",
        "grey",
        "indigo",
        "violet",
        "turquoise",
        "salmon",
        "tan",
        "skyblue",
        "plum",
        "gold",
        "silver",
        "charcoal",
    ]
    n_components = embedded_points.shape[1]
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            # print(mask)
            plt.scatter(
                embedded_points[mask, 0],
                embedded_points[mask, 1],
                label=label,
                color=colors[i],
                s=10,
                alpha=0.1,
            )
            if plot_mean:
                try:
                    from statistics import geometric_mean

                    d1_m = geometric_mean(embedded_points[mask, 0])
                    d2_m = geometric_mean(embedded_points[mask, 1])
                    plt.scatter(
                        d1_m,
                        d2_m,
                        label=label + "_m",
                        color=colors[i],
                        marker=">",
                        s=50,
                    )
                except:
                    print(label, np.sum(mask))
        plt.legend()
        # plt.colorbar(label='Labels')
        plt.title("UMAP Embedding (2D)")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.savefig(save_path)
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, label in enumerate(unique_labels):
            mask = labels == label
            scatter = ax.scatter(
                embedded_points[mask, 0],
                embedded_points[mask, 1],
                embedded_points[mask, 2],
                label=label,
                color=colors[i],
                alpha=0.2,
                s=10,
            )
        ax.set_title("UMAP Embedding (3D)")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")
        plt.legend()
        # plt.colorbar(scatter, label='Labels')
        plt.savefig(save_path)
    else:
        print("Number of components should be either 2 or 3.")


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def generate_features(feat_compressed_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config1 = OmegaConf.load(
        "/scratch/users/tle1302/stable-diffusion/configs/autoencoder/lmc_autoencoder_ActinMitoTubulin.f512_0.5.yaml"
    )
    checkpoint1 = "/scratch/users/tle1302/stable-diffusion/logs/2024-03-27T01-24-26_lmc_autoencoder_ActinMitoTubulin.f512_0.5/checkpoints/best.ckpt"
    # Get Dataloader
    model = instantiate_from_config(config1["model"])
    model.load_state_dict(
        torch.load(checkpoint1, map_location="cpu")["state_dict"], strict=False
    )
    model = model.to(device)
    model.eval()

    metadf = pd.read_csv("/scratch/groups/emmalu/lightmycell/meta.csv")
    img_modality_dict = (
        metadf[metadf.Type == "BF"].set_index("Image_id").to_dict()["Ch"]
    )
    metadf = metadf[metadf.Type != "BF"]
    metadf = metadf[metadf.Image_id != "image_2542"]
    print(f"Generating embeddings of {metadf.shape[0]} images")
    metadf["img_paths"] = [
        "/".join(("/scratch/groups/emmalu/lightmycell/Images", r.Study, r.Id))
        for i, r in metadf.iterrows()
    ]
    feats = []
    loss_ = []
    with torch.no_grad():
        with model.ema_scope():
            for df_ in chunker(metadf, 16):
                x = prepare_input1(df_.img_paths.tolist())
                h, _, _ = model.encode(x.to(device))
                ypred = model.decode(h)
                loss = (x - ypred.to("cpu")).abs()
                loss_.append(loss.mean())
                feats.append(h.to("cpu").detach().numpy())
    print(f"MAE={loss.mean()}")
    feats = np.vstack(feats)  # , axis=1)
    print(feats.shape)
    tl = [img_modality_dict[f] for f in metadf.Image_id]
    np.savez(
        feat_compressed_path,
        feats=feats,
        tl=tl,
        study=metadf.Study.tolist(),
        ch=metadf.Ch.tolist(),
    )
    return feats, tl, metadf.Study.tolist(), metadf.Ch.tolist()


def main():
    feat_compressed_path = "/scratch/groups/emmalu/lightmycell/vqgan_orgs_embed.npz"
    if not os.path.exists(feat_compressed_path):
        feats, tl, studies, chs = generate_features(feat_compressed_path)
    else:
        compressed = np.load(feat_compressed_path)
        feats = compressed["feats"]
        tl = compressed["tl"]
        studies = compressed["study"]
        chs = compressed["ch"]
    # print(f'Embedding dim {feats.shape}, {feats.reshape(feats.shape[0], -1).shape}')
    # print(tl, chs)
    not_nu = chs != "Nucleus"
    print(not_nu)
    print(feats.reshape(feats.shape[0], -1)[not_nu, :].squeeze().shape)
    tmp = feats.reshape(feats.shape[0], -1)[not_nu, :].squeeze()
    for nc in [2, 3]:
        embedded_points = run_umap(tmp, n_components=nc)
        print(embedded_points.shape)
        tl_ch = [f"{it1}_{it2}" for it1, it2 in zip(chs, tl)]
        # print(tl_ch[np.where(not_nu)[0]])
        plot_dir = "/scratch/groups/emmalu/lightmycell/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_umap(
            embedded_points,
            np.array(tl_ch)[not_nu],
            save_path=f"{plot_dir}/umap_{nc}d_tlch.png",
        )
        plot_umap(
            embedded_points,
            tl[not_nu],
            save_path=f"{plot_dir}/umap_{nc}d_transmittedlight.png",
        )
        plot_umap(
            embedded_points,
            chs[not_nu],
            save_path=f"{plot_dir}/umap_{nc}d_orgchannels.png",
        )
        plot_umap(
            embedded_points,
            studies[not_nu],
            plot_mean=False,
            save_path=f"{plot_dir}/umap_{nc}d_studies.png",
        )


if __name__ == "__main__":
    main()

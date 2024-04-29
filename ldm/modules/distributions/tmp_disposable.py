input_channels = ['BF','Nucleus']
output_channels = 'org'
df['io'] = [('ref' if f in ['BF','Nucleus'] else 'org') for f in df.Type ]
df = df[df.Image_id !="image_2542"] # Study_29/image_2542 is empty on all channels
print(df.io.value_counts()) #print(df.shape, set(input_channels + output_channels))

imgs_keep = set(df.Image_id)
for ch in set(input_channels): # BF, Nucleus
    imgs_k = df.groupby('Image_id').agg({'Type': set })
    imgs_k['Type'] = [(set(list(t)[0].split(',')) if len(t)==1 else t) for t in imgs_k.Type]
    imgs_k = imgs_k.iloc[[(ch in f) for f in imgs_k.Type]]
    imgs_k = imgs_k.iloc[[len({'Mitochondria','Tubulin','Actin'}.intersection(f))>0 for f in imgs_k.Type]]
    imgs_keep = imgs_keep.intersection(imgs_k.index.tolist())
df = df[df.Image_id.isin(imgs_keep)].drop_duplicates()

df = df.iloc[np.repeat(df[df.Ch=='Actin'].index, 65)]
df = df.iloc[np.repeat(df[df.Ch=='Tubulin'].index, 8)]


train_data, test_data = train_test_split(df, test_size=0.05, stratify=df.Study, random_state=42)
#if refine:
    
df["split"] = ["train" if idx in train_data.index else "validation" for idx in df.index]

python main.py -t -b configs/autoencoder/lmc_bf_Tubulin.yaml --gpus=0, --resume=/scratch/users/tle1302/stable-diffusion/logs/2024-04-10T11-38-48_lmc_bf_Tubulin/checkpoints/last.ckpt


results = []
results.append(['Id','TL','Nu2','Nu998','Mi2','Mi998','Ac2','Ac998','Tu2','Tu998'])
for i,r in df.iterrows():
    imid = r.Id
    img = tifffile.imread(f'./datasets/{imid}')
    line = [imid, r.TL]
    for ch in range(1,5):
        img_ch = img[:,:,ch]
        (l, h) = np.percentile(img_ch, (2, 99.8))
        line += [l,h]
    print(line)
    results.append(line)
    breakme

/scratch/users/tle1302/miniconda3/pkgs/libiconv-1.17-hd590300_2
/scratch/users/tle1302/miniconda3/pkgs/lerc-4.0.0-h27087fc_0
/scratch/users/tle1302/miniconda3/pkgs/keyutils-1.6.1-h166bdaf_0
/scratch/users/tle1302/miniconda3/pkgs/freetype-2.12.1-h267a509_2
/scratch/users/tle1302/miniconda3/pkgs/colorama-0.4.6-pyhd8ed1ab_0
/scratch/users/tle1302/miniconda3/pkgs/natsort-8.4.0-pyhd8ed1ab_0
/scratch/users/tle1302/miniconda3/pkgs/six-1.16.0-pyh6c4a22f_0
/scratch/users/tle1302/miniconda3/pkgs/zipp-3.17.0-pyhd8ed1ab_0
/scratch/users/tle1302/miniconda3/pkgs/patsy-0.5.6-pyhd8ed1ab_0
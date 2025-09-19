import os
from glob import glob
from tqdm import tqdm

data_list =sorted(glob("data/open3dhoi_p1/*"))
with open("tmp.txt", "w") as f:
    for path in tqdm(data_list):
        sample = path.split('/')[-1]

        if not os.path.isfile(f"output/fitting/{sample}.jpg/object_mesh.obj"):
            f.write(f"python -m optim.fit --input_path data/open3dhoi_p1_new/{sample}/{sample}.jpg --cfg optim/cfg/fit.yaml\n")

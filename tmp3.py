import os
from glob import glob
from tqdm import tqdm
import numpy as np
import trimesh

with open("tmp2.txt") as f:
    sample_list = f.readlines()

sample_list = [sample.split('\n')[0] for sample in sample_list]

for sample in sample_list[:]:
    os.system(f"rm -rf /root/InteractVLM/data/open3dhoi_p1/{sample}/sam_inp_objs")
    # os.system(f"rm /root/InteractVLM/data/open3dhoi_p1/{sample}/{sample}_oafford_vertices.npz")

# data_list =sorted(glob("data/open3dhoi_p1/*"))
# with open("tmp.txt", "w") as f:
#     for path in tqdm(data_list):
#         sample = path.split('/')[-1]

#         if not os.path.isfile(f"data/open3dhoi_p1/{sample}/sam_inp_objs/obj_render_color_frontleft.png"):
#             f.write(f"python -m optim.fit --input_path data/open3dhoi_p1_new/{sample}/{sample}.jpg --cfg optim/cfg/fit.yaml\n")

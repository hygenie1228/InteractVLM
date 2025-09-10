from glob import glob
from tqdm import tqdm
import os
import cv2
import json
import numpy as np
import trimesh

def vis_bbox(img, bbox):
    img = img.copy()
    color, thickness = (0, 255, 0), 5

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        pos1 = (x_min, y_min)
        pos2 = (x_min, y_max)
        pos3 = (x_max, y_min)
        pos4 = (x_max, y_max)

        img = cv2.line(img, pos1, pos2, color, thickness) 
        img = cv2.line(img, pos1, pos3, color, thickness) 
        img = cv2.line(img, pos2, pos4, color, thickness) 
        img = cv2.line(img, pos3, pos4, color, thickness) 
    elif len(bbox) == 8:
        kps_line = obj_dict.skeleton
        for l in range(len(kps_line)):
            i1, i2 = kps_line[l][0], kps_line[l][1]
            
            p1 = bbox[i1,0].astype(np.int32), bbox[i1,1].astype(np.int32)
            p2 = bbox[i2,0].astype(np.int32), bbox[i2,1].astype(np.int32)
            cv2.line(img, p1, p2, color, thickness)

    return img


def apply_uniform_pad_and_shift(bbox, p, shift_x=0.0, shift_y=0.0, clip_hw=None):
    """
    uniform pad와 shift를 적용해 새 bbox를 생성.
    """
    x1,y1,x2,y2 = np.asarray(bbox, float).ravel()
    w  = x2 - x1;  h  = y2 - y1
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2

    new_w = w * (1 + 2*p)
    new_h = h * (1 + 2*p)
    new_cx = cx + shift_x * w
    new_cy = cy + shift_y * h

    nx1 = new_cx - new_w/2; ny1 = new_cy - new_h/2
    nx2 = new_cx + new_w/2; ny2 = new_cy + new_h/2
    out = np.array([nx1, ny1, nx2, ny2], dtype=float)

    if clip_hw is not None:
        H, W = clip_hw
        out[0] = np.clip(out[0], 0, W-1)
        out[2] = np.clip(out[2], 0, W-1)
        out[1] = np.clip(out[1], 0, H-1)
        out[3] = np.clip(out[3], 0, H-1)
    return out



img = cv2.imread("/root/InteractVLM/data/optim_data/tennis_racket__000000041045.jpg")
data = json.load(open("/root/InteractVLM/data/optim_data/human_detection.json"))
mask = np.array(data['mask'])
bbox = np.array(data['bbox'])


with np.load("/root/InteractVLM/data/optim_data/osx_human3.npz") as f:  # f: NpzFile (dict-like)
    data = {k: f[k] for k in f.files}  # numpy 배열들이 값

bbox_2 = data['bbox_2']

bbox_22 = apply_uniform_pad_and_shift(bbox, 0.25, 0, 0)



img = vis_bbox(img, bbox_22.tolist())
cv2.imwrite('debug.png', img)




# del data['smpl_faces']
# del data['smpl_vertices']
del data['bbox_1']
# del data['bbox_2']
del data['bbox_3']
del data['joint_img']
del data['smplx_joint_proj']
del data['og_cam']
del data['og_vertices']
del data['smplx_root_pose']
np.savez("/root/InteractVLM/data/optim_data/osx_human2.npz", **data)  


import pdb; pdb.set_trace()

print(data.keys())



# img = vis_bbox(img, bbox.tolist())
# cv2.imwrite('debug.png', img)

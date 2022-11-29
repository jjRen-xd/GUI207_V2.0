# coding=utf-8
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# raw_data_path = "E:/workspace/Pycharm_workspace/Project_207/data/"
# raw_data_path = "/home/hp/Desktop/Project_207/data/"

raw_data_path = "E:/GitHub/dev_zyx/GUI207_V2.0/db/datasets/falseHRRPmat_1x128"  # (128, 1)
# raw_data_path = "/media/hp/新加卷/data/DD_data/tohrrp/hrrp_22june/" # (256, 1)
# raw_data_path = "/media/hp/新加卷/data/DD_data/fea/fea_data/"   # (39, 1)

data_path = "./data/"

# class_name = ["Ball_bottom_cone", "Big_ball", "Cone", "DT", "Small_ball"]

class_name = ["bigball", "DT", "Moxiu", "smallball", "taper", "WD"]

# file_name = ["bigball1_hrrp", "bigball2_hrrp", "DT_hrrp", "Moxiu_hrrp", "smallball1_hrrp", "smallball2_hrrp", "taper1_hrrp", "taper2_hrrp", "WD_19_hrrp"]


model_path = "./checkpoint/"

log_path = "./result/"





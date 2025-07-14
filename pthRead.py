import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

print("torch sees {} GPUs".format(torch.cuda.device_count()))
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
device=torch.device("cuda:0")
# 1) 载入文件
ckpt = torch.load(
    "./weights/sam_vit_l_0b3195.pth",
    map_location=device
)

# 2) 查看载入对象的类型和内容概要
print(type(ckpt))        # <class 'dict'> 或其它
if isinstance(ckpt, dict):
    print("Keys in checkpoint:", ckpt.keys())
else:
    print("Loaded object:", ckpt)

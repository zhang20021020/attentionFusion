import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as da
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils2 import *
from torch.autograd import Variable
from IPython.display import clear_output
from UNetFormer_MMSAM_heatmap import UNetFormer as MFNet

# from UNet2  import UNetFormer as MFNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
import os
from datetime import datetime

# 获取当前时间并格式化输出
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
# 只让程序“看见”第 4、5、6 号卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_seed(42)

# net = MFNet(num_classes=N_CLASSES).cuda()
device = torch.device("cuda:0")  # 逻辑第 0 块 → 物理卡 6
net = MFNet(num_classes=N_CLASSES).to(device)
# 打印逻辑 GPU 和物理卡信息
print(f"Using torch device: {device}")
if device.type == 'cuda':
    # 当前逻辑 GPU 索引
    cur_idx = torch.cuda.current_device()
    # 对应的物理卡名称
    name = torch.cuda.get_device_name(cur_idx)
    print(f" → Logical GPU index: {cur_idx}")
    print(f" → Physical GPU name: {name}")
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
# for name, param in net.image_encoder.named_parameters():
#     # if "Adapter" not in name:
#     if "lora_" not in name:
#     # if "lora_" not in name and "Adapter" not in name:
#         params1 += param.nelement()
#     else:
#         params2 += param.nelement()
# print('ImgEncoder:   ', params1)
# # print('Adapter:       ', params2)
# print('Lora: ', params2)
# # print('Adapter_Lora: ', params2)
# print('Others: ', params-params1-params2)

# for name, parms in net.named_parameters():
#     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))

# params = 0
# for name, param in net.sam.prompt_encoder.named_parameters():
#     params += param.nelement()
# print('prompt_encoder: ', params)

# params = 0
# for name, param in net.sam.mask_decoder.named_parameters():
#     params += param.nelement()
# print('mask_decoder: ', params)

# print(net)

print("training : ", train_ids)
print("testing : ", test_ids)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

base_lr = 6e-4
weight_decay = 5e-4
# base_lr = 0.01
# weight_decay = 0.0005
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in
                       test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids),
                                       leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                # Do the inference
                outs = net(image_patches, dsm_patches, mode='Test')
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.83
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        start_time = time.time()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, dsm, mode='Train')
            ce = F.cross_entropy(output, target, weight=weights)

            # —— 2) Dice Loss ——
            dl = dice_loss(output, target)
            loss = ce + dl
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0 and e >= 20:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    print('MIoU_best: ', MIoU_best)

def build_heatmap(feat, out_size):
    # feat: [1, C, h, w]
    fmap = torch.norm(feat, p=2, dim=1, keepdim=True)
    fmap = F.interpolate(fmap, size=out_size,
                         mode='bilinear', align_corners=False)
    fmap = fmap.squeeze().cpu().numpy()
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
    fmap = np.uint8(255 * fmap)
    return cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
def build_heatmap_from_feat(feat, out_size):
    """
    feat: Tensor, shape [1, C, h, w]
    out_size: (H, W), e.g. (256, 256)
    """
    # 1) 通道聚合（推荐 L2-norm）
    feat_map = torch.norm(feat, p=2, dim=1, keepdim=True)  # [1, 1, h, w]

    # 2) 上采样到 patch 尺度
    feat_up = F.interpolate(
        feat_map,
        size=out_size,
        mode='bilinear',
        align_corners=False
    )  # [1, 1, H, W]

    # 3) 转 numpy + 归一化
    feat_np = feat_up.squeeze(0).squeeze(0).cpu().numpy()
    feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-6)
    feat_uint8 = (feat_np * 255).astype(np.uint8)

    # 4) 伪彩色
    heat_color = cv2.applyColorMap(feat_uint8, cv2.COLORMAP_JET)
    return heat_color

def export_vis_patches_256_with_heatmap(
        net,
        ids,
        split_name='test',
        patch_size=256,
        out_root='./vis_256'):
    """
    对给定 ids（如 test_ids）：
      - 切 256×256 patch
      - 输出：RGB / DSM / GT / Pred
      - 额外输出：叠加 feature heatmap 的 RGB / DSM 图
    """
    net.eval()
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, f'{DATASET}_{split_name}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[*] 导出目录: {out_dir}')
    global_idx = 0


    for tile_id in ids:
            print(f'--- 处理影像: {tile_id} ---')

            # 1) 读取 RGB
            if DATASET == 'Potsdam':
                rgb_full = io.imread(DATA_FOLDER.format(tile_id))[:, :, :3].astype('float32') / 255.0
            else:
                rgb_full = io.imread(DATA_FOLDER.format(tile_id)).astype('float32') / 255.0
            H, W, _ = rgb_full.shape

            # 2) 读取 DSM 并归一化
            dsm_full = io.imread(DSM_FOLDER.format(tile_id)).astype('float32')
            dsm_min, dsm_max = dsm_full.min(), dsm_full.max()
            dsm_full = (dsm_full - dsm_min) / (dsm_max - dsm_min + 1e-6)

            # 3) 读取 GT（彩色）并转 label
            gt_color_full = io.imread(LABEL_FOLDER.format(tile_id))
            gt_full = convert_from_color(gt_color_full).astype('int64')

            # 4) 256×256 滑窗
            step = patch_size
            window_size = (patch_size, patch_size)
            patch_idx = 0
            for (x, y, w, h) in sliding_window(rgb_full,
                                               step=step,
                                               window_size=window_size):
                rgb_patch = rgb_full[x:x + w, y:y + h]      # H×W×3
                dsm_patch = dsm_full[x:x + w, y:y + h]      # H×W
                gt_patch = gt_full[x:x + w, y:y + h]        # H×W

                # 5) 构造输入 tensor
                rgb_tensor = torch.from_numpy(
                    rgb_patch.transpose(2, 0, 1)
                ).unsqueeze(0).float().to(device)           # 1×3×256×256

                dsm_tensor = torch.from_numpy(
                    dsm_patch
                ).unsqueeze(0).unsqueeze(0).float().to(device)  # 1×1×256×256

                # 6) 正常前向，得到预测
                # logits = net(rgb_tensor, dsm_tensor, mode='Test')  # 1×C×256×256
                # pred_patch = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                # 预测可以 no_grad（可选）
                with torch.no_grad():
                    logits = net(rgb_tensor, dsm_tensor, mode='Test')
                    pred_patch = logits.argmax(dim=1).squeeze(0).cpu().numpy()




                # 8) 原始图转 uint8
                rgb_img = np.clip(rgb_patch * 255.0, 0, 255).astype(np.uint8)   # 256×256×3
                dsm_img = np.clip(dsm_patch * 255.0, 0, 255).astype(np.uint8)   # 256×256



                # 10) 上色 GT 和 Pred
                gt_color_patch   = convert_to_color(gt_patch)
                pred_color_patch = convert_to_color(pred_patch)

                # 11) 保存所有图像
                patch_idx += 1
                idx_str = f'{tile_id}_x{x:04d}_y{y:04d}_p{patch_idx:03d}'

                rgb_path = os.path.join(out_dir, f'{idx_str}_rgb.png')
                dsm_path = os.path.join(out_dir, f'{idx_str}_dsm.png')
                gt_path = os.path.join(out_dir, f'{idx_str}_gt.png')
                pred_path = os.path.join(out_dir, f'{idx_str}_pred.png')


                Image.fromarray(rgb_img).save(rgb_path)
                Image.fromarray(dsm_img).save(dsm_path)
                Image.fromarray(gt_color_patch).save(gt_path)
                Image.fromarray(pred_color_patch).save(pred_path)





if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('F:/code/attentionFusion/resultsv/UNetformer_epoch25_0.8458.pth'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=24)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/UNetformer_epoch44_0.857365.pth'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=24)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)

elif MODE == 'VIS':
    # 1) 加载你想可视化的权重
    if DATASET == 'Vaihingen':
        weight_path = './resultsv/UNetformer_epoch25_0.8458.pth'
    else:  # Potsdam
        weight_path = './resultsp/UNetformer_epoch44_0.857365.pth'

    print(f'[*] 加载权重: {weight_path}')
    net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    net.eval()

    # 2) 对 test_ids 导出可视化 patch
    export_vis_patches_256_with_heatmap(
        net,
        ids=test_ids,
        split_name='test',
        patch_size=256,
        out_root='./vis_256'
    )

    # 如果你也想导出训练集：
    # export_vis_patches_256(net,
    #                        ids=train_ids,
    #                        split_name='train',
    #                        patch_size=256,
    #                        out_root='./vis_256')

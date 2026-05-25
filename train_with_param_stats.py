import numpy as np
from glob import glob
# from tqdm import tqdm_notebook as tqdm  # 已移除进度条
from sklearn.metrics import confusion_matrix
import time
from tqdm import tqdm
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
# from UNetFormer_MMSAM import UNetFormer as MFNet
from DoubleSwinMambnClean import UNetFormer_TwoModal as MFNet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

set_seed(42)
# Helper: pad or crop a patch to fixed size
def pad_patch(patch, target_h, target_w):
    """Pad or crop patch to (target_h, target_w)."""
    if patch.ndim == 3:
        c, h, w = patch.shape
        if h == target_h and w == target_w:
            return patch
        out = np.zeros((c, target_h, target_w), dtype=patch.dtype)
        out[:, :h, :w] = patch[:,:target_h,:target_w]
        return out
    else:
        h, w = patch.shape
        if h == target_h and w == target_w:
            return patch
        out = np.zeros((target_h, target_w), dtype=patch.dtype)
        out[:h, :w] = patch[:target_h,:target_w]
        return out

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("torch sees {} GPUs".format(torch.cuda.device_count()))
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

net = MFNet(num_classes=N_CLASSES).cuda()


# ============================================================
# 参数量统计功能
# 适配当前 UNetFormer_TwoModal:
#   encoder.rgb_backbone / encoder.dsm_backbone
#   encoder.to256_* / encoder.fuse
#   decoder.low_proj / mid_proj / high_proj / b3 / fuse_mid / fuse_low / seg_head
# 运行后会在当前目录保存 params.csv
# ============================================================
def _count_params(module, trainable_only=False):
    """统计一个模块的参数量。"""
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _get_module(model, module_path):
    """
    根据字符串路径安全获取模块。
    例如：_get_module(net, "encoder.rgb_backbone")
    如果模块不存在，返回 None，避免代码报错。
    """
    cur = model
    for name in module_path.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _fmt_num(n):
    return f"{n:,}"


def _fmt_m(n):
    return f"{n / 1e6:.4f}M"


def summarize_model_params(model, save_path="./params.csv"):
    """
    统计模型参数量，并保存为 CSV。

    CSV 字段说明：
    - module: 模块名称
    - total_params: 模块总参数量
    - trainable_params: 可训练参数量
    - frozen_params: 冻结参数量
    - total_params_M: 百万参数形式
    - trainable_params_M: 百万参数形式
    - ratio_in_model_%: 该模块参数占全模型总参数比例
    """
    import csv

    total_params = _count_params(model, trainable_only=False)
    trainable_params = _count_params(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    rows = []

    def add_row(module_name, module_or_count):
        if isinstance(module_or_count, int):
            total = module_or_count
            trainable = module_or_count
        else:
            total = _count_params(module_or_count, trainable_only=False)
            trainable = _count_params(module_or_count, trainable_only=True)

        frozen = total - trainable
        ratio = 100.0 * total / total_params if total_params > 0 else 0.0

        rows.append({
            "module": module_name,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "total_params_M": round(total / 1e6, 6),
            "trainable_params_M": round(trainable / 1e6, 6),
            "ratio_in_model_%": round(ratio, 4),
        })

    # 1. 全模型统计
    add_row("all_model", model)

    # 2. encoder / decoder 总体
    encoder = _get_module(model, "encoder")
    decoder = _get_module(model, "decoder")
    add_row("encoder_total", encoder)
    add_row("decoder_total", decoder)

    # 3. 双分支 backbone
    rgb_backbone = _get_module(model, "encoder.rgb_backbone")
    dsm_backbone = _get_module(model, "encoder.dsm_backbone")
    add_row("rgb_backbone", rgb_backbone)
    add_row("dsm_backbone", dsm_backbone)

    # 4. encoder 中除 backbone 之外的部分：通道投影层 + 融合模块等
    encoder_total = _count_params(encoder)
    rgb_backbone_total = _count_params(rgb_backbone)
    dsm_backbone_total = _count_params(dsm_backbone)
    encoder_non_backbone = encoder_total - rgb_backbone_total - dsm_backbone_total
    add_row("encoder_non_backbone", encoder_non_backbone)

    # 5. RGB / DSM 三尺度通道投影层
    rgb_proj_paths = [
        "encoder.to256_rgb_low",
        "encoder.to256_rgb_mid",
        "encoder.to256_rgb_high",
    ]
    dsm_proj_paths = [
        "encoder.to256_dsm_low",
        "encoder.to256_dsm_mid",
        "encoder.to256_dsm_high",
    ]

    rgb_proj_total = sum(_count_params(_get_module(model, p)) for p in rgb_proj_paths)
    dsm_proj_total = sum(_count_params(_get_module(model, p)) for p in dsm_proj_paths)
    add_row("rgb_projection_layers", rgb_proj_total)
    add_row("dsm_projection_layers", dsm_proj_total)

    # 6. encoder 融合模块，例如 CBAMFusion
    add_row("encoder_fusion_module", _get_module(model, "encoder.fuse"))

    # 7. decoder 细分模块
    decoder_parts = [
        ("decoder_low_proj", "decoder.low_proj"),
        ("decoder_mid_proj", "decoder.mid_proj"),
        ("decoder_high_proj", "decoder.high_proj"),
        ("decoder_mamba_block_b3", "decoder.b3"),
        ("decoder_up_high_to_mid", "decoder.up_high_to_mid"),
        ("decoder_up_mid_to_low", "decoder.up_mid_to_low"),
        ("decoder_skip_fuse_mid", "decoder.fuse_mid"),
        ("decoder_skip_fuse_low", "decoder.fuse_low"),
        ("decoder_seg_head", "decoder.seg_head"),
    ]
    for name, path in decoder_parts:
        add_row(name, _get_module(model, path))

    # 8. 其他参数：理论上当前模型一般为 0；如果不为 0，说明还有未统计到的模块
    counted_major = encoder_total + _count_params(decoder)
    other_params = total_params - counted_major
    add_row("others_not_in_encoder_decoder", other_params)

    # 9. 优化器实际可训练参数量参考
    add_row("optimizer_trainable_total", trainable_params)

    # 控制台打印
    print("\n" + "=" * 80)
    print("Model Parameter Statistics")
    print("=" * 80)
    print(f"{'Module':35s} {'Params':>18s} {'Trainable':>18s} {'Ratio':>10s}")
    print("-" * 80)

    for r in rows:
        print(
            f"{r['module']:35s} "
            f"{_fmt_num(r['total_params']):>18s} "
            f"{_fmt_num(r['trainable_params']):>18s} "
            f"{r['ratio_in_model_%']:>9.2f}%"
        )

    print("-" * 80)
    print(f"Total Params    : {_fmt_num(total_params)} ({_fmt_m(total_params)})")
    print(f"Trainable Params: {_fmt_num(trainable_params)} ({_fmt_m(trainable_params)})")
    print(f"Frozen Params   : {_fmt_num(frozen_params)} ({_fmt_m(frozen_params)})")
    print(f"Saved CSV       : {save_path}")
    print("=" * 80 + "\n")

    # 保存 CSV
    fieldnames = [
        "module",
        "total_params",
        "trainable_params",
        "frozen_params",
        "total_params_M",
        "trainable_params_M",
        "ratio_in_model_%",
    ]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


param_rows = summarize_model_params(net, save_path="./params.csv")

print("training : ", train_ids)
print("testing  : ", test_ids)
train_set    = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

base_lr = 6e-4
weight_decay = 5e-4
# base_lr    = 0.01
params_dict = dict(net.named_parameters())
params_list = []
for key, value in params_dict.items():
    if '_D' in key:
        params_list += [{'params': [value], 'lr': base_lr}]
    else:
        params_list += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    if DATASET == 'Potsdam':
        test_images = (
            1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32')
            for id in test_ids
        )
    else:
        test_images = (
            1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32')
            for id in test_ids
        )
    test_dsms     = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels   = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8')  for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id)))      for id in test_ids)
    all_preds = []
    all_gts   = []

    with torch.no_grad():
        for img, dsm, gt, gt_e in zip(test_images, test_dsms, test_labels, eroded_labels) :
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
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

    accuracy = metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([g.ravel() for g in all_gts]).ravel()
    )
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses      = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_     = 0
    MIoU_best = 0.82
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        start_time = time.time()
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, dsm, mode='Train')
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb  = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt   = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1
            del data, target, loss

        if e % save_epoch == 0 and e>=10:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
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


if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/UNetformer_epoch31_0.8423784622411172'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.load_state_dict(torch.load('./resultsp/UNetformer_epoch36_0.8640.pth'), strict=False)
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)

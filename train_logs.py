import os
import csv
from datetime import datetime
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


def bytes_to_mb(x):
    return x / 1024 / 1024


def get_cuda_memory():
    """Return current CUDA memory stats in MB."""
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'max_allocated_mb': 0.0,
            'max_reserved_mb': 0.0,
        }
    torch.cuda.synchronize()
    return {
        'allocated_mb': bytes_to_mb(torch.cuda.memory_allocated()),
        'reserved_mb': bytes_to_mb(torch.cuda.memory_reserved()),
        'max_allocated_mb': bytes_to_mb(torch.cuda.max_memory_allocated()),
        'max_reserved_mb': bytes_to_mb(torch.cuda.max_memory_reserved()),
    }


def append_csv_row(csv_path, fieldnames, row):
    """Append one CSV row and create the header on first write."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def count_params(module, trainable_only=False):
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def safe_get_module(model, module_path):
    current = model
    for name in module_path.split('.'):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def save_param_log(model, csv_path):
    total_params = count_params(model)
    trainable_params = count_params(model, trainable_only=True)
    rows = []

    def add_row(module_name, module_or_count):
        if module_or_count is None:
            return
        if isinstance(module_or_count, int):
            total = module_or_count
            trainable = module_or_count
        else:
            total = count_params(module_or_count)
            trainable = count_params(module_or_count, trainable_only=True)
        rows.append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'module': module_name,
            'total_params': total,
            'trainable_params': trainable,
            'frozen_params': total - trainable,
            'total_params_M': round(total / 1e6, 6),
            'trainable_params_M': round(trainable / 1e6, 6),
            'ratio_in_model_%': round(100.0 * total / total_params, 4) if total_params > 0 else 0.0,
        })

    add_row('all_model', model)
    encoder = safe_get_module(model, 'encoder')
    decoder = safe_get_module(model, 'decoder')
    add_row('encoder_total', encoder)
    add_row('decoder_total', decoder)
    add_row('rgb_backbone', safe_get_module(model, 'encoder.rgb_backbone'))
    add_row('dsm_backbone', safe_get_module(model, 'encoder.dsm_backbone'))

    encoder_total = count_params(encoder)
    backbone_total = (
        count_params(safe_get_module(model, 'encoder.rgb_backbone'))
        + count_params(safe_get_module(model, 'encoder.dsm_backbone'))
    )
    add_row('encoder_non_backbone', encoder_total - backbone_total)

    for module_name, module_path in [
        ('encoder_fusion_module', 'encoder.fuse'),
        ('decoder_mamba_block_b3', 'decoder.b3'),
        ('decoder_mamba_mid', 'decoder.mamba_mid'),
        ('decoder_up1', 'decoder.up1'),
        ('decoder_up2', 'decoder.up2'),
        ('decoder_up3', 'decoder.up3'),
        ('decoder_pre_conv', 'decoder.pre_conv'),
        ('decoder_head', 'decoder.head'),
        ('decoder_seg_head', 'decoder.seg_head'),
    ]:
        add_row(module_name, safe_get_module(model, module_path))

    add_row('others_not_in_encoder_decoder', total_params - encoder_total - count_params(decoder))
    add_row('optimizer_trainable_total', trainable_params)

    fieldnames = [
        'time', 'module', 'total_params', 'trainable_params', 'frozen_params',
        'total_params_M', 'trainable_params_M', 'ratio_in_model_%'
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Parameter log saved to: {csv_path}')


def save_run_info(txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'model: {MODEL}\n')
        f.write(f'mode: {MODE}\n')
        f.write(f'dataset: {DATASET}\n')
        f.write(f'window_size: {WINDOW_SIZE}\n')
        f.write(f'batch_size: {BATCH_SIZE}\n')
        f.write(f'stride_size: {Stride_Size}\n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'save_epoch: {save_epoch}\n')
        f.write(f'train_ids: {train_ids}\n')
        f.write(f'test_ids: {test_ids}\n')
        f.write(f'cuda_available: {torch.cuda.is_available()}\n')
        if torch.cuda.is_available():
            f.write(f'gpu_count: {torch.cuda.device_count()}\n')
            f.write(f'gpu_name: {torch.cuda.get_device_name(torch.cuda.current_device())}\n')


RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = os.path.join('logs', f'{MODEL}_{DATASET}_{RUN_ID}')
RUN_INFO_TXT = os.path.join(LOG_DIR, 'run_info.txt')
PARAM_CSV = os.path.join(LOG_DIR, 'params.csv')
ITER_LOG_CSV = os.path.join(LOG_DIR, 'iter_log.csv')
EPOCH_LOG_CSV = os.path.join(LOG_DIR, 'epoch_log.csv')
TEST_LOG_CSV = os.path.join(LOG_DIR, 'test_log.csv')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("torch sees {} GPUs".format(torch.cuda.device_count()))
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

net = MFNet(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()

params_rgb = sum(p.numel() for p in net.encoder.rgb_backbone.parameters())
params_dsm = sum(p.numel() for p in net.encoder.dsm_backbone.parameters())
params_encoder = params_rgb + params_dsm
params_non_backbone = (
    sum(p.numel() for p in net.parameters())
    - params_encoder
)

print(f"All Params        : {params:,}")
print(f"RGB Backbone      : {params_rgb:,}")
print(f"DSM Backbone      : {params_dsm:,}")
print(f"Encoder (Total)   : {params_encoder:,}")
print(f"Decoder + Others  : {params_non_backbone:,}")
print(f"Log directory     : {LOG_DIR}")
save_run_info(RUN_INFO_TXT)
save_param_log(net, PARAM_CSV)

# params1 = 0
# params2 = 0
# for name, param in net.encoder.rgb_backbone.named_parameters():
#     if "lora_" not in name:
#         params1 += param.nelement()
#     else:
#         params2 += param.nelement()
# print('ImgEncoder:   ', params1)
# print('Lora:         ', params2)
# print('Others:       ', params - params1 - params2)

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
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        epoch_batches = 0
        epoch_samples = 0
        epoch_compute_time = 0.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            iter_start_time = time.time()
            optimizer.zero_grad()
            output = net(data, dsm, mode='Train')
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            iter_time = time.time() - iter_start_time

            loss_value = loss.item()
            losses[iter_] = loss_value
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            with torch.no_grad():
                train_acc = 100.0 * (output.argmax(dim=1) == target).float().mean().item()
            batch_size_now = data.size(0)
            batch_fps = batch_size_now / max(iter_time, 1e-12)

            epoch_loss_sum += loss_value
            epoch_acc_sum += train_acc
            epoch_batches += 1
            epoch_samples += batch_size_now
            epoch_compute_time += iter_time

            if iter_ % 100 == 0:
                clear_output()
                mem = get_cuda_memory()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}\tFPS: {:.2f}\tGPU max_alloc: {:.2f} MB'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss_value, train_acc, batch_fps, mem['max_allocated_mb']))

                append_csv_row(
                    ITER_LOG_CSV,
                    [
                        'time', 'epoch', 'iter', 'batch_idx', 'loss', 'mean_loss_100',
                        'train_acc', 'iter_time_sec', 'batch_size', 'fps_images_per_sec',
                        'lr', 'allocated_mb', 'reserved_mb', 'max_allocated_mb', 'max_reserved_mb'
                    ],
                    {
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'epoch': e,
                        'iter': iter_,
                        'batch_idx': batch_idx,
                        'loss': round(loss_value, 6),
                        'mean_loss_100': round(float(mean_losses[iter_]), 6),
                        'train_acc': round(train_acc, 6),
                        'iter_time_sec': round(iter_time, 6),
                        'batch_size': batch_size_now,
                        'fps_images_per_sec': round(batch_fps, 4),
                        'lr': optimizer.param_groups[0]['lr'],
                        'allocated_mb': round(mem['allocated_mb'], 2),
                        'reserved_mb': round(mem['reserved_mb'], 2),
                        'max_allocated_mb': round(mem['max_allocated_mb'], 2),
                        'max_reserved_mb': round(mem['max_reserved_mb'], 2),
                    }
                )
            iter_ += 1
            del data, dsm, target, output, loss

        epoch_time = time.time() - start_time
        mem_epoch = get_cuda_memory()
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        avg_acc = epoch_acc_sum / max(epoch_batches, 1)
        train_fps_wall = epoch_samples / max(epoch_time, 1e-12)
        train_fps_compute = epoch_samples / max(epoch_compute_time, 1e-12)

        append_csv_row(
            EPOCH_LOG_CSV,
            [
                'time', 'epoch', 'avg_loss', 'avg_train_acc', 'epoch_time_sec',
                'epoch_compute_time_sec', 'train_samples', 'train_fps_wall_images_per_sec',
                'train_fps_compute_images_per_sec', 'lr', 'allocated_mb', 'reserved_mb',
                'max_allocated_mb', 'max_reserved_mb'
            ],
            {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': e,
                'avg_loss': round(avg_loss, 6),
                'avg_train_acc': round(avg_acc, 6),
                'epoch_time_sec': round(epoch_time, 3),
                'epoch_compute_time_sec': round(epoch_compute_time, 3),
                'train_samples': epoch_samples,
                'train_fps_wall_images_per_sec': round(train_fps_wall, 4),
                'train_fps_compute_images_per_sec': round(train_fps_compute, 4),
                'lr': optimizer.param_groups[0]['lr'],
                'allocated_mb': round(mem_epoch['allocated_mb'], 2),
                'reserved_mb': round(mem_epoch['reserved_mb'], 2),
                'max_allocated_mb': round(mem_epoch['max_allocated_mb'], 2),
                'max_reserved_mb': round(mem_epoch['max_reserved_mb'], 2),
            }
        )
        print(f"Epoch {e}: avg_loss={avg_loss:.6f}, avg_acc={avg_acc:.4f}, "
              f"fps={train_fps_compute:.2f} img/s, log={EPOCH_LOG_CSV}")

        if e % save_epoch == 0 and e>=10:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            append_csv_row(
                TEST_LOG_CSV,
                ['time', 'epoch', 'miou', 'best_miou_before_update', 'test_time_sec', 'stride'],
                {
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'epoch': e,
                    'miou': round(float(MIoU), 6),
                    'best_miou_before_update': round(float(MIoU_best), 6),
                    'test_time_sec': round(test_time - train_time, 3),
                    'stride': Stride_Size,
                }
            )
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), './resultsp/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU
    print('MIoU_best: ', MIoU_best)


if MODE == 'Train':
    print(f'Training logs will be saved to: {LOG_DIR}')
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

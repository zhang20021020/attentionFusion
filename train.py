import numpy as np
from glob import glob
# from tqdm import tqdm_notebook as tqdm  # 已移除进度条
from sklearn.metrics import confusion_matrix
import time
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
from UNetFormer_MMSAM import UNetFormer as MFNet
from segmentation_loss import segmentation_loss
from checkpoint_utils import checkpoint_state_dict


import random

SEED = 42  # 你可以选择任意整数作为种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


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



print("torch sees {} GPUs".format(torch.cuda.device_count()))
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

if MODEL_OPTIONS['use_category_supervision'] and LOSS_OPTIONS['category_weight'] <= 0:
    raise ValueError('Category prediction heads require a positive category_weight.')
if MODE == 'Test' and not CHECKPOINT_PATH:
    raise ValueError('Set CHECKPOINT_PATH in utils.py to the trained experiment checkpoint.')
net = MFNet(num_classes=N_CLASSES, **MODEL_OPTIONS).cuda()
experiment_name = MODEL + '_' + '_'.join(
    '{}{}'.format(name, int(MODEL_OPTIONS[key])) for name, key in (
        ('sup', 'use_category_supervision'), ('conf', 'use_confidence_fusion'),
        ('pan', 'use_category_pan'), ('geo', 'use_dsm_geometry'),
        ('n2', 'use_n2_category_gate'), ('gbr', 'use_dsm_geometry_branch')))
if MODEL_OPTIONS['use_dsm_geometry_branch']:
    experiment_name += '_gw' + 'x'.join(str(k) for k in MODEL_OPTIONS['dsm_geometry_windows'])
experiment_name += '_cw{:g}_pw{:g}_seed{}'.format(
    LOSS_OPTIONS['category_weight'], LOSS_OPTIONS['pan_weight'], SEED)
print('Experiment:', experiment_name)
print('Model options:', MODEL_OPTIONS)
print('Loss weights:', LOSS_OPTIONS)



params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
for name, param in net.image_encoder.named_parameters():
    if "lora_" not in name:
        params1 += param.nelement()
    else:
        params2 += param.nelement()
print('ImgEncoder:   ', params1)
print('Lora:         ', params2)
print('Others:       ', params - params1 - params2)

print("training : ", train_ids)
print("testing  : ", test_ids)
train_set    = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

base_lr    = 0.01
params_dict = dict(net.named_parameters())
params_list = []
for key, value in params_dict.items():
    if '_D' in key:
        params_list += [{'params': [value], 'lr': base_lr}]
    else:
        params_list += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
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
        for img, dsm, gt, gt_e in zip(test_images, test_dsms, test_labels, eroded_labels):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            norm_dsm = normalize_dsm(dsm)

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                dsm_patches = [np.copy(norm_dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)



                outs = net(image_patches, dsm_patches, mode='Test')
                outs = outs.data.cpu().numpy()

                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del outs

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

    iter_         = 0
    MIoU_best     = float('-inf')
    epoch_losses  = []

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        start_time = time.time()

        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            model_output = net(data, dsm, mode='Train')
            loss, loss_terms = segmentation_loss(model_output, target, weight=weights,
                                                  **LOSS_OPTIONS)
            output = model_output[0] if isinstance(model_output, tuple) else model_output
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
                print('Loss terms:', {name: round(value.detach().item(), 6)
                                      for name, value in loss_terms.items()})
            iter_ += 1
            del data, dsm, target, loss, model_output, output, loss_terms

        # 每个epoch结束，计算当前epoch的平均loss
        epoch_mean_loss = np.mean(losses[max(0, iter_ - len(train_loader)):iter_])
        epoch_losses.append(epoch_mean_loss)

        if e % save_epoch == 0 and e > 15:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            test_time = time.time()
            print("Test time: {:.3f} seconds".format(test_time - train_time))
            if MIoU > MIoU_best:
                result_dir = './resultsv' if DATASET == 'Vaihingen' else './resultsp'
                os.makedirs(result_dir, exist_ok=True)
                torch.save(dict(model_state_dict=net.state_dict(), model_options=MODEL_OPTIONS,
                                loss_options=LOSS_OPTIONS, num_classes=N_CLASSES, epoch=e,
                                miou=float(MIoU), seed=SEED, dataset=DATASET,
                                train_ids=train_ids, test_ids=test_ids, window_size=WINDOW_SIZE),
                           os.path.join(result_dir, '{}_epoch{}_{}.pth'.format(
                               experiment_name, e, MIoU)))
                MIoU_best = MIoU

    print('MIoU_best: ', MIoU_best)

    # 所有epoch结束后，绘制loss图
    plt.figure()
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label='Epoch Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss_curve.png')
    plt.show()


if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = checkpoint_state_dict(checkpoint, MODEL_OPTIONS, N_CLASSES)
    net.load_state_dict(state_dict, strict=True)
    if DATASET == 'Vaihingen':
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsv/inference_UNetFormer_{}_tile_{}.png'.format('huge', id_), img)

    elif DATASET == 'Potsdam':
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave('./resultsp/inference_UNetFormer_{}_tile_{}.png'.format('base', id_), img)

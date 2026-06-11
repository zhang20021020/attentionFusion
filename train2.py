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
from utils2 import *
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


class LegacyManBaBlock(nn.Module):
    """ManBaBlock layout used by checkpoints trained before decoder residuals."""
    def __init__(self, mamba, in_chs, hidden_ch, out_ch, drop=0.1):
        super().__init__()
        self.mamba = mamba
        self.conv_ffn = nn.Sequential(
            nn.Conv2d(in_chs, hidden_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.conv_ffn(self.mamba(x))


def checkpoint_state_dict(checkpoint):
    """Extract and normalize a state dict saved by common PyTorch wrappers."""
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint does not contain a valid state_dict.')
    if checkpoint and all(key.startswith('module.') for key in checkpoint):
        checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
    return checkpoint


def load_visualization_weights(net, weight_path, map_location='cuda:0'):
    """Load current or pre-residual mamba decoder checkpoints."""
    state_dict = checkpoint_state_dict(torch.load(weight_path, map_location=map_location))
    legacy_key = 'decoder.b3.conv_ffn.3.weight'
    current_b3 = net.decoder.b3

    if legacy_key in state_dict:
        checkpoint_out_ch = state_dict[legacy_key].shape[0]
        current_out_ch = current_b3.conv_ffn[3].weight.shape[0]
        if checkpoint_out_ch != current_out_ch:
            in_chs = state_dict['decoder.b3.conv_ffn.0.weight'].shape[1]
            hidden_ch = state_dict['decoder.b3.conv_ffn.0.weight'].shape[0]
            print(
                '[*] Detected pre-residual decoder checkpoint: '
                f'conv_ffn output {checkpoint_out_ch} instead of {current_out_ch}.'
            )
            net.decoder.b3 = LegacyManBaBlock(
                mamba=current_b3.mamba,
                in_chs=in_chs,
                hidden_ch=hidden_ch,
                out_ch=checkpoint_out_ch
            ).to(next(net.parameters()).device)

    incompatible = net.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f'[!] Missing checkpoint keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys}')
    if incompatible.unexpected_keys:
        print(f'[!] Unexpected checkpoint keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys}')
    return incompatible


def normalize_cam(cam):
    """Normalize a BxHxW CAM tensor to [0, 1] per image."""
    cam = F.relu(cam)
    flat = cam.flatten(1)
    cam_min = flat.min(dim=1).values[:, None, None]
    cam_max = flat.max(dim=1).values[:, None, None]
    return (cam - cam_min) / (cam_max - cam_min + 1e-6)


def gradcam_from_feature(feature, gradient, out_size):
    """Build a Grad-CAM from one encoder feature map and its gradient."""
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature).sum(dim=1)
    cam = normalize_cam(cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=out_size,
        mode='bilinear',
        align_corners=False
    )
    return cam.squeeze(1)


def predict_with_branch_gradcams(net, rgb_tensor, dsm_tensor):
    """Return logits plus Grad-CAMs for the mamba RGB and DSM low-level branches."""
    features = {}

    def save_feature(name):
        def hook(_module, _inputs, output):
            features[name] = output
        return hook

    rgb_handle = net.encoder.to256_rgb_low.register_forward_hook(save_feature('rgb'))
    dsm_handle = net.encoder.to256_dsm_low.register_forward_hook(save_feature('dsm'))
    try:
        net.zero_grad(set_to_none=True)
        logits = net(rgb_tensor, dsm_tensor, mode='VIS')
        pred = logits.argmax(dim=1)
        selected_score = logits.gather(1, pred.unsqueeze(1)).mean()
        rgb_grad, dsm_grad = torch.autograd.grad(
            selected_score,
            (features['rgb'], features['dsm']),
            retain_graph=False,
            create_graph=False
        )
        out_size = rgb_tensor.shape[-2:]
        rgb_cam = gradcam_from_feature(features['rgb'], rgb_grad, out_size)
        dsm_cam = gradcam_from_feature(features['dsm'], dsm_grad, out_size)
        return logits.detach(), rgb_cam.detach(), dsm_cam.detach()
    finally:
        rgb_handle.remove()
        dsm_handle.remove()


def export_vis_patches_256_with_heatmap(
        net,
        ids,
        split_name='test',
        patch_size=256,
        out_root='./vis_256_mamba'):
    """Export 256x256 RGB, DSM, GT, prediction and branch Grad-CAM images."""
    net.eval()
    out_dir = os.path.join(out_root, f'{DATASET}_{split_name}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[*] Export directory: {out_dir}')

    for tile_id in ids:
        print(f'--- Processing tile: {tile_id} ---')
        if DATASET == 'Potsdam':
            rgb_full = io.imread(DATA_FOLDER.format(tile_id))[:, :, :3].astype('float32') / 255.0
        else:
            rgb_full = io.imread(DATA_FOLDER.format(tile_id)).astype('float32') / 255.0

        dsm_full = io.imread(DSM_FOLDER.format(tile_id)).astype('float32')
        dsm_full = (dsm_full - dsm_full.min()) / (dsm_full.max() - dsm_full.min() + 1e-6)
        gt_full = convert_from_color(io.imread(LABEL_FOLDER.format(tile_id))).astype('int64')

        patch_idx = 0
        for x, y, w, h in sliding_window(
                rgb_full,
                step=patch_size,
                window_size=(patch_size, patch_size)):
            rgb_patch = rgb_full[x:x + w, y:y + h]
            dsm_patch = dsm_full[x:x + w, y:y + h]
            gt_patch = gt_full[x:x + w, y:y + h]

            rgb_tensor = torch.from_numpy(
                rgb_patch.transpose(2, 0, 1)
            ).unsqueeze(0).float().cuda()
            dsm_tensor = torch.from_numpy(
                dsm_patch
            ).unsqueeze(0).unsqueeze(0).float().cuda()

            logits, rgb_cam, dsm_cam = predict_with_branch_gradcams(
                net, rgb_tensor, dsm_tensor
            )
            pred_patch = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            rgb_cam = rgb_cam.squeeze(0).cpu().numpy()
            dsm_cam = dsm_cam.squeeze(0).cpu().numpy()

            rgb_img = np.clip(rgb_patch * 255.0, 0, 255).astype(np.uint8)
            dsm_img = np.clip(dsm_patch * 255.0, 0, 255).astype(np.uint8)
            dsm_color = cv2.cvtColor(dsm_img, cv2.COLOR_GRAY2BGR)
            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            rgb_heat = cv2.applyColorMap((rgb_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            dsm_heat = cv2.applyColorMap((dsm_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            rgb_heat_overlay = cv2.addWeighted(rgb_bgr, 0.5, rgb_heat, 0.5, 0)
            dsm_heat_overlay = cv2.addWeighted(dsm_color, 0.5, dsm_heat, 0.5, 0)

            patch_idx += 1
            prefix = f'{tile_id}_x{x:04d}_y{y:04d}_p{patch_idx:03d}'
            Image.fromarray(rgb_img).save(os.path.join(out_dir, f'{prefix}_rgb.png'))
            Image.fromarray(dsm_img).save(os.path.join(out_dir, f'{prefix}_dsm.png'))
            Image.fromarray(convert_to_color(gt_patch)).save(os.path.join(out_dir, f'{prefix}_gt.png'))
            Image.fromarray(convert_to_color(pred_patch)).save(os.path.join(out_dir, f'{prefix}_pred.png'))
            cv2.imwrite(os.path.join(out_dir, f'{prefix}_rgb_heat.png'), rgb_heat)
            cv2.imwrite(os.path.join(out_dir, f'{prefix}_dsm_heat.png'), dsm_heat)
            cv2.imwrite(os.path.join(out_dir, f'{prefix}_rgb_heat_overlay.png'), rgb_heat_overlay)
            cv2.imwrite(os.path.join(out_dir, f'{prefix}_dsm_heat_overlay.png'), dsm_heat_overlay)


if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/UNetformer_epoch38_0.8379.pth'), strict=False)
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

elif MODE == 'VIS':
    print(f'[*] Loading weights: {VIS_WEIGHT_PATH}')
    load_visualization_weights(net, VIS_WEIGHT_PATH, map_location='cuda:0')
    export_vis_patches_256_with_heatmap(
        net,
        ids=test_ids,
        split_name='test',
        patch_size=256,
        out_root='./vis_256_mamba'
    )

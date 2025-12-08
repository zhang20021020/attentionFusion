import os
import torch
import numpy as np
import cv2
from skimage import io
from UNetFormer_MMSAM_heatmap_my import UNetFormer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# =================== 加载函数 =====================
def load_rgb_image(path):
    img = io.imread(path).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[:, :, :3]
    img = img / 255.0
    return img   # 不转 tensor，保持 numpy 用于 overlay


def load_dsm_image(path):
    dsm = io.imread(path).astype(np.float32)
    if dsm.ndim == 3:
        dsm = dsm[:, :, 0]
    return dsm  # numpy


# ============ Sliding Window ================
def sliding_window(image, step, window_size):
    H, W = image.shape[-2:]
    coords = []
    for y in range(0, H - window_size + 1, step):
        for x in range(0, W - window_size + 1, step):
            coords.append((y, x))
    return coords


# =================== 主程序 =====================
if __name__ == "__main__":

    # 1. 输入路径

    dsm_path = "/home/p24030854116/datebase/ISPRS_dataset/Potsdam/1_DSM_normalisation/dsm_potsdam_4_10_normalized_lastools.jpg"
    rgb_path = "/home/p24030854116/datebase/ISPRS_dataset/Potsdam/4_Ortho_RGBIR/top_potsdam_4_10_RGBIR.tif"

    save_dir = "resultsp"
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(rgb_path))[0]

    # 2. 加载整图（用于 overlay）
    orig_rgb = load_rgb_image(rgb_path)  # (6000,6000,3)
    dsm_np = load_dsm_image(dsm_path)    # (6000,6000)

    H, W, C = orig_rgb.shape

    # 3. 加载模型
    net = UNetFormer().cuda()
    net.load_state_dict(torch.load("./resultsp/UNetformer_epoch41_0.8573.pth"))
    net.eval()

    # 4. 建立 Heatmap 画布
    full_heatmap = np.zeros((H, W), dtype=np.float32)
    counter = np.zeros((H, W), dtype=np.float32)

    window = 256
    stride = 128

    coords = sliding_window(orig_rgb.transpose(2,0,1), step=stride, window_size=window)

    for (y, x) in coords:
        # 取 patch
        img_patch = orig_rgb[y:y + window, x:x + window, :]
        dsm_patch = dsm_np[y:y + window, x:x + window]

        img_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
        dsm_tensor = torch.from_numpy(dsm_patch).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.enable_grad():
            _, hm, _ = net(img_tensor, dsm_tensor, mode='Heatmap')  # hm: (16,16)

        # 确保是 float32 numpy
        hm = hm.astype(np.float32)

        # ⭐ 将 16×16 的 heatmap 插值放大到 256×256
        hm_resized = cv2.resize(hm, (window, window), interpolation=cv2.INTER_LINEAR)

        # 放入画布（两者都是 256×256 就不会报错了）
        full_heatmap[y:y + window, x:x + window] += hm_resized
        counter[y:y + window, x:x + window] += 1

    # 平均融合
    full_heatmap /= counter
    full_heatmap = np.nan_to_num(full_heatmap)

    # 保存 heatmap 灰度
    hm_path = os.path.join(save_dir, f"{base_name}_full_heatmap.png")
    cv2.imwrite(hm_path, (full_heatmap * 255).astype(np.uint8))

    # ================= Overlay =================
    heatmap_color = cv2.applyColorMap((full_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = (orig_rgb * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)

    overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print("全图 Heatmap 已保存：", hm_path)
    print("叠加图已保存：", overlay_path)

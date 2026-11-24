# 植被判别辅助监督

本版在五通道 DSM 几何分支上增加仅训练时使用的二分类头，直接监督几何增强后的 `res2y`（1/8 尺度）。现有几何编码器、双尺度残差注入、类别融合、PAN、解码器及其损失权重保持原配置。

## 数据流与损失

```text
DSM 五通道描述 → 轻量编码器 → 双尺度残差注入
                                    │
                           增强后的 res2y
                            ├─ 原有融合、PAN、解码器
                            └─ 1×1 Conv(256→64) → GN(8) → GELU
                               → 1×1 Conv(64→2) → 植被辅助损失
```

辅助头共有 16,642 个可训练参数。输出两个原始 logits，分别对应低矮植被和树木。训练时将 logits 双线性上采样到标签大小；只在原标签 ID=2、3 且非 ignore_index 的像素上计算交叉熵，映射为二分类 ID=0、1。损失按有效植被像素数取平均，没有额外类别权重或 Focal 项。无植被裁剪块返回保留计算图的零损失。

总损失为：

`L = Lmain + 0.1*(Lopt + LDSM) + 0.05*LPAN + 0.02*Lvegetation`

新的 `vegetation_weight=0.02` 独立于原有 `category_weight` 和 `pan_weight`。训练日志中的 `vegetation` 是未加权的辅助 CE，与其他 loss terms 的打印口径相同。

辅助头输入没有 detach，梯度可传回 DSM 特征、1/8 几何注入和几何编码器。由于原有残差投影零初始化，第一次优化先更新投影；后续优化才将该损失的梯度传入几何编码器和门控。这与之前的初始化设计一致。

`net.eval()` 时不执行辅助头，主模型仍只返回六类分割 logits，推理不需要标签。辅助头参数仍保存在检查点中，以便严格加载；训练和保存会增加参数，主推理路径不增加辅助头计算。

## 当前配置及 Potsdam 实验

本地 `utils.py` 已启用：

```python
MODEL_OPTIONS['use_vegetation_aux'] = True
LOSS_OPTIONS['vegetation_weight'] = 0.02
```

这两个键已写入现有字典，无需在其他地方重复添加。模型构造函数的默认值仍为 False，兼容直接构造旧模型的代码；训练入口显式传入 `MODEL_OPTIONS`。

本地原有数据集为 Vaihingen、窗口为 `(9,25)`，本次没有切换数据集或窗口。继续用户已完成的 Potsdam `(17,45)` 实验时，在 `utils.py` 对应配置位置设置：

```python
MODE = 'Train'
DATASET = 'Potsdam'
MODEL_OPTIONS = dict(
    use_category_supervision=True,
    use_confidence_fusion=True,
    use_category_pan=True,
    use_dsm_geometry=False,
    use_n2_category_gate=False,
    use_dsm_geometry_branch=True,
    dsm_geometry_windows=(17, 45),
    use_vegetation_aux=True,
)
LOSS_OPTIONS = dict(category_weight=0.1, pan_weight=0.05, decoder_weight=0.0,
                    ignore_index=6, vegetation_weight=0.02)
```

在数据和 SAM 预训练文件齐备的训练环境运行 `python train.py`。`MODE='Train'` 仍采用原训练初始化流程，`CHECKPOINT_PATH` 不会自动使其从上次最优模型续训。

Potsdam 训练期间验证仍为 stride=128，独立 Test 模式仍为 stride=32，本次没有改变评估协议。与 86.3467% 对照时，应使用同样的 stride=32 独立测试。

## 消融和检查点

- 保持几何分支开启，设置 `use_vegetation_aux=False` 且 `vegetation_weight=0.0`，恢复本次新增辅助监督前的结构和损失。模型头开关和正权重不一致时，训练入口报错。
- 本版植被辅助头要求 `use_dsm_geometry_branch=True`，类别编号使用 ISPRS 的 2、3；它不依赖第一项类别引导模块必须开启，可用于后续独立消融。
- 新实验文件名包含 `veg1` 和 `vw0.02`，完整 Potsdam 示例前缀为 `UNetformer_sup1_conf1_pan1_geo0_n20_gbr1_veg1_gw17x45_cw0.1_pw0.05_vw0.02_seed42`。新开关、窗口和损失权重都会随检查点保存。
- 测试新模型时保持 `use_vegetation_aux=True`，将 `MODE='Test'`，并指定本次新权重。`net.eval()` 自动跳过辅助头；不要通过关闭构造开关来跳过它，否则与检查点结构不匹配。
- 测试以前的几何分支模型时使用 `use_vegetation_aux=False`，建议同时设 `vegetation_weight=0.0`。旧检查点缺失此开关时自动按 False 解释，仍严格加载参数。
- 旧权重不包含新辅助头，不能开启新头后直接作为已经训练的新模型加载。

## 文件与验证

- `vegetation_supervision.py`：轻量辅助头、标签掩码和植被 CE。
- `UNetFormer_MMSAM.py`：训练时读取增强后的 res2y，返回 vegetation 辅助输出。
- `segmentation_loss.py`：接入独立权重和日志项；正权重但缺少辅助输出时明确报错。
- `utils.py`、`train.py`：配置、开关一致性校验、实验命名和检查点元数据。
- `checkpoint_utils.py`：旧检查点默认关闭新头，校验新旧模型开关一致。
- `tests/test_vegetation_supervision.py`：标签映射、忽略区域、无植被裁剪块、损失加权、梯度接线、推理跳过、旧权重兼容和新检查点往返测试。

测试命令：`python -m unittest discover -s tests -v`。模型集成测试使用缩小的真实 SAM/LoRA 编码器和现有解码器，不需要下载预训练权重。功能验证不能证明真实数据集的性能提升；实验应同时记录五类 mIoU、两类植被 IoU、双向混淆、汽车和不透水面表现。

本次本地验证结果：PyTorch 1.13.0 CPU，37 项测试中 33 项通过、4 项 CUDA 测试因无 GPU 跳过。新辅助损失传回几何分支、空植被批次、推理不执行辅助头、保持已有参数初始化、旧模型严格加载及新检查点往返均通过。7 个相关 Python 文件的语法检查和 `git diff --check` 通过。当前环境无 CUDA，且未找到配置指定的 SAM 预训练权重，未运行完整真实数据集训练。

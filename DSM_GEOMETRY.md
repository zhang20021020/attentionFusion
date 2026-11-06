# 第二项创新的首版实现

本版完成三个步骤：五通道几何描述、轻量几何编码器、光学引导的双尺度残差注入。未加入植被二分类头或新的损失项。现有类别监督、置信度融合、PAN 和解码器继续工作。

默认新分支共有 77,140 个可训练参数，约 0.077M；这是参数统计，不代表实测推理开销。

默认新分支共有 77,140 个可训练参数，约 0.077M；这是参数统计，不代表实测推理开销。

## 当前配置与运行

在 utils.py 中统一配置：

```python
MODEL_OPTIONS = dict(
    use_category_supervision=True,
    use_confidence_fusion=True,
    use_category_pan=True,
    use_dsm_geometry=False,          # 旧 Sobel 输入替换关闭，主编码器仍输入 [D,D,D]
    use_n2_category_gate=False,      # 保持 R2 的设置
    use_dsm_geometry_branch=True,    # 新增的独立残差分支
    dsm_geometry_windows=(9, 25),    # 两个局部窗口，单位为像素
)
LOSS_OPTIONS = dict(category_weight=0.1, pan_weight=0.05,
                    decoder_weight=0.0, ignore_index=6)
```

保持 `MODE='Train'`，确认 `DATASET`、数据目录和 `weights/sam_vit_l_0b3195.pth` 后执行 `python train.py`。新分支默认已开启。训练流程仍从 SAM 预训练权重和新初始化的其余网络开始，不会因为填写了 CHECKPOINT_PATH 就自动从 R2 继续训练；该路径只在 Test 模式使用。

只把 `use_dsm_geometry_branch=False` 即可恢复 R2 的网络结构。旧的 `use_dsm_geometry` 与新的 `use_dsm_geometry_branch` 不能同时为 True，构造函数会直接报错，避免混入另一项输入改变。

旧权重兼容性需要模型构造函数默认新分支关闭；训练入口显式传入 utils.py 的 MODEL_OPTIONS，所以训练时以这里的 True 为准。其他脚本如果直接调用 UNetFormer()，应改为传入对应检查点的 model_options。

## 三步数据流

1. 对增强后的归一化 DSM D，计算 `Hk=D-AvgPool_k(D)` 和 `Sk=sqrt(max(AvgPool_k(D²)-AvgPool_k(D)²,0)+1e-6)`。拼接为 `[D,H9,H25,S9,S25]`，形状为 B×5×H×W。窗口边缘使用 replicate padding；统计计算保持 float32。平坦区域的 H 为 0，S 的数值下限为 0.001。
2. 几何编码器为 `Conv(5→32,s2) → DSConv(32→48,s2) → DSConv(48→64,s2) → DSConv(64→96,s2)`。采用 GroupNorm 和 GELU；输出 1/8 的 64 通道特征与 1/16 的 96 通道特征。
3. 光学特征与该尺度几何特征拼接，通过小卷积网络产生空间门控 M。利用 1×1 投影 P 将几何特征对齐到 256 通道，执行 `D_new = D_old + sigmoid(a) * M * P(G)`。a 初始为 -2；P 的权重零初始化，保证共享基线权重时初始前向等同 R2。第一次反向主要更新 P；P 完成第一次优化后，几何编码器、门控与残差强度开始获得梯度。这不是分支失效。

256×256 输入时：

| 特征 | 原尺寸 | 本版处理 |
|---|---|---|
| res1y | 256×64×64 | 不注入 |
| res2y | 256×32×32 | 注入 1/8 几何特征 |
| res3y | 256×16×16 | 注入 1/16 几何特征 |
| res4y | 256×8×8 | 不注入 |

注入发生在 `dsm_category_heads` 和 `SEFusion` 之前。增强后的 DSM 特征同时进入显式类别预测与主分割路径，现有 DSM 辅助 CE 和主任务均可训练新分支。分支训练和测试都运行，输入只需要光学图像与 DSM，不需要真值标签。

## 修改文件

- dsm_geometry.py：几何描述、编码器、光学引导残差模块。
- UNetFormer_MMSAM.py：新分支构造和双尺度注入。
- utils.py：开关和窗口，保留用户已有的旧几何输入关闭设置。
- train.py：实验命名和配置打印，调用检查点校验。
- checkpoint_utils.py：旧元数据缺少新开关时按 False 处理，检查新分支及窗口是否匹配。
- tests/test_dsm_geometry.py：数值、梯度、注入位置、模型兼容性和权重加载测试。

不需要修改 segmentation_loss.py；总损失仍为 `Lmain+0.1*(Lopt+LDSM)+0.05*LPAN`。

## 检查点与消融

新实验名会包含 `geo0_n20_gbr1_gw9x25_cw0.1_pw0.05_seed42`，其中 geo 是旧 Sobel 输入，gbr 是新增分支。窗口保存在 model_options 中，即使改成不同窗口也能在测试加载时检查。

测试新模型：`MODE='Test'`、CHECKPOINT_PATH 指向本次训练保存的权重，并保持对应的 MODEL_OPTIONS。不能打开新分支后直接把旧 R2 检查点当作已训练的新模型；严格加载会阻止这种情况。测试旧 R2 时把新分支关闭，其旧元数据会自动补上关闭标记。关闭分支时窗口不影响前向，因此加载校验不比较窗口。

第一轮只比较 R2 与 R2+新分支，固定其他设置。以后可以对比不含第一项的原模型+几何分支，或普通 DSM CNN 分支与五通道几何分支，分别判断独立作用与几何描述的作用。

Vaihingen 的窗口暂用 9/25；Potsdam 可把 17/45 作为近似物理范围的候选，需独立验证，不能假设最佳。这里的 D 是整幅 tile min-max 归一化值，H 是局部相对差，不是米制离地高度或 nDSM。当前数据加载会清理非有限值；本版没有新增 NoData 掩码或地面高程估计。若改用新的 DSM 数据源，基线也要使用同一数据源。

## 验证范围

运行 `python -m unittest discover -s tests -v`。测试使用真实 SAM/LoRA 实现的缩小编码器及现有解码器，另对新分支运行 256×256 输入的前后向；不依赖大体积预训练文件。CUDA 测试仅在 GPU 可用时运行。功能测试不能证明真实数据集上的 mIoU 提升，需要使用相同训练/测试协议验证，并同时检查植被双向混淆、汽车及建筑。

本次本地验证：PyTorch 1.13.0 CPU 环境，26 项测试中 23 项通过、3 项 CUDA 测试跳过。新分支的 256×256 前后向、零初始化等价、两步优化后的编码器/门控梯度、DSM 辅助监督接线、新旧检查点校验均通过。新增及修改的 Python 文件语法检查、git diff --check 通过。没有运行真实数据集训练；本地未找到配置指定的 SAM 预训练文件。

本次本地验证：PyTorch 1.13.0 CPU 环境，26 项测试中 23 项通过、3 项 CUDA 测试跳过。新分支的 256×256 前后向、零初始化等价、两步优化后的编码器/门控梯度、DSM 辅助监督接线、新旧检查点校验均通过。新增及修改的 Python 文件语法检查、git diff --check 通过。没有运行真实数据集训练；本地未找到配置指定的 SAM 预训练文件。

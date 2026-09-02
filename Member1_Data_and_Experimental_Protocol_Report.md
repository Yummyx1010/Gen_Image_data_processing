# Member 1: Data & Experimental Protocol Report

## 1. 数据集选择（Dataset Selection）

### 1.1 数据集选择原因

本项目最初计划使用 Tiny-GenImage 数据集进行 AI 生成图像检测实验。然而，在进一步分析后发现，原始数据集可能存在潜在的数据偏差，例如：

- AI-generated images 与 real images 之间可能存在图像格式差异；
- 不同生成器可能具有明显不同的原始分辨率；
- 模型可能学习 shortcut features，而不是 AI 生成图像本身的特征。

由于本研究关注的是 frequency-domain features 是否能够提升 cross-generator generalisation，因此需要尽量减少非目标因素影响。

因此，本项目最终选择使用 **Unbiased Tiny GenImage** 数据集。

---

## 2. Dataset Structure Inspection

Unbiased Tiny GenImage 按 generator/source 组织：

|类别|说明|
|-|-|
|ADM|AI-generated image|
|BigGAN|AI-generated image|
|glide|AI-generated image|
|Midjourney|AI-generated image|
|stable_diffusion_v_1_5|AI-generated image|
|VQDM|AI-generated image|
|wukong|AI-generated image|
|Nature|real image|

Dataset statistics:

|类别|图片数量|格式|
|-|-:|-|
|ADM|2500|JPG|
|BigGAN|2500|JPG|
|glide|2500|JPG|
|Midjourney|2500|JPG|
|stable_diffusion_v_1_5|2500|JPG|
|VQDM|2500|JPG|
|wukong|2500|JPG|
|Nature|5828|JPEG|

Total:

- Fake images: 17500
- Real images: 5828
- Total images: 23328

---

# 3. Dataset Bias Inspection

## 3.1 Image Format Analysis

通过 PIL 检查图片真实格式。

结果：

- AI-generated images: JPEG
- Real images: JPEG

因此不存在明显 JPEG/PNG format shortcut bias。

---

## 3.2 Resolution Analysis

原始图片分辨率存在 generator-dependent difference：

|Generator|主要分辨率|
|-|-|
|ADM|256×256|
|BigGAN|128×128|
|glide|256×256|
|VQDM|256×256|
|stable_diffusion_v_1_5|512×512|
|wukong|512×512|
|Midjourney|1024×1024|
|Nature|多种分辨率|

因此需要统一 preprocessing。

---

# 4. Shared Preprocessing Pipeline

为了保证 Baseline A（Spatial-only）和 Baseline B（Spatial + Frequency）输入一致，所有图片使用相同 preprocessing：

```
Original Image
        ↓
RGB Conversion
        ↓
Resize (shorter side = 256)
        ↓
Center Crop (224×224)
        ↓
Tensor Conversion
        ↓
ImageNet Normalization
        ↓
Model Input
```

最终输入：

```
[3,224,224]
```

测试 ADM、BigGAN、Nature 图片后，均成功转换为：

```
torch.Size([3,224,224])
```

---

# 5. Generator-disjoint Experimental Split

## 5.1 Split Strategy

采用 generator-disjoint split，确保测试阶段的 fake generator 不出现在训练阶段，以评估模型面对 unseen generators 的泛化能力。

## 5.2 Generator Allocation

### Training Generators

|Generator|Images|
|-|-:|
|ADM|2500|
|BigGAN|2500|
|stable_diffusion_v_1_5|2500|

Fake images: 7500

Real images: 3496

Total: 10996

---

### Validation Generator

|Generator|Images|
|-|-:|
|VQDM|2500|

Real images: 1166

Total: 3666

---

### Unseen Test Generators

|Generator|Images|
|-|-:|
|glide|2500|
|Midjourney|2500|
|wukong|2500|

Fake images: 7500

Real images: 1166

Total: 8666

---

# 6. CSV Dataset Generation

生成：

```
train.csv
validation.csv
test.csv
```

每条记录包含：

|字段|说明|
|-|-|
|image_path|图片相对路径|
|label|real=0, fake=1|
|generator|图片来源 generator|
|split|train/validation/test|

---

# 7. PyTorch Dataset Loader

Dataset loader 负责：

1. 从 CSV 读取图片路径；
2. 加载图片；
3. 应用统一 preprocessing；
4. 返回模型输入。

输出：

```
Image:
[3,224,224]

Label:
0/1

Generator:
source generator
```

测试结果：

```
Image shape:
torch.Size([4,3,224,224])

Labels:
tensor([1,1,0,0])

Generators:
['BigGAN',
 'stable_diffusion_v_1_5',
 'Nature',
 'Nature']
```

说明：

- 图片读取正常；
- label 映射正确；
- generator 信息保留；
- preprocessing 成功应用。

---

# 8. Member 1 Final Deliverables

```
Member1_Data/

├── inspect_dataset.py
├── check_image_properties.py
├── resolution_analysis.py
├── preprocessing.py
├── create_split_csv.py
├── dataset.py

├── dataset_statistics.csv
├── image_property_statistics.csv
├── generator_resolution_analysis.csv

├── train.csv
├── validation.csv
└── test.csv
```

---

# 9. Handover Notes

后续模型成员直接使用：

- train.csv
- validation.csv
- test.csv
- dataset.py

所有模型共享：

- 相同数据划分；
- 相同 preprocessing；
- 相同输入尺寸 [3,224,224]。

这样可以保证：

- Spatial baseline；
- Frequency model；
- Feature fusion model；

之间的实验公平性。

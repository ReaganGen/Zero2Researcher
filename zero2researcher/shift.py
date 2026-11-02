# This file is a modified version of ImageNet-C/create_c/make_cifar_c.py from
# https://github.com/hendrycks/robustness (commit 8190fe3), which is licensed under the
# Apache License, Version 2.0. The upstream file has no copyright line of its own (the
# repository's LICENSE keeps only the template line), so none is reproduced here.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE OF MODIFICATION (Apache-2.0 §4(b)): modified by the Zero2Researcher authors in
# 2026. The changes are listed in the module docstring below.
"""CIFAR-10-C 式的图像腐蚀（第 10 周讲义 S2"研究方法插曲：误差分析与数据错配"）。

Hendrycks & Dietterich (ICLR 2019, arXiv:1903.12261) 把常见的图像退化整理成 CIFAR-10-C：
15 种基准腐蚀 × 5 个严重度，另有论文附录 B 的 4 种验证用腐蚀（speckle noise、Gaussian blur、
spatter、saturate）。本模块按他们的代码实现其中 4 种（``gaussian_blur`` 属于验证用的那 4 种），
在内存里生成，用于讲义里的分布偏移扫描::

    from zero2researcher import shift
    x_c = shift.corrupt_cifar(cifar_test.data[:5000], "contrast", severity=2)

**来源与改动**（Apache-2.0 §4(b) 要求修改过的文件写明改动）：

- 改编自 hendrycks/robustness 仓库 ``ImageNet-C/create_c/make_cifar_c.py``（提交 8190fe3，
  Apache License 2.0，https://github.com/hendrycks/robustness ）的 ``gaussian_noise``、
  ``gaussian_blur``、``contrast``、``brightness`` 四个函数；**严重度常数逐字照抄**。
- 上游逐张处理 PIL 图像；这里改为对整批 uint8 数组 ``(N, H, W, 3)`` 一次处理（也接受单张 ``(H, W, 3)``）。
- 上游的高斯噪声用全局的 ``np.random``；这里改用带种子的 ``np.random.default_rng(seed)``，
  同一种子两次生成逐位相同。
- ``skimage.filters.gaussian(x, sigma=c, multichannel=True)`` 换成等价的
  ``scipy.ndimage.gaussian_filter``（空间两轴 sigma = c、通道轴与批次轴不模糊，``mode="nearest"``、
  ``truncate=4.0``，即 scikit-image 0.19 的默认值）。
- ``skimage.color.rgb2hsv / hsv2rgb`` 换成 ``matplotlib.colors.rgb_to_hsv / hsv_to_rgb``（同一套标准公式）。
- 上游 ``make_cifar_c.py`` 共 19 种腐蚀（15 种基准 + 4 种验证用），其余 15 种没有收录
  （其中不少依赖 OpenCV、Wand/ImageMagick 或 scikit-image）。
- 与上游相同：最后乘 255 后直接转成 uint8（截断，不四舍五入）。

**注意**：论文作者建议用他们发布的成品数据集（CIFAR-10-C，Zenodo 记录 2535967，CC BY 4.0）
做评测，因为在内存里生成的腐蚀不保证与它逐位相同。用本模块得到的准确率只在课程内部可比，
不能与论文里的 CIFAR-10-C 结果直接比较。

只依赖 numpy、scipy 与 matplotlib（scikit-learn 与 matplotlib 已经把它们装好）。
"""
import numpy as np

__all__ = ["SEVERITY", "CORRUPTIONS", "corrupt_cifar"]

# 严重度 1–5 对应的参数，逐字照抄上游 make_cifar_c.py
SEVERITY = {
    "gaussian_noise": [0.04, 0.06, .08, .09, .10],     # 噪声的标准差（像素在 0–1 尺度上）
    "gaussian_blur": [.4, .6, 0.7, .8, 1],             # 高斯核的 sigma（像素）
    "contrast": [.75, .5, .4, .3, 0.15],               # 向均值收缩的系数 c：越小对比度越低
    "brightness": [.05, .1, .15, .2, .3],              # HSV 空间里 V（亮度）的增量
}
CORRUPTIONS = tuple(SEVERITY)


def corrupt_cifar(images, name, severity, seed=0):
    """对一批 uint8 彩色图施加 CIFAR-10-C 定义的一种腐蚀。

    参数
    ----
    images : uint8 数组，形状 (N, H, W, 3)（例如 ``cifar_test.data``）或单张 (H, W, 3)。
    name : ``"gaussian_noise"``、``"gaussian_blur"``、``"contrast"``、``"brightness"`` 之一。
    severity : 严重度，1–5 的整数。
    seed : 随机种子，只影响 ``gaussian_noise``；其余三种是确定性的。

    返回与输入同形状的 uint8 新数组（不修改输入）。
    """
    images = np.asarray(images)
    if images.dtype != np.uint8:
        raise TypeError(f"images 应是 uint8 数组（0–255 的原始像素，例如 cifar_test.data），实际 dtype 为 {images.dtype}")
    if images.ndim not in (3, 4) or images.shape[-1] != 3:
        raise ValueError(f"images 的形状应是 (N, H, W, 3) 或 (H, W, 3)，实际 {images.shape}")
    if name not in SEVERITY:
        raise ValueError(f"不认识的腐蚀 {name!r}，可选：{', '.join(CORRUPTIONS)}")
    if isinstance(severity, bool) or int(severity) != severity or not 1 <= severity <= 5:
        raise ValueError(f"severity 应是 1–5 的整数，实际 {severity!r}")

    single = images.ndim == 3
    x = (images[None] if single else images).astype(np.float64) / 255.0
    c = SEVERITY[name][int(severity) - 1]

    if name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        out = np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1)
    elif name == "gaussian_blur":
        from scipy.ndimage import gaussian_filter
        # 批次轴与通道轴的 sigma 为 0（scipy 跳过这两轴），等价于逐张做 skimage 的 multichannel 模糊
        out = np.clip(gaussian_filter(x, sigma=(0, c, c, 0), mode="nearest", truncate=4.0), 0, 1)
    elif name == "contrast":
        means = x.mean(axis=(1, 2), keepdims=True)          # 每张图、每个通道的均值（上游 axis=(0, 1)）
        out = np.clip((x - means) * c + means, 0, 1)
    else:  # brightness
        from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
        hsv = rgb_to_hsv(x)
        hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
        out = np.clip(hsv_to_rgb(hsv), 0, 1)

    out = (out * 255).astype(np.uint8)                      # 与上游 np.uint8(...) 相同：截断
    return out[0] if single else out

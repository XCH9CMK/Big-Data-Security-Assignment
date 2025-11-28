# 大数据安全 - 语音降噪与增强实验

## 项目简介

本项目针对窃听场景中语音数据易受复杂噪声（如交通、人声、机械噪声等）干扰、信噪比低且可能伴随混响失真的特点，基于VCTK数据集，设计和实现了多种语音降噪与增强算法。

## 项目结构

```
Big-Data-Security-Assignment/
├── src/                          # 源代码目录
│   ├── data/                     # 数据准备模块
│   │   └── preparation.py       # 数据准备和预处理
│   ├── algorithms/               # 降噪算法模块
│   │   └── denoise.py           # 所有降噪算法实现
│   ├── evaluation/               # 评估模块
│   │   └── metrics.py           # 评估指标计算
│   ├── utils/                    # 工具模块
│   │   ├── config.py             # 配置文件
│   │   └── visualization.py     # 结果可视化
│   └── main.py                   # 主程序入口
├── run.py                        # 项目入口脚本（推荐使用）
├── requirements.txt              # 依赖包列表
├── README.md                     # 项目说明文档
├── data/                         # 数据目录
│   └── VCTK-Corpus/             # VCTK数据集（需下载）
│       ├── clean/               # 干净音频
│       ├── noisy/               # 含噪音频
│       └── txt/                 # 参考文本
└── output/                       # 输出目录
    ├── denoised/                # 降噪后音频
    ├── evaluation_*.csv          # 评估结果
    └── dl_model.pth             # 深度学习模型权重
```

## 功能特点

### 1. 数据准备模块 (`src/data/preparation.py`)
- 支持VCTK数据集的自动处理
- 生成多种类型的噪声：
  - 白噪声 (White Noise)
  - 粉红噪声 (Pink Noise)
  - 交通噪声 (Traffic Noise)
  - 机械噪声 (Mechanical Noise)
  - 人声噪声 (Babble Noise)
- 添加混响效果模拟真实环境
- 可自定义信噪比范围

### 2. 降噪算法模块 (`src/algorithms/denoise.py`)
实现了多种降噪算法：
- **谱减法 (Spectral Subtraction)**: 经典频域降噪方法
- **维纳滤波 (Wiener Filter)**: 基于统计的最优滤波
- **带通滤波 (Band-pass Filter)**: 保留语音频段
- **深度学习降噪 (Deep Learning)**: 基于U-Net架构的神经网络
- **混合降噪 (Hybrid)**: 结合多种方法的混合策略

### 3. 评估模块 (`src/evaluation/metrics.py`)
支持多种评估指标：
- **SNR (信噪比)**: 评估噪声抑制能力
- **MCD (梅尔倒谱失真)**: 评估音质保持能力
- **WER (错词率)**: 评估语音可懂度
- **STOI (短时客观可懂度)**: 评估语音清晰度（可选）
- **PESQ (感知语音质量)**: 评估主观感知质量（可选）

## 安装依赖

### 创建虚拟环境（推荐）

#### Windows
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1
```

#### Linux/Mac
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 安装依赖包

**方法1: 使用requirements.txt（推荐）**
```bash
pip install -r requirements.txt
```

**方法2: 分步安装**
```bash
# 核心库
pip install librosa soundfile numpy scipy pandas tqdm

# 深度学习
pip install torch torchaudio

# 评估指标
pip install pysptk fastdtw openai-whisper jiwer

# 可选库
pip install pesq pystoi
```


## 数据集准备

### VCTK数据集
1. 从以下链接下载VCTK数据集：
   https://datashare.ed.ac.uk/handle/10283/2791

2. 将数据集解压到项目目录：
   ```
   data/VCTK-Corpus/
   ├── clean/          # 放置 clean_testset_wav.zip 解压后的文件
   ├── noisy/          # 放置 noisy_testset_wav.zip 解压后的文件
   └── txt/            # 放置 testset_txt.zip 解压后的文件
   ```

**注意**: 如果没有VCTK数据集，程序会自动生成演示数据进行实验。

## 使用方法

### 环境准备

**重要**: 运行前请确保已激活虚拟环境或使用虚拟环境的Python解释器。

#### Windows (PowerShell)
```powershell
# 方法1: 激活虚拟环境后运行（推荐）
.\venv\Scripts\Activate.ps1
python run.py [参数]

# 方法2: 直接使用虚拟环境的Python
.\venv\Scripts\python.exe run.py [参数]
```

#### Windows (CMD)
```cmd
# 激活虚拟环境
venv\Scripts\activate.bat
python run.py [参数]
```

#### Linux/Mac
```bash
# 激活虚拟环境
source venv/bin/activate
python run.py [参数]
```

### 快速开始 - 运行完整实验

```bash
# 标准实验（包含深度学习训练）
python run.py --num_samples 50 --train_dl --max_eval 20

# 仅使用已训练模型
python run.py --num_samples 50 --max_eval 20
```

**Windows用户示例**:
```powershell
.\venv\Scripts\python.exe run.py --num_samples 50 --train_dl --max_eval 20
```

### 分步执行

#### 步骤1: 准备数据
```bash
python run.py --step data --num_samples 50
```

#### 步骤2: 训练深度学习模型（可选）
```bash
python run.py --step train --train_dl
```

#### 步骤3: 应用降噪算法
```bash
python run.py --step denoise
```

#### 步骤4: 评估效果
```bash
python run.py --step evaluate --max_eval 20
```

**注意**: 所有命令都需要在激活虚拟环境后执行，或使用虚拟环境的Python解释器。

### 命令行参数说明

```
--data_root       数据根目录 (默认: ./data)
--output_root     输出根目录 (默认: ./output)
--num_samples     生成的数据样本数量 (默认: 30)
--train_dl        是否训练深度学习模型
--max_eval        最大评估文件数 (默认: 20)
--step            执行的步骤: all, data, train, denoise, evaluate
```

## 实验结果

程序会生成以下输出：

1. **降噪音频文件**: `output/denoised/[algorithm_name]/`
2. **评估结果CSV**: `output/evaluation_[algorithm_name].csv`
3. **算法比较报告**: `output/algorithm_comparison.csv`

### 评估指标说明

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| SNR | 信噪比，衡量信号与噪声的功率比 | 越高越好 |
| SNR Improvement | SNR提升量 | 越高越好 |
| MCD | 梅尔倒谱失真，衡量音质变化 | 越低越好 |
| STOI | 短时客观可懂度 | 越高越好 (0-1) |
| WER | 错词率，衡量语音识别准确度 | 越低越好 |

## GPU加速

项目支持GPU加速训练和评估：

1. **安装CUDA版本的PyTorch**:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **自动检测**: 程序会自动检测并使用GPU（如果可用）

3. **验证GPU**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## 技术细节

### 噪声生成
- **白噪声**: 随机高斯噪声
- **粉红噪声**: 1/f噪声特性
- **交通噪声**: 低频周期性信号
- **机械噪声**: 脉冲型噪声
- **人声噪声**: 多频率混合模拟

### 降噪算法原理

1. **谱减法**: 在频域减去噪声功率谱
2. **维纳滤波**: 基于最小均方误差准则
3. **带通滤波**: 保留语音主要频率范围(80-8000Hz)
4. **深度学习**: U-Net架构预测增益掩码
5. **混合方法**: 级联多种算法

### 深度学习模型架构
- 编码器-解码器结构
- 输入: 噪声语音频谱
- 输出: 增益掩码 (0-1)
- 损失函数: MSE损失

## 注意事项

1. **计算资源**: 深度学习模型训练需要较多计算资源，建议使用GPU
2. **数据量**: 建议至少准备30个样本进行实验
3. **评估时间**: WER计算需要加载Whisper模型，首次运行较慢
4. **内存占用**: 处理大量音频文件时注意内存使用
5. **pysptk安装**: Windows用户需要安装Microsoft C++ Build Tools

## 常见问题

**Q: 没有VCTK数据集怎么办？**
A: 程序会自动生成合成音频数据用于演示实验。

**Q: 如何只测试某个算法？**
A: 使用 `--step denoise` 并在代码中指定算法名称，或修改main.py中的算法选择。

**Q: 如何调整算法参数？**
A: 在 `src/algorithms/denoise.py` 中修改各算法类的初始化参数。

**Q: 评估太慢怎么办？**
A: 使用 `--max_eval` 参数限制评估的文件数量，或使用更小的Whisper模型（已默认使用tiny模型）。

**Q: GPU不可用怎么办？**
A: 程序会自动回退到CPU模式，但速度会较慢。

**Q: ModuleNotFoundError: No module named 'librosa' 怎么办？**
A: 请确保已激活虚拟环境，或使用虚拟环境的Python解释器运行脚本。Windows用户可以使用 `.\venv\Scripts\python.exe run.py`。

## 扩展功能

可以进一步扩展的方向：
- 添加更多降噪算法（如RNNoise、CRN等）
- 实现实时降噪处理
- 优化深度学习模型架构
- 添加更多评估指标
- 支持更多音频格式
- Web界面展示结果

## 参考文献

1. Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction.
2. Wiener, N. (1949). Extrapolation, interpolation, and smoothing of stationary time series.
3. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

## 作者

大数据安全课程作业

## 许可证

MIT License


# 项目完整文件清单

## 核心模块

### 1. main.py
**主程序入口**
- 整合所有功能模块
- 提供命令行界面
- 支持分步或一键执行实验

### 2. data_preparation.py
**数据准备模块**
- 处理VCTK数据集
- 生成多种类型噪声（白噪声、粉红噪声、交通噪声、机械噪声、人声噪声）
- 添加混响效果
- 支持自定义SNR范围

### 3. denoise_algorithms.py
**降噪算法模块**
实现的算法：
- SpectralSubtraction（谱减法）
- WienerFilter（维纳滤波）
- BandPassFilter（带通滤波）
- DeepLearningDenoiser（深度学习降噪）
- HybridDenoiser（混合降噪）

### 4. evaluation.py
**评估模块**
支持的指标：
- SNR（信噪比）
- MCD（梅尔倒谱失真）
- WER（错词率）
- STOI（短时客观可懂度）
- PESQ（感知语音质量）

### 5. visualize_results.py
**结果可视化模块**
- 波形对比图
- 频谱图对比
- 算法性能比较图
- SNR分布图

### 6. compute.py
**原有评估函数**（已集成到evaluation.py）
- compute_MCD()
- compute_WER()

## 辅助文件

### 7. config.py
**配置文件**
- 数据集配置
- 噪声配置
- 算法参数配置
- 评估配置
- 输出配置

### 8. quick_test.py
**快速测试脚本**
- 验证系统功能
- 生成测试报告

### 9. requirements.txt
**Python依赖包列表**
所有必需和可选的第三方库

## 文档

### 10. README.md
**项目主文档**
- 项目简介
- 功能特点
- 安装说明
- 使用方法
- 项目结构

### 11. RUN_GUIDE.md
**快速运行指南**
- 详细的步骤说明
- 常见问题解决
- 预期输出说明

### 12. FILE_LIST.md
**本文件 - 完整文件清单**

## 批处理脚本（Windows）

### 13. run_experiment.bat
**一键运行实验**
- 检查环境
- 运行快速测试
- 执行完整实验

### 14. install_dependencies.bat
**安装依赖包**
- 自动安装所有必需的Python包

## 目录结构

```
Big-Data-Security-Assignment/
│
├── 核心模块
│   ├── main.py                      # 主程序
│   ├── data_preparation.py          # 数据准备
│   ├── denoise_algorithms.py        # 降噪算法
│   ├── evaluation.py                # 评估模块
│   ├── visualize_results.py         # 可视化
│   └── compute.py                   # 原有评估函数
│
├── 辅助文件
│   ├── config.py                    # 配置
│   ├── quick_test.py                # 快速测试
│   └── requirements.txt             # 依赖包
│
├── 文档
│   ├── README.md                    # 主文档
│   ├── RUN_GUIDE.md                 # 运行指南
│   └── FILE_LIST.md                 # 文件清单
│
├── 脚本
│   ├── run_experiment.bat           # 运行脚本
│   └── install_dependencies.bat     # 安装脚本
│
├── 数据目录（运行时生成）
│   └── data/
│       ├── VCTK-Corpus/            # VCTK数据集
│       ├── clean/                   # 干净音频
│       └── noisy/                   # 含噪音频
│
└── 输出目录（运行时生成）
    └── output/
        ├── denoised/                # 降噪结果
        │   ├── spectral_subtraction/
        │   ├── wiener_filter/
        │   ├── bandpass_filter/
        │   ├── hybrid/
        │   └── deep_learning/
        ├── plots/                   # 可视化图表
        ├── evaluation_*.csv         # 评估结果
        ├── algorithm_comparison.csv # 算法比较
        └── dl_model.pth            # 模型权重
```

## 使用流程

### 最简单的方式（Windows）
1. 双击 `install_dependencies.bat` 安装依赖
2. 双击 `run_experiment.bat` 运行实验

### 命令行方式
```bash
# 安装依赖
pip install -r requirements.txt

# 快速测试
python quick_test.py

# 运行完整实验
python main.py --num_samples 30 --max_eval 20

# 生成可视化报告
python visualize_results.py
```

## 输出文件说明

### 音频文件
- `data/clean/*.wav` - 干净音频
- `data/noisy/*.wav` - 含噪音频
- `output/denoised/*/*.wav` - 降噪后音频

### 评估结果
- `output/evaluation_*.csv` - 各算法详细评估结果
- `output/algorithm_comparison.csv` - 算法性能比较

### 可视化图表
- `output/plots/algorithm_comparison.png` - 算法比较图
- `output/plots/snr_distribution_*.png` - SNR分布图
- `output/plots/example_waveform.png` - 波形对比示例
- `output/plots/example_spectrogram.png` - 频谱图对比示例

### 模型文件
- `output/dl_model.pth` - 深度学习模型权重

## 代码统计

| 模块 | 行数（约） | 主要功能 |
|------|-----------|----------|
| main.py | 350 | 主程序控制 |
| data_preparation.py | 350 | 数据处理 |
| denoise_algorithms.py | 450 | 降噪算法 |
| evaluation.py | 300 | 评估指标 |
| visualize_results.py | 350 | 结果可视化 |
| 其他 | 200 | 配置和测试 |
| **总计** | **~2000** | 完整系统 |

## 技术栈

### 核心库
- librosa: 音频处理
- numpy, scipy: 数值计算
- torch: 深度学习
- soundfile: 音频I/O

### 评估库
- pysptk: 梅尔倒谱
- fastdtw: 动态时间规整
- whisper: 语音识别
- jiwer: 错词率计算

### 可视化
- matplotlib: 绘图
- pandas: 数据处理

## 扩展性

系统设计具有良好的扩展性：

1. **添加新算法**: 在`denoise_algorithms.py`中创建新类
2. **添加新指标**: 在`evaluation.py`中添加新方法
3. **添加新噪声**: 在`data_preparation.py`中扩展`generate_noise()`
4. **自定义可视化**: 在`visualize_results.py`中添加新图表

## 许可与引用

本项目为教育目的开发，使用的数据集和算法请遵循相应的许可协议。

VCTK数据集引用：
```
Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). 
CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92), [sound]. 
University of Edinburgh. The Centre for Speech Technology Research (CSTR). 
https://doi.org/10.7488/ds/2645.
```

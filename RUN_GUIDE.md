# 语音降噪实验 - 快速运行指南

## 前提条件

确保已安装Python 3.8或更高版本。

## 第一步：安装依赖

打开PowerShell，进入项目目录，运行：

```powershell
pip install -r requirements.txt
```

这将安装所有必需的Python库。

### 依赖安装说明

如果某些包安装失败，可以逐个安装：

```powershell
# 核心库
pip install numpy scipy pandas
pip install librosa soundfile tqdm

# 深度学习（如果有GPU，可以安装GPU版本）
pip install torch torchaudio

# 语音处理和评估
pip install pysptk fastdtw
pip install openai-whisper jiwer

# 可选库（用于更多评估指标）
pip install pesq pystoi
```

**注意事项：**
- `pysptk` 在Windows上可能需要编译器，如果安装失败，可以跳过（MCD评估将不可用）
- `pesq` 和 `pystoi` 是可选的，不影响核心功能

## 第二步：准备数据（可选）

### 选项A：使用VCTK数据集（推荐）

1. 下载VCTK数据集：https://datashare.ed.ac.uk/handle/10283/2791
2. 解压到项目目录：
   ```
   data/VCTK-Corpus/
   ```

### 选项B：使用生成的演示数据

如果不下载VCTK数据集，程序会自动生成合成音频数据用于实验。

## 第三步：运行快速测试

首次运行前，建议先进行快速测试以验证系统：

```powershell
python quick_test.py
```

测试将会：
- 生成测试音频
- 应用降噪算法
- 计算评估指标
- 生成测试报告

预期输出：
```
==================================================
快速测试 - 语音降噪系统
==================================================

[1/4] 测试数据准备模块...
   ✓ 生成干净音频: test_output\test_clean.wav
   ✓ 生成含噪音频 (white): test_output\test_noisy_white.wav
   ...

[2/4] 测试降噪算法...
   ...

==================================================
快速测试完成!
==================================================

系统运行正常，可以开始完整实验!
```

## 第四步：运行完整实验

### 方式1：一键运行完整实验（推荐新手）

```powershell
python main.py --num_samples 30 --max_eval 20
```

这将自动执行：
1. 准备数据（生成30个样本）
2. 应用所有降噪算法
3. 评估效果（评估20个文件）
4. 生成比较报告

### 方式2：分步执行（推荐进阶）

#### 步骤1：准备数据
```powershell
python main.py --step data --num_samples 50
```

#### 步骤2：应用降噪算法
```powershell
python main.py --step denoise
```

#### 步骤3：评估效果
```powershell
python main.py --step evaluate --max_eval 20
```

#### 步骤4（可选）：训练深度学习模型
```powershell
python main.py --step train --train_dl
```

## 查看结果

实验完成后，结果保存在 `output/` 目录：

```
output/
├── denoised/                      # 降噪后的音频文件
│   ├── spectral_subtraction/
│   ├── wiener_filter/
│   ├── bandpass_filter/
│   └── hybrid/
├── evaluation_*.csv               # 各算法的详细评估结果
└── algorithm_comparison.csv       # 算法性能比较
```

### 查看评估结果

1. **查看算法比较**（推荐）：
   ```powershell
   # 用Excel或文本编辑器打开
   notepad output\algorithm_comparison.csv
   ```

2. **查看详细评估**：
   ```powershell
   notepad output\evaluation_hybrid.csv
   ```

## 常用命令参数

```powershell
# 生成更多样本
python main.py --num_samples 100

# 只评估10个文件（加快速度）
python main.py --max_eval 10

# 训练深度学习模型
python main.py --train_dl

# 自定义输出路径
python main.py --output_root ./my_results

# 查看所有参数
python main.py --help
```

## 预期运行时间

基于标准配置（30样本，不训练深度学习）：

- **快速测试**: 约1-2分钟
- **数据准备**: 约2-5分钟
- **降噪处理**: 约3-5分钟
- **评估**: 约5-10分钟（首次运行需要下载Whisper模型）
- **总时间**: 约15-20分钟

**注意**: 
- 首次运行时Whisper会下载模型文件（约140MB）
- 如果训练深度学习模型，时间会显著增加（30分钟-2小时）

## 常见问题解决

### 问题1：导入错误
```
ImportError: No module named 'librosa'
```
**解决**: 重新安装依赖
```powershell
pip install librosa
```

### 问题2：CUDA错误（GPU相关）
```
CUDA out of memory
```
**解决**: PyTorch会自动使用CPU，不影响功能

### 问题3：pysptk安装失败
**解决**: MCD指标将不可用，但其他功能正常。可以继续实验。

### 问题4：运行太慢
**解决**: 减少样本数量
```powershell
python main.py --num_samples 10 --max_eval 5
```

### 问题5：没有VCTK数据集
**解决**: 无需担心，程序会自动生成演示数据

## 高级用法

### 只测试特定算法

修改 `main.py` 中的代码：

```python
# 在 step3_apply_denoising 中修改
experiment.step3_apply_denoising(algorithm_name='hybrid')  # 只用混合算法
```

### 调整算法参数

在 `denoise_algorithms.py` 中修改：

```python
# 例如调整谱减法参数
SpectralSubtraction(alpha=3.0, beta=0.02)
```

### 添加自定义噪声

在 `data_preparation.py` 的 `generate_noise()` 方法中添加新的噪声类型。

## 输出解读

### SNR (信噪比)
- 值越大越好
- 提升5dB以上表示明显改善

### MCD (梅尔倒谱失真)
- 值越小越好
- 小于5表示音质保持良好

### STOI (可懂度)
- 范围0-1，越接近1越好
- 大于0.8表示清晰可懂

## 下一步

1. 分析 `algorithm_comparison.csv` 中的结果
2. 听取 `output/denoised/` 中的音频文件
3. 根据需要调整算法参数
4. 尝试训练深度学习模型以获得更好效果

## 技术支持

如有问题，请检查：
1. Python版本是否正确
2. 所有依赖是否安装成功
3. 查看控制台的错误信息

祝实验顺利！

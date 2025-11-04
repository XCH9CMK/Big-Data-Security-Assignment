# 项目更新说明

## 更新日期
2025年11月4日

## 主要更新内容

### 1. 评估指标调整 ✅

**移除的指标：**
- ❌ SNR (信噪比)
- ❌ SNR Improvement (信噪比改善)
- ❌ STOI (短时客观可懂度) - 作为可选指标保留

**保留的核心指标：**
- ✅ MCD (梅尔倒谱失真) - 主要评估指标
- ✅ WER (错词率) - 主要评估指标

### 2. 修改的文件

#### compute.py
- ✅ 修改 `compute_WER()` 函数
  - 添加参数支持：`audio_file` 和 `reference_text`
  - 支持自动查找项目生成的音频文件
  - 移除硬编码的文件路径 `E:\dataset\result\test.wav`
  - 优先从以下目录查找：
    1. `./output/denoised/hybrid`
    2. `./output/denoised/wiener_filter`
    3. `./data/clean`
    4. `./data/noisy`

#### evaluation.py
- ✅ 移除 `compute_snr()` 函数
- ✅ 修改 `evaluate_denoising()` 函数
  - 移除SNR相关计算
  - 添加WER计算
  - 输出字段改为：`mcd_noisy`, `mcd_denoised`, `mcd_improvement`, `wer_noisy`, `wer_denoised`, `wer_improvement`
- ✅ 修改 `_print_statistics()` 函数
  - 只显示MCD和WER统计
- ✅ 修改 `evaluate_single_pair()` 函数
  - 只计算MCD和WER

#### main.py
- ✅ 修改 `_generate_comparison_report()` 函数
  - 输出字段改为：`Avg_MCD_Denoised`, `Avg_MCD_Improvement`, `Avg_WER_Denoised`, `Avg_WER_Improvement`
  - 移除SNR相关字段

#### visualize_results.py
- ✅ 修改 `plot_algorithm_comparison()` 函数
  - 只显示MCD和WER对比图（2个子图）
  - 移除SNR和STOI图表
- ✅ 重命名函数：`plot_snr_distribution()` → `plot_metrics_distribution()`
  - 显示MCD和WER的分布对比
  - 移除SNR分布图
- ✅ 修改 `generate_full_report()` 函数
  - 使用新的指标分布函数

#### quick_test.py
- ✅ 修改测试报告生成
  - 只显示MCD和WER结果
  - 移除SNR相关输出

### 3. 输出格式变化

#### CSV文件字段
**之前：**
```
file_index, clean_file, noisy_file, denoised_file,
snr_noisy_db, snr_denoised_db, snr_improvement_db,
mcd_noisy, mcd_denoised, stoi_noisy, stoi_denoised
```

**现在：**
```
file_index, clean_file, noisy_file, denoised_file,
mcd_noisy, mcd_denoised, mcd_improvement,
wer_noisy, wer_denoised, wer_improvement
```

#### 算法比较CSV字段
**之前：**
```
Algorithm, Avg_SNR_Improvement_dB, Final_SNR_dB, Avg_MCD, Avg_STOI
```

**现在：**
```
Algorithm, Avg_MCD_Denoised, Avg_MCD_Improvement, Avg_WER_Denoised, Avg_WER_Improvement
```

### 4. 可视化变化

#### 算法比较图
**之前：** 2x2布局，4个子图（SNR提升、最终SNR、MCD、STOI）
**现在：** 1x2布局，2个子图（MCD、WER）

#### 分布图
**之前：** SNR分布和SNR改善分布
**现在：** MCD分布和WER分布

### 5. 评估指标说明

#### MCD (梅尔倒谱失真)
- **含义**: 衡量降噪后音频与干净音频的音质差异
- **取值**: 数值越小越好
- **改善**: 正值表示改善（noisy_mcd - denoised_mcd）

#### WER (错词率)
- **含义**: 语音识别的错误率，反映语音可懂度
- **取值**: 0-1之间，越接近0越好
- **改善**: 正值表示改善（noisy_wer - denoised_wer）

### 6. 使用示例

#### 使用compute.py中的函数

```python
from compute import compute_MCD, compute_WER

# 计算MCD
mcd_value = compute_MCD("clean.wav", "denoised.wav")
print(f"MCD: {mcd_value:.2f}")

# 计算WER - 自动查找音频文件
wer_score = compute_WER()

# 计算WER - 指定音频文件
wer_score = compute_WER(
    audio_file="./output/denoised/hybrid/test.wav",
    reference_text="This is a test"
)
```

#### 使用evaluation.py中的类

```python
from evaluation import Evaluator

evaluator = Evaluator()

# 评估单个文件对
result = evaluator.evaluate_single_pair(
    clean_file="clean.wav",
    noisy_file="noisy.wav", 
    denoised_file="denoised.wav"
)

print(f"MCD改善: {result['mcd_improvement']:.2f}")
print(f"WER改善: {result['wer_improvement']:.4f}")
```

### 7. 向后兼容性

⚠️ **重要提示：**
- 本次更新移除了SNR相关的所有代码
- 如果之前的实验结果依赖SNR，需要重新运行实验
- CSV文件格式已变化，旧版本的结果文件无法直接使用

### 8. 测试建议

运行快速测试验证修改：

```bash
python quick_test.py
```

预期输出应包含：
- ✓ MCD评估结果
- ✓ WER评估结果（如果Whisper模型可用）
- ❌ 不应出现SNR相关输出

### 9. 完整实验流程

```bash
# 1. 准备数据
python main.py --step data --num_samples 30

# 2. 应用降噪
python main.py --step denoise

# 3. 评估（只计算MCD和WER）
python main.py --step evaluate --max_eval 20

# 4. 生成可视化（MCD和WER图表）
python visualize_results.py
```

### 10. 输出文件

运行实验后，您将得到：

```
output/
├── denoised/                          # 降噪音频
├── evaluation_*.csv                   # 评估结果（MCD + WER）
├── algorithm_comparison.csv           # 算法比较（MCD + WER）
└── plots/
    ├── algorithm_comparison.png       # MCD & WER对比图
    └── metrics_distribution_*.png     # MCD & WER分布图
```

### 11. 预期性能指标

典型的评估结果示例：

| 算法 | MCD (降噪后) | MCD改善 | WER (降噪后) | WER改善 |
|------|-------------|---------|-------------|---------|
| 谱减法 | 5.2 | 1.5 | 0.35 | 0.08 |
| 维纳滤波 | 4.8 | 1.9 | 0.32 | 0.11 |
| 混合方法 | 4.3 | 2.4 | 0.28 | 0.15 |

**解读：**
- MCD越低表示音质越好（越接近干净音频）
- WER越低表示可懂度越高（识别错误越少）
- 改善值为正表示降噪效果好

## 总结

本次更新专注于MCD和WER两个核心指标，使评估更加聚焦于：
1. **音质保持** (MCD) - 降噪是否损坏了语音的音质特征
2. **可懂度** (WER) - 降噪后的语音是否仍然清晰可懂

所有相关代码已更新完毕，系统可以正常运行。

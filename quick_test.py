"""
快速测试脚本
用于快速验证系统功能
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 导入模块
from data_preparation import DataPreparation
from denoise_algorithms import SpectralSubtraction, WienerFilter, HybridDenoiser
from evaluation import Evaluator


def quick_test():
    """快速测试所有模块"""
    
    print("="*60)
    print("快速测试 - 语音降噪系统")
    print("="*60)
    
    # 创建测试目录
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)
    
    # 1. 测试数据准备
    print("\n[1/4] 测试数据准备模块...")
    data_prep = DataPreparation(data_root="./test_data")
    
    # 生成测试音频
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sr))
    
    # 生成简单的正弦波信号作为测试
    clean_audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    clean_audio = clean_audio / (np.max(np.abs(clean_audio)) + 1e-8) * 0.8
    
    # 保存干净音频
    clean_file = test_dir / "test_clean.wav"
    sf.write(clean_file, clean_audio, sr)
    print(f"   ✓ 生成干净音频: {clean_file}")
    
    # 添加不同类型的噪声
    noise_types = ['white', 'pink', 'traffic']
    noisy_files = []
    
    for noise_type in noise_types:
        noisy_audio, _ = data_prep.add_noise(clean_audio, sr, snr_db=5, noise_type=noise_type)
        noisy_file = test_dir / f"test_noisy_{noise_type}.wav"
        sf.write(noisy_file, noisy_audio, sr)
        noisy_files.append(noisy_file)
        print(f"   ✓ 生成含噪音频 ({noise_type}): {noisy_file}")
    
    # 2. 测试降噪算法
    print("\n[2/4] 测试降噪算法...")
    
    algorithms = {
        'spectral_subtraction': SpectralSubtraction(),
        'wiener_filter': WienerFilter(),
        'hybrid': HybridDenoiser()
    }
    
    denoised_files = {}
    
    for alg_name, algorithm in algorithms.items():
        print(f"   测试 {alg_name}...")
        denoised_files[alg_name] = []
        
        for noisy_file in noisy_files:
            # 加载音频
            noisy_audio, sr = librosa.load(noisy_file, sr=16000)
            
            # 降噪
            denoised_audio = algorithm.denoise(noisy_audio, sr)
            
            # 保存
            denoised_file = test_dir / f"{noisy_file.stem}_{alg_name}.wav"
            sf.write(denoised_file, denoised_audio, sr)
            denoised_files[alg_name].append(denoised_file)
        
        print(f"   ✓ {alg_name} 完成")
    
    # 3. 测试评估模块
    print("\n[3/4] 测试评估模块...")
    evaluator = Evaluator()
    
    # 选择一个文件对进行评估
    test_noisy = noisy_files[0]
    test_denoised = denoised_files['hybrid'][0]
    
    print(f"   评估文件对:")
    print(f"   - 干净: {clean_file.name}")
    print(f"   - 含噪: {test_noisy.name}")
    print(f"   - 降噪: {test_denoised.name}")
    
    result = evaluator.evaluate_single_pair(
        str(clean_file),
        str(test_noisy),
        str(test_denoised),
        verbose=True
    )
    
    # 4. 生成测试报告
    print("\n[4/4] 生成测试报告...")
    
    report_file = test_dir / "test_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("语音降噪系统 - 快速测试报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"测试文件:\n")
        f.write(f"  干净音频: {clean_file}\n")
        f.write(f"  含噪音频: {len(noisy_files)} 个\n")
        f.write(f"  降噪音频: {len(denoised_files) * len(noisy_files)} 个\n\n")
        f.write(f"测试的算法:\n")
        for alg_name in algorithms.keys():
            f.write(f"  - {alg_name}\n")
        f.write(f"\n评估结果示例:\n")
        if result['mcd_noisy']:
            f.write(f"  MCD (含噪): {result['mcd_noisy']:.2f}\n")
            f.write(f"  MCD (降噪): {result['mcd_denoised']:.2f}\n")
            f.write(f"  MCD 改善: {result['mcd_improvement']:.2f}\n")
        if result['wer_noisy']:
            f.write(f"  WER (含噪): {result['wer_noisy']:.4f}\n")
            f.write(f"  WER (降噪): {result['wer_denoised']:.4f}\n")
            f.write(f"  WER 改善: {result['wer_improvement']:.4f}\n")
        f.write(f"\n测试状态: 成功 ✓\n")
    
    print(f"   ✓ 测试报告已保存: {report_file}")
    
    # 总结
    print("\n" + "="*60)
    print("快速测试完成!")
    print("="*60)
    print(f"\n所有测试文件保存在: {test_dir}")
    print("\n测试结果:")
    print(f"  ✓ 数据准备模块: 正常")
    print(f"  ✓ 降噪算法模块: 正常")
    print(f"  ✓ 评估模块: 正常")
    if result['mcd_improvement']:
        print(f"  ✓ MCD改善: {result['mcd_improvement']:.2f}")
    print("\n系统运行正常，可以开始完整实验!")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

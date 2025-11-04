"""
主程序 - 语音降噪与增强实验
整合数据准备、降噪算法和评估功能
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preparation import DataPreparation
from denoise_algorithms import (
    SpectralSubtraction, 
    WienerFilter, 
    BandPassFilter,
    DeepLearningDenoiser,
    HybridDenoiser
)
from evaluation import Evaluator


class VoiceEnhancementExperiment:
    """语音降噪与增强实验主类"""
    
    def __init__(self, data_root="./data", output_root="./output"):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.denoised_path = self.output_root / "denoised"
        self.denoised_path.mkdir(exist_ok=True)
        
        # 初始化模块
        self.data_prep = DataPreparation(data_root=str(self.data_root))
        self.evaluator = Evaluator()
        
        # 初始化降噪算法
        self.algorithms = {
            'spectral_subtraction': SpectralSubtraction(alpha=2.0, beta=0.01),
            'wiener_filter': WienerFilter(),
            'bandpass_filter': BandPassFilter(lowcut=80, highcut=8000),
            'hybrid': HybridDenoiser(),
        }
        
        # 深度学习模型路径
        self.dl_model_path = self.output_root / "dl_model.pth"
    
    def step1_prepare_data(self, num_samples=50, snr_range=(-5, 15)):
        """
        步骤1: 准备数据集
        """
        print("\n" + "="*60)
        print("步骤1: 准备数据集")
        print("="*60)
        
        self.data_prep.prepare_dataset(num_samples=num_samples, snr_range=snr_range)
    
    def step2_train_deep_model(self, epochs=20, batch_size=16):
        """
        步骤2: 训练深度学习降噪模型（可选）
        """
        print("\n" + "="*60)
        print("步骤2: 训练深度学习模型")
        print("="*60)
        
        # 获取训练数据
        clean_files = sorted(list(self.data_prep.clean_path.glob("*.wav")))
        noisy_files = sorted(list(self.data_prep.noisy_path.glob("*.wav")))
        
        if not clean_files or not noisy_files:
            print("未找到训练数据，跳过深度学习模型训练")
            return
        
        # 限制训练数据量以加快速度
        max_train_samples = min(len(clean_files), len(noisy_files), 100)
        clean_files = clean_files[:max_train_samples]
        noisy_files = noisy_files[:max_train_samples]
        
        print(f"使用 {len(clean_files)} 个样本训练模型...")
        
        # 创建并训练模型
        dl_denoiser = DeepLearningDenoiser()
        dl_denoiser.train(
            clean_files=clean_files,
            noisy_files=noisy_files,
            epochs=epochs,
            batch_size=batch_size,
            save_path=str(self.dl_model_path)
        )
        
        # 添加到算法字典
        self.algorithms['deep_learning'] = dl_denoiser
        print("深度学习模型训练完成!")
    
    def step3_apply_denoising(self, algorithm_name='all', max_files=None):
        """
        步骤3: 应用降噪算法
        
        Args:
            algorithm_name: 算法名称 ('all', 'spectral_subtraction', 'wiener_filter', 等)
            max_files: 最大处理文件数
        """
        print("\n" + "="*60)
        print(f"步骤3: 应用降噪算法 ({algorithm_name})")
        print("="*60)
        
        # 获取含噪音频文件
        noisy_files = sorted(list(self.data_prep.noisy_path.glob("*.wav")))
        
        if not noisy_files:
            print("未找到含噪音频文件!")
            return
        
        if max_files:
            noisy_files = noisy_files[:max_files]
        
        # 选择算法
        if algorithm_name == 'all':
            algorithms_to_use = self.algorithms
        elif algorithm_name in self.algorithms:
            algorithms_to_use = {algorithm_name: self.algorithms[algorithm_name]}
        else:
            print(f"未知算法: {algorithm_name}")
            print(f"可用算法: {list(self.algorithms.keys())}")
            return
        
        # 对每个算法应用降噪
        for alg_name, algorithm in algorithms_to_use.items():
            print(f"\n使用算法: {alg_name}")
            alg_output_path = self.denoised_path / alg_name
            alg_output_path.mkdir(exist_ok=True)
            
            for noisy_file in tqdm(noisy_files, desc=f"降噪 ({alg_name})"):
                try:
                    # 加载音频
                    noisy_audio, sr = librosa.load(noisy_file, sr=16000)
                    
                    # 应用降噪
                    denoised_audio = algorithm.denoise(noisy_audio, sr=sr)
                    
                    # 保存结果
                    output_file = alg_output_path / noisy_file.name
                    sf.write(output_file, denoised_audio, sr)
                
                except Exception as e:
                    print(f"\n处理文件 {noisy_file.name} 时出错: {e}")
                    continue
        
        print("\n降噪处理完成!")
    
    def step4_evaluate(self, algorithm_name='all', max_files=None):
        """
        步骤4: 评估降噪效果
        
        Args:
            algorithm_name: 算法名称
            max_files: 最大评估文件数
        """
        print("\n" + "="*60)
        print(f"步骤4: 评估降噪效果 ({algorithm_name})")
        print("="*60)
        
        # 获取文件列表
        clean_files = sorted(list(self.data_prep.clean_path.glob("*.wav")))
        noisy_files = sorted(list(self.data_prep.noisy_path.glob("*.wav")))
        
        if not clean_files or not noisy_files:
            print("未找到评估所需的音频文件!")
            return
        
        if max_files:
            clean_files = clean_files[:max_files]
            noisy_files = noisy_files[:max_files]
        
        # 选择算法
        if algorithm_name == 'all':
            algorithms_to_eval = list(self.algorithms.keys())
        else:
            algorithms_to_eval = [algorithm_name]
        
        # 评估每个算法
        all_results = {}
        
        for alg_name in algorithms_to_eval:
            alg_output_path = self.denoised_path / alg_name
            
            if not alg_output_path.exists():
                print(f"未找到算法 {alg_name} 的输出文件，跳过")
                continue
            
            print(f"\n评估算法: {alg_name}")
            print("-" * 60)
            
            # 构建文件对应关系
            eval_clean = []
            eval_noisy = []
            eval_denoised = []
            
            for noisy_file in noisy_files:
                denoised_file = alg_output_path / noisy_file.name
                
                if not denoised_file.exists():
                    continue
                
                # 找到对应的干净文件
                # 假设文件命名格式: noisy_XXXX_type_snrX.wav 对应 clean_XXXX.wav
                clean_idx = noisy_file.name.split('_')[1]
                clean_file = self.data_prep.clean_path / f"clean_{clean_idx}.wav"
                
                if clean_file.exists():
                    eval_clean.append(str(clean_file))
                    eval_noisy.append(str(noisy_file))
                    eval_denoised.append(str(denoised_file))
            
            if not eval_clean:
                print(f"未找到有效的文件对，跳过算法 {alg_name}")
                continue
            
            print(f"评估 {len(eval_clean)} 对音频文件...")
            
            # 评估
            results_csv = self.output_root / f"evaluation_{alg_name}.csv"
            results_df = self.evaluator.evaluate_denoising(
                clean_files=eval_clean,
                noisy_files=eval_noisy,
                denoised_files=eval_denoised,
                output_csv=str(results_csv)
            )
            
            all_results[alg_name] = results_df
        
        # 生成比较报告
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _generate_comparison_report(self, all_results):
        """生成算法比较报告"""
        if not all_results:
            return
        
        print("\n" + "="*60)
        print("算法性能比较")
        print("="*60)
        
        comparison = []
        
        for alg_name, results_df in all_results.items():
            metrics = {
                'Algorithm': alg_name,
            }
            
            if 'mcd_denoised' in results_df.columns and results_df['mcd_denoised'].notna().any():
                metrics['Avg_MCD_Denoised'] = results_df['mcd_denoised'].mean()
                if 'mcd_improvement' in results_df.columns:
                    metrics['Avg_MCD_Improvement'] = results_df['mcd_improvement'].mean()
            
            if 'wer_denoised' in results_df.columns and results_df['wer_denoised'].notna().any():
                metrics['Avg_WER_Denoised'] = results_df['wer_denoised'].mean()
                if 'wer_improvement' in results_df.columns:
                    metrics['Avg_WER_Improvement'] = results_df['wer_improvement'].mean()
            
            comparison.append(metrics)
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison)
        
        print("\n")
        print(comparison_df.to_string(index=False))
        
        # 保存比较结果
        comparison_csv = self.output_root / "algorithm_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\n比较结果已保存到: {comparison_csv}")
    
    def run_full_experiment(self, num_samples=50, train_dl=False, max_eval_files=20):
        """
        运行完整实验流程
        
        Args:
            num_samples: 数据样本数量
            train_dl: 是否训练深度学习模型
            max_eval_files: 最大评估文件数
        """
        print("\n" + "="*80)
        print(" " * 20 + "语音降噪与增强实验")
        print("="*80)
        
        # 步骤1: 准备数据
        self.step1_prepare_data(num_samples=num_samples)
        
        # 步骤2: 训练深度学习模型（可选）
        if train_dl:
            self.step2_train_deep_model(epochs=10, batch_size=8)
        
        # 步骤3: 应用降噪算法
        self.step3_apply_denoising(algorithm_name='all', max_files=max_eval_files)
        
        # 步骤4: 评估
        results = self.step4_evaluate(algorithm_name='all', max_files=max_eval_files)
        
        print("\n" + "="*80)
        print(" " * 25 + "实验完成!")
        print("="*80)
        print(f"\n所有结果已保存到: {self.output_root}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语音降噪与增强实验')
    parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
    parser.add_argument('--output_root', type=str, default='./output', help='输出根目录')
    parser.add_argument('--num_samples', type=int, default=30, help='数据样本数量')
    parser.add_argument('--train_dl', action='store_true', help='是否训练深度学习模型')
    parser.add_argument('--max_eval', type=int, default=20, help='最大评估文件数')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'data', 'train', 'denoise', 'evaluate'],
                       help='执行的步骤')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = VoiceEnhancementExperiment(
        data_root=args.data_root,
        output_root=args.output_root
    )
    
    # 执行实验
    if args.step == 'all':
        experiment.run_full_experiment(
            num_samples=args.num_samples,
            train_dl=args.train_dl,
            max_eval_files=args.max_eval
        )
    elif args.step == 'data':
        experiment.step1_prepare_data(num_samples=args.num_samples)
    elif args.step == 'train':
        experiment.step2_train_deep_model()
    elif args.step == 'denoise':
        experiment.step3_apply_denoising()
    elif args.step == 'evaluate':
        experiment.step4_evaluate()


if __name__ == "__main__":
    main()

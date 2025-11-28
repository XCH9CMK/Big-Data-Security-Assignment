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
from .data import DataPreparation
from .algorithms import (
    SpectralSubtraction, 
    WienerFilter, 
    BandPassFilter,
    DeepLearningDenoiser,
    HybridDenoiser
)
from .evaluation import Evaluator
from .utils.config import ALGORITHM_CONFIG, DATA_CONFIG
from .utils.file_utils import get_sample_rate, parse_noisy_filename


class VoiceEnhancementExperiment:
    """语音降噪与增强实验主类"""
    
    def __init__(self, data_root="./data", output_root="./output"):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.denoised_path = self.output_root / "denoised"
        self.denoised_path.mkdir(exist_ok=True)
        
        self.data_prep = DataPreparation(data_root=str(self.data_root))
        
        import torch
        import os
        use_gpu = torch.cuda.is_available()
        n_jobs = os.cpu_count() if use_gpu else -1
        self.evaluator = Evaluator(use_gpu=use_gpu, whisper_model_size="base", n_jobs=n_jobs)
        
        self.algorithms = {
            'spectral_subtraction': SpectralSubtraction(
                **ALGORITHM_CONFIG['spectral_subtraction']
            ),
            'wiener_filter': WienerFilter(),
            'bandpass_filter': BandPassFilter(
                **ALGORITHM_CONFIG['bandpass_filter']
            ),
            'hybrid': HybridDenoiser(),
        }
        
        self.dl_model_path = self.output_root / "dl_model.pth"
        
        if self.dl_model_path.exists():
            try:
                print(f"加载已存在的深度学习模型: {self.dl_model_path}")
                self.algorithms['deep_learning'] = DeepLearningDenoiser(
                    model_path=str(self.dl_model_path)
                )
            except Exception as e:
                print(f"加载深度学习模型失败: {e}")
    
    def step1_prepare_data(self, num_samples=50, snr_range=(-5, 15), force=False):
        """
        步骤1: 准备数据集（仅在数据不存在时生成）
        
        Args:
            num_samples: 数据样本数量
            snr_range: SNR范围
            force: 是否强制重新生成（默认False，如果数据已存在则跳过）
        """
        print("\n" + "="*60)
        print("步骤1: 准备数据集")
        print("="*60)
        
        # 检查数据是否已存在
        clean_files = list(self.data_prep.clean_path.glob("*.wav"))
        noisy_files = list(self.data_prep.noisy_path.glob("*.wav"))
        
        if not force and (clean_files or noisy_files):
            print(f"检测到已有数据文件:")
            print(f"  Clean文件: {len(clean_files)} 个")
            print(f"  Noisy文件: {len(noisy_files)} 个")
            print("跳过数据生成步骤（如需重新生成，请使用 --force_data 参数）")
            return
        
        # 如果数据不存在或强制生成，则生成数据
        self.data_prep.prepare_dataset(num_samples=num_samples, snr_range=snr_range)
    
    def step2_train_deep_model(self, epochs=None, batch_size=None):
        """
        步骤2: 训练深度学习降噪模型
        
        Args:
            epochs: 训练轮次（如果为None，使用配置文件中的值）
            batch_size: 批次大小（如果为None，使用配置文件中的值）
        """
        print("\n" + "="*60)
        print("步骤2: 训练深度学习模型")
        print("="*60)
        
        import random
        clean_files = list(self.data_prep.clean_path.glob("*.wav"))
        noisy_files = list(self.data_prep.noisy_path.glob("*.wav"))
        
        if not clean_files or not noisy_files:
            print("未找到训练数据，跳过深度学习模型训练")
            return
        
        # 使用更多训练数据以提高模型性能（随机选择）
        max_train_samples = min(len(clean_files), len(noisy_files), 200)
        clean_files = random.sample(clean_files, max_train_samples)
        noisy_files = random.sample(noisy_files, max_train_samples)
        
        # 使用配置文件中的参数，如果未指定则使用默认值
        if epochs is None:
            epochs = ALGORITHM_CONFIG['deep_learning']['epochs']
        if batch_size is None:
            batch_size = ALGORITHM_CONFIG['deep_learning']['batch_size']
        
        learning_rate = ALGORITHM_CONFIG['deep_learning']['learning_rate']
        
        print(f"训练参数:")
        print(f"  训练样本数: {len(clean_files)}")
        print(f"  训练轮次: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率: {learning_rate}")
        
        dl_denoiser = DeepLearningDenoiser()
        dl_denoiser.train(
            clean_files=clean_files,
            noisy_files=noisy_files,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_path=str(self.dl_model_path)
        )
        
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
        noisy_files = self._get_audio_files(self.data_prep.noisy_path, max_files)
        
        if not noisy_files:
            print("未找到含噪音频文件!")
            return
        
        # 选择算法
        algorithms_to_use = self._select_algorithms(algorithm_name)
        if not algorithms_to_use:
            return
        
        # 对每个算法应用降噪
        for alg_name, algorithm in algorithms_to_use.items():
            print(f"\n使用算法: {alg_name}")
            alg_output_path = self.denoised_path / alg_name
            alg_output_path.mkdir(exist_ok=True)
            
            for noisy_file in tqdm(noisy_files, desc=f"降噪 ({alg_name})"):
                try:
                    # 加载音频（使用配置的采样率）
                    sample_rate = DATA_CONFIG['sample_rate']
                    noisy_audio, sr = librosa.load(noisy_file, sr=sample_rate)
                    
                    # 应用降噪
                    denoised_audio = algorithm.denoise(noisy_audio, sr=sr)
                    
                    # 保存结果
                    output_file = alg_output_path / noisy_file.name
                    sf.write(output_file, denoised_audio, sr)
                
                except Exception as e:
                    print(f"\n处理文件 {noisy_file.name} 时出错: {e}")
                    continue
        
        print("\n降噪处理完成!")
    
    def step4_evaluate(self, algorithm_name='all', max_files=None, compute_snr=False):
        """
        步骤4: 评估降噪效果
        
        Args:
            algorithm_name: 算法名称
            max_files: 最大评估文件数
            compute_snr: 是否计算SNR（默认False，加快评估速度）
        """
        print("\n" + "="*60)
        print(f"步骤4: 评估降噪效果 ({algorithm_name})")
        print("="*60)
        
        # 获取文件列表
        clean_files = self._get_audio_files(self.data_prep.clean_path, max_files)
        noisy_files = self._get_audio_files(self.data_prep.noisy_path, max_files)
        
        if not clean_files or not noisy_files:
            print("未找到评估所需的音频文件!")
            return
        
        # 选择算法
        algorithms_to_eval = self._select_algorithms(algorithm_name, return_names=True)
        if not algorithms_to_eval:
            return
        
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
                
                # 找到对应的干净文件（使用工具函数）
                clean_file = self._match_clean_file(noisy_file)
                
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
                output_csv=str(results_csv),
                compute_snr=compute_snr
            )
            
            all_results[alg_name] = results_df
        
        # 生成比较报告
        if all_results:
            comparison_csv = self.output_root / "algorithm_comparison.csv"
            self.evaluator.generate_comparison_report(
                all_results=all_results,
                output_csv=str(comparison_csv)
            )
        
        return all_results
    
    def _get_audio_files(self, path, max_files=None, random_select=True):
        """
        获取音频文件列表（辅助方法）
        
        Args:
            path: 文件路径
            max_files: 最大文件数
            random_select: 是否随机选择（默认True）
        
        Returns:
            list: 文件列表
        """
        import random
        files = list(path.glob("*.wav"))
        if max_files and max_files < len(files):
            if random_select:
                files = random.sample(files, max_files)
            else:
                files = sorted(files)[:max_files]
        else:
            files = sorted(files)
        return files
    
    def _select_algorithms(self, algorithm_name, return_names=False):
        """
        选择要使用的算法（辅助方法）
        
        Args:
            algorithm_name: 算法名称或'all'
            return_names: 如果True，返回算法名称列表；否则返回算法字典
        
        Returns:
            dict或list: 算法字典或名称列表，如果无效则返回None或空列表
        """
        if algorithm_name == 'all':
            if return_names:
                return list(self.algorithms.keys())
            return self.algorithms
        elif algorithm_name in self.algorithms:
            if return_names:
                return [algorithm_name]
            return {algorithm_name: self.algorithms[algorithm_name]}
        else:
            print(f"未知算法: {algorithm_name}")
            print(f"可用算法: {list(self.algorithms.keys())}")
            return [] if return_names else None
    
    def _match_clean_file(self, noisy_file):
        """
        匹配对应的干净文件（辅助方法）
        优先使用VCTK同名文件进行匹配
        
        支持的匹配顺序：
        1. clean目录中的同名文件: clean/p232_001.wav (优先，用于VCTK数据集)
        2. 生成的clean文件: clean/clean_0000.wav (用于生成的noisy文件)
        
        Args:
            noisy_file: 含噪音频文件路径
        
        Returns:
            Path: 对应的干净文件路径
        """
        noisy_name = noisy_file.name
        
        # 优先级1: 尝试在clean目录中直接匹配同名文件（用于VCTK数据集）
        # VCTK数据集结构: VCTK-Corpus/clean/p232_001.wav 和 VCTK-Corpus/noisy/p232_001.wav
        clean_file = self.data_prep.clean_path / noisy_name
        if clean_file.exists():
            return clean_file
        
        # 优先级2: 尝试使用工具函数解析索引号（用于生成的noisy文件）
        # 格式: noisy_0000_babble_snr11.5.wav -> clean_0000.wav
        idx = parse_noisy_filename(noisy_name)
        if idx is not None:
            from .utils.file_utils import get_clean_filename
            clean_file = self.data_prep.clean_path / get_clean_filename(idx)
            if clean_file.exists():
                return clean_file
        
        # 如果都找不到，返回clean_path下的同名文件（即使不存在，用于错误提示）
        return self.data_prep.clean_path / noisy_name
    
    def run_full_experiment(self, num_samples=50, train_dl=True, max_eval_files=20, force_data=False):
        """
        运行完整实验流程
        
        Args:
            num_samples: 数据样本数量（仅在数据不存在时使用）
            train_dl: 是否训练深度学习模型（默认True）
            max_eval_files: 最大评估文件数
            force_data: 是否强制重新生成数据（默认False）
        """
        print("\n" + "="*80)
        print(" " * 20 + "语音降噪与增强实验")
        print("="*80)
        
        # 只在数据不存在或强制生成时才准备数据
        self.step1_prepare_data(num_samples=num_samples, force=force_data)
        
        if train_dl or 'deep_learning' not in self.algorithms:
            self.step2_train_deep_model()
        
        self.step3_apply_denoising(algorithm_name='all', max_files=max_eval_files)
        
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
    parser.add_argument('--train_dl', action='store_true', help='训练深度学习模型（默认启用，除非使用--no_train_dl）')
    parser.add_argument('--no_train_dl', dest='train_dl', action='store_false', help='禁用深度学习模型训练')
    parser.set_defaults(train_dl=True)
    parser.add_argument('--max_eval', type=int, default=20, help='最大评估文件数')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'data', 'train', 'denoise', 'evaluate'],
                       help='执行的步骤')
    parser.add_argument('--force_data', action='store_true', 
                       help='强制重新生成数据（即使数据已存在）')
    parser.add_argument('--algorithm', type=str, default='all',
                       help='指定要评估的算法名称（all, spectral_subtraction, wiener_filter, bandpass_filter, hybrid, deep_learning）')
    
    args = parser.parse_args()
    
    experiment = VoiceEnhancementExperiment(
        data_root=args.data_root,
        output_root=args.output_root
    )
    
    # 执行实验
    if args.step == 'all':
        experiment.run_full_experiment(
            num_samples=args.num_samples,
            train_dl=args.train_dl,
            max_eval_files=args.max_eval,
            force_data=args.force_data
        )
    elif args.step == 'data':
        experiment.step1_prepare_data(num_samples=args.num_samples, force=args.force_data)
    elif args.step == 'train':
        experiment.step2_train_deep_model()
    elif args.step == 'denoise':
        experiment.step3_apply_denoising(algorithm_name=args.algorithm, max_files=args.max_eval)
    elif args.step == 'evaluate':
        experiment.step4_evaluate(algorithm_name=args.algorithm, max_files=args.max_eval)


if __name__ == "__main__":
    main()


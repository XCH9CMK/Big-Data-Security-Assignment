"""
带暂停/恢复功能的语音降噪实验程序
支持每50个样本暂停一次，询问用户是否继续
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
import json
import time
from datetime import datetime
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


class ProgressManager:
    """实验进度管理器"""
    
    def __init__(self, output_root, batch_size=50):
        self.output_root = Path(output_root)
        self.progress_file = self.output_root / "experiment_progress.json"
        self.batch_size = batch_size
        self.progress_data = self._load_progress()
    
    def _load_progress(self):
        """加载实验进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "start_time": datetime.now().isoformat(),
                "completed_algorithms": [],
                "algorithm_progress": {},
                "total_samples": 0,
                "current_algorithm": None,
                "status": "not_started",
                "experiment_config": {}
            }
    
    def save_progress(self):
        """保存实验进度"""
        self.progress_data["last_update"] = datetime.now().isoformat()
        self.output_root.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress_data, f, indent=2, ensure_ascii=False)
    
    def update_algorithm_progress(self, algorithm, processed_count):
        """更新算法进度"""
        self.progress_data["algorithm_progress"][algorithm] = processed_count
        self.progress_data["current_algorithm"] = algorithm
        
    def mark_algorithm_complete(self, algorithm):
        """标记算法完成"""
        if algorithm not in self.progress_data["completed_algorithms"]:
            self.progress_data["completed_algorithms"].append(algorithm)
        
    def is_algorithm_complete(self, algorithm):
        """检查算法是否已完成"""
        return algorithm in self.progress_data["completed_algorithms"]
    
    def get_algorithm_progress(self, algorithm):
        """获取算法进度"""
        return self.progress_data["algorithm_progress"].get(algorithm, 0)
    
    def ask_continue(self):
        """询问用户是否继续"""
        total_processed = sum(self.progress_data["algorithm_progress"].values())
        
        print(f"\n{'='*70}")
        print(f"⏸️  暂停点: 已处理 {self.batch_size} 个文件")
        print(f"📊 当前进度:")
        print(f"   • 当前算法: {self.progress_data['current_algorithm']}")
        print(f"   • 已完成算法: {len(self.progress_data['completed_algorithms'])}")
        print(f"   • 总处理文件数: {total_processed}")
        
        for alg, count in self.progress_data["algorithm_progress"].items():
            status = "✅" if alg in self.progress_data["completed_algorithms"] else "🔄"
            print(f"   • {alg}: {count} 个文件 {status}")
        
        print(f"{'='*70}")
        
        while True:
            choice = input("是否继续实验？ (y=继续, n=暂停并保存): ").lower().strip()
            if choice in ['y', 'yes', '是', 'Y']:
                return True
            elif choice in ['n', 'no', '否', 'N']:
                print("\n🛑 实验已暂停，进度已保存。")
                print(f"📁 进度文件: {self.progress_file}")
                print(f"🔄 要恢复实验，请运行: python resume_experiment.py")
                self.save_progress()
                return False
            else:
                print("请输入 y (继续) 或 n (暂停)")


class PausableVoiceExperiment:
    """支持暂停/恢复的语音降噪实验"""
    
    def __init__(self, data_root="./data", output_root="./output", batch_size=50):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        
        # 创建输出目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.denoised_path = self.output_root / "denoised"
        self.denoised_path.mkdir(exist_ok=True)
        
        # 进度管理器
        self.progress_mgr = ProgressManager(output_root, batch_size)
        
        # 初始化模块
        self.data_prep = DataPreparation(data_root=str(self.data_root))
        self.evaluator = Evaluator()
        
        # 初始化降噪算法
        self.algorithms = {
            'spectral_subtraction': SpectralSubtraction(alpha=2.0, beta=0.01),
            'wiener_filter': WienerFilter(),
            'bandpass_filter': BandPassFilter(lowcut=80, highcut=8000),
            'deep_learning': DeepLearningDenoiser(),
            'hybrid': HybridDenoiser(),
        }
    
    def prepare_data(self, num_samples=50):
        """准备数据集"""
        print("\n" + "="*60)
        print("步骤1: 准备数据集")
        print("="*60)
        
        # 检查VCTK数据集
        vctk_path = self.data_prep.vctk_path
        if vctk_path.exists():
            clean_files = list((vctk_path / "clean").glob("*.wav"))
            noisy_files = list((vctk_path / "noisy").glob("*.wav"))
            
            if clean_files and noisy_files:
                print(f"✅ 发现VCTK数据集:")
                print(f"   干净音频: {len(clean_files)} 个文件")
                print(f"   含噪音频: {len(noisy_files)} 个文件")
                
                # 限制使用的样本数
                if num_samples < len(clean_files):
                    print(f"📊 将使用前 {num_samples} 个样本进行实验")
                
                return True
        
        print("❌ 未找到VCTK数据集，请检查数据放置")
        return False
    
    def apply_denoising_with_pause(self, algorithm_name='all', max_samples=None):
        """应用降噪算法（支持暂停）"""
        print("\n" + "="*60)
        print(f"步骤2: 应用降噪算法 ({algorithm_name})")
        print("="*60)
        
        # 获取含噪音频文件 - 修复路径问题
        vctk_noisy_path = self.data_prep.vctk_path / "noisy"
        noisy_files = list(vctk_noisy_path.glob("*.wav"))
        if not noisy_files:
            print("❌ 未找到含噪音频文件")
            print(f"   查找路径: {vctk_noisy_path}")
            return False
        
        # 限制样本数
        if max_samples and max_samples < len(noisy_files):
            noisy_files = noisy_files[:max_samples]
        
        # 更新总样本数
        self.progress_mgr.progress_data['total_samples'] = len(noisy_files)
        self.progress_mgr.save_progress()
        
        print(f"📊 总计处理 {len(noisy_files)} 个音频文件")
        
        # 选择要运行的算法
        if algorithm_name == 'all':
            target_algorithms = list(self.algorithms.keys())
        else:
            target_algorithms = [algorithm_name] if algorithm_name in self.algorithms else []
        
        print(f"🔧 将运行算法: {target_algorithms}")
        
        # 处理每个算法
        for alg_name in target_algorithms:
            # 检查算法是否已完成
            if self.progress_mgr.is_algorithm_complete(alg_name):
                print(f"\n✅ 算法 {alg_name} 已完成，跳过...")
                continue
            
            # 创建算法输出目录
            alg_output_path = self.denoised_path / alg_name
            alg_output_path.mkdir(exist_ok=True)
            
            print(f"\n🔄 开始处理算法: {alg_name}")
            algorithm = self.algorithms[alg_name]
            
            # 获取已处理的文件数
            start_idx = self.progress_mgr.get_algorithm_progress(alg_name)
            
            # 处理文件
            for i in tqdm(range(start_idx, len(noisy_files)), 
                         desc=f"降噪 ({alg_name})", 
                         initial=start_idx, 
                         total=len(noisy_files)):
                
                noisy_file = noisy_files[i]
                
                try:
                    # 读取含噪音频
                    noisy_audio, sr = librosa.load(noisy_file, sr=None)
                    
                    # 应用降噪
                    denoised_audio = algorithm.denoise(noisy_audio, sr)
                    
                    # 保存降噪后的音频
                    output_file = alg_output_path / noisy_file.name
                    sf.write(output_file, denoised_audio, sr)
                    
                    # 更新进度
                    self.progress_mgr.update_algorithm_progress(alg_name, i + 1)
                    
                    # 检查是否需要暂停
                    if (i + 1) % self.progress_mgr.batch_size == 0:
                        self.progress_mgr.save_progress()
                        if not self.progress_mgr.ask_continue():
                            return False  # 用户选择暂停
                    
                except Exception as e:
                    print(f"\n❌ 处理文件 {noisy_file.name} 时出错: {str(e)}")
                    continue
            
            # 标记算法完成
            self.progress_mgr.mark_algorithm_complete(alg_name)
            self.progress_mgr.save_progress()
            print(f"\n✅ {alg_name} 算法处理完成!")
        
        return True
    
    def evaluate_results(self, max_files=None):
        """评估降噪结果"""
        print("\n" + "="*60)
        print("步骤3: 评估降噪效果")
        print("="*60)
        
        # 获取干净和含噪音频文件 - 修复路径问题
        vctk_clean_path = self.data_prep.vctk_path / "clean"
        vctk_noisy_path = self.data_prep.vctk_path / "noisy"
        clean_files = sorted(list(vctk_clean_path.glob("*.wav")))
        noisy_files = sorted(list(vctk_noisy_path.glob("*.wav")))
        
        if not clean_files or not noisy_files:
            print("❌ 未找到评估所需的音频文件")
            return None
        
        # 限制评估文件数（None或非正数表示评估全部）
        if max_files is not None and max_files > 0:
            clean_files = clean_files[:max_files]
            noisy_files = noisy_files[:max_files]
        
        print(f"📊 将评估 {len(clean_files)} 对音频文件")
        
        all_results = {}
        
        # 评估每个已完成的算法
        for alg_name in self.progress_mgr.progress_data['completed_algorithms']:
            print(f"\n🔄 评估算法: {alg_name}")
            
            alg_output_path = self.denoised_path / alg_name
            if not alg_output_path.exists():
                print(f"⚠️ 未找到 {alg_name} 的输出文件")
                continue
            
            # 准备文件列表
            denoised_files = []
            valid_clean = []
            valid_noisy = []
            
            for clean_file, noisy_file in zip(clean_files, noisy_files):
                denoised_file = alg_output_path / noisy_file.name
                if denoised_file.exists():
                    denoised_files.append(str(denoised_file))
                    valid_clean.append(str(clean_file))
                    valid_noisy.append(str(noisy_file))
            
            if not denoised_files:
                print(f"⚠️ {alg_name} 没有可评估的文件")
                continue
            
            # 运行评估
            try:
                results_csv = self.output_root / f"evaluation_{alg_name}.csv"
                results_df = self.evaluator.evaluate_denoising(
                    clean_files=valid_clean,
                    noisy_files=valid_noisy,
                    denoised_files=denoised_files,
                    output_csv=str(results_csv)
                )
                
                if results_df is not None and not results_df.empty:
                    all_results[alg_name] = results_df
                    print(f"✅ {alg_name} 评估完成，结果保存到: {results_csv}")
                
            except Exception as e:
                print(f"❌ 评估 {alg_name} 时出错: {str(e)}")
                continue
        
        return all_results
    
    def generate_final_report(self, results):
        """生成最终报告"""
        if not results:
            print("\n❌ 没有结果可供生成报告")
            return
        
        print("\n" + "="*60)
        print("生成最终实验报告")
        print("="*60)
        
        # 算法对比数据
        comparison_data = []
        
        for alg_name, df in results.items():
            if df is not None and not df.empty:
                avg_mcd_improvement = df['mcd_improvement'].mean()
                avg_wer_improvement = df['wer_improvement'].mean()
                
                comparison_data.append({
                    'algorithm': alg_name,
                    'avg_mcd_improvement': avg_mcd_improvement,
                    'avg_wer_improvement': avg_wer_improvement,
                    'sample_count': len(df)
                })
        
        if comparison_data:
            import pandas as pd
            comparison_df = pd.DataFrame(comparison_data)
            comparison_csv = self.output_root / "algorithm_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            
            print("\n📊 算法性能对比:")
            print(comparison_df.to_string(index=False))
            print(f"\n💾 对比结果已保存到: {comparison_csv}")
    
    def run_experiment(self, num_samples=50, algorithm='all', max_eval=None, resume=False):
        """运行完整实验"""
        print("="*80)
        print("                    语音降噪与增强实验")
        if resume:
            print("                        (恢复模式)")
        print("="*80)
        
        # 更新实验配置
        self.progress_mgr.progress_data['experiment_config'] = {
            'num_samples': num_samples,
            'algorithm': algorithm,
            'max_eval': max_eval,
            'batch_size': self.progress_mgr.batch_size
        }
        self.progress_mgr.progress_data['status'] = 'running'
        
        if resume:
            print(f"\n🔄 恢复之前的实验...")
            print(f"📅 开始时间: {self.progress_mgr.progress_data.get('start_time', 'Unknown')}")
            print(f"✅ 已完成算法: {self.progress_mgr.progress_data['completed_algorithms']}")
            for alg, count in self.progress_mgr.progress_data['algorithm_progress'].items():
                print(f"   • {alg}: {count} 个文件")
        
        try:
            # 步骤1: 准备数据
            if not resume:
                if not self.prepare_data(num_samples):
                    return False
            
            # 步骤2: 应用降噪算法
            if not self.apply_denoising_with_pause(algorithm, num_samples):
                print("\n⏸️ 实验已暂停")
                return False
            
            # 步骤3: 评估结果
            results = self.evaluate_results(max_eval)
            
            # 步骤4: 生成报告
            self.generate_final_report(results)
            
            # 标记实验完成
            self.progress_mgr.progress_data['status'] = 'completed'
            self.progress_mgr.progress_data['end_time'] = datetime.now().isoformat()
            self.progress_mgr.save_progress()
            
            print(f"\n🎉 实验完成! 结果保存在: {self.output_root}")
            return True
            
        except KeyboardInterrupt:
            print("\n⏸️ 用户中断实验，进度已保存")
            self.progress_mgr.save_progress()
            return False
        except Exception as e:
            print(f"\n❌ 实验过程中出错: {str(e)}")
            self.progress_mgr.save_progress()
            return False


def main():
    parser = argparse.ArgumentParser(description='语音降噪与增强实验 (支持暂停/恢复)')
    parser.add_argument('--num_samples', type=int, default=50, help='处理的样本数量')
    parser.add_argument('--algorithm', type=str, default='all', 
                       choices=['all', 'spectral_subtraction', 'wiener_filter', 
                               'bandpass_filter', 'deep_learning', 'hybrid'],
                       help='要使用的降噪算法')
    parser.add_argument('--max_eval', type=int, default=None, help='最大评估文件数（不传或为0表示评估全部）')
    parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
    parser.add_argument('--output_root', type=str, default='./output', help='输出根目录')
    parser.add_argument('--batch_size', type=int, default=50, help='每批处理文件数（暂停间隔）')
    parser.add_argument('--resume', action='store_true', help='恢复之前的实验进度')
    
    args = parser.parse_args()
    
    # 创建实验实例
    experiment = PausableVoiceExperiment(
        data_root=args.data_root,
        output_root=args.output_root,
        batch_size=args.batch_size
    )
    
    # 运行实验
    success = experiment.run_experiment(
        num_samples=args.num_samples,
        algorithm=args.algorithm,
        max_eval=args.max_eval,
        resume=args.resume
    )
    
    if success:
        print("\n✅ 实验成功完成!")
    else:
        print("\n⏸️ 实验已暂停或出现错误")


if __name__ == "__main__":
    main()
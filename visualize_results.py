"""
结果可视化模块
用于可视化实验结果和生成图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
import librosa
import librosa.display


class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, output_dir="./output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_waveform_comparison(self, clean_file, noisy_file, denoised_file, save_name="waveform_comparison.png"):
        """
        绘制波形对比图
        
        Args:
            clean_file: 干净音频文件
            noisy_file: 含噪音频文件
            denoised_file: 降噪后音频文件
            save_name: 保存文件名
        """
        # 加载音频
        clean, sr = librosa.load(clean_file, sr=16000)
        noisy, _ = librosa.load(noisy_file, sr=16000)
        denoised, _ = librosa.load(denoised_file, sr=16000)
        
        # 确保长度一致
        min_len = min(len(clean), len(noisy), len(denoised))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        denoised = denoised[:min_len]
        
        # 创建时间轴
        time = np.arange(len(clean)) / sr
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        axes[0].plot(time, clean, color='blue', alpha=0.7)
        axes[0].set_title('干净音频', fontsize=12)
        axes[0].set_ylabel('幅度')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time, noisy, color='red', alpha=0.7)
        axes[1].set_title('含噪音频', fontsize=12)
        axes[1].set_ylabel('幅度')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time, denoised, color='green', alpha=0.7)
        axes[2].set_title('降噪后音频', fontsize=12)
        axes[2].set_xlabel('时间 (秒)')
        axes[2].set_ylabel('幅度')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"波形对比图已保存: {save_path}")
        return str(save_path)
    
    def plot_spectrogram_comparison(self, clean_file, noisy_file, denoised_file, save_name="spectrogram_comparison.png"):
        """
        绘制频谱图对比
        
        Args:
            clean_file: 干净音频文件
            noisy_file: 含噪音频文件
            denoised_file: 降噪后音频文件
            save_name: 保存文件名
        """
        # 加载音频
        clean, sr = librosa.load(clean_file, sr=16000)
        noisy, _ = librosa.load(noisy_file, sr=16000)
        denoised, _ = librosa.load(denoised_file, sr=16000)
        
        # 计算频谱图
        n_fft = 512
        hop_length = 128
        
        D_clean = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length)
        D_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
        D_denoised = librosa.stft(denoised, n_fft=n_fft, hop_length=hop_length)
        
        # 转换为dB
        S_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        S_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
        S_denoised = librosa.amplitude_to_db(np.abs(D_denoised), ref=np.max)
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        img1 = librosa.display.specshow(S_clean, sr=sr, hop_length=hop_length, 
                                        x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis')
        axes[0].set_title('干净音频频谱图', fontsize=12)
        axes[0].set_ylabel('频率 (Hz)')
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
        
        img2 = librosa.display.specshow(S_noisy, sr=sr, hop_length=hop_length,
                                        x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
        axes[1].set_title('含噪音频频谱图', fontsize=12)
        axes[1].set_ylabel('频率 (Hz)')
        fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
        
        img3 = librosa.display.specshow(S_denoised, sr=sr, hop_length=hop_length,
                                        x_axis='time', y_axis='hz', ax=axes[2], cmap='viridis')
        axes[2].set_title('降噪后音频频谱图', fontsize=12)
        axes[2].set_xlabel('时间 (秒)')
        axes[2].set_ylabel('频率 (Hz)')
        fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"频谱图对比已保存: {save_path}")
        return str(save_path)
    
    def plot_algorithm_comparison(self, comparison_csv, save_name="algorithm_comparison.png"):
        """
        绘制算法性能比较图
        
        Args:
            comparison_csv: 算法比较CSV文件路径
            save_name: 保存文件名
        """
        # 读取数据
        df = pd.read_csv(comparison_csv)
        
        # 设置算法名称（用于显示）
        algorithms = df['Algorithm'].tolist()
        
        # 创建子图 - 只显示MCD和WER
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('算法性能比较 (MCD & WER)', fontsize=16, fontweight='bold')
        
        # 1. MCD (降噪后)
        if 'Avg_MCD_Denoised' in df.columns:
            ax = axes[0]
            bars = ax.bar(algorithms, df['Avg_MCD_Denoised'], color='lightcoral', edgecolor='darkred')
            ax.set_title('平均MCD - 降噪后 (越低越好)', fontsize=12)
            ax.set_ylabel('MCD')
            ax.grid(True, axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        # 2. WER (降噪后)
        if 'Avg_WER_Denoised' in df.columns:
            ax = axes[1]
            bars = ax.bar(algorithms, df['Avg_WER_Denoised'], color='plum', edgecolor='purple')
            ax.set_title('平均WER - 降噪后 (越低越好)', fontsize=12)
            ax.set_ylabel('WER')
            ax.grid(True, axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"算法比较图已保存: {save_path}")
        return str(save_path)
    
    def plot_metrics_distribution(self, evaluation_csv, save_name="metrics_distribution.png"):
        """
        绘制MCD和WER分布图
        
        Args:
            evaluation_csv: 评估结果CSV文件
            save_name: 保存文件名
        """
        df = pd.read_csv(evaluation_csv)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MCD before/after
        ax = axes[0]
        if 'mcd_noisy' in df.columns and df['mcd_noisy'].notna().any():
            ax.hist(df['mcd_noisy'], bins=20, alpha=0.5, label='含噪音频', color='red', edgecolor='black')
            ax.hist(df['mcd_denoised'], bins=20, alpha=0.5, label='降噪后', color='green', edgecolor='black')
            ax.set_xlabel('MCD')
            ax.set_ylabel('频数')
            ax.set_title('MCD分布对比 (越低越好)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # WER before/after
        ax = axes[1]
        if 'wer_noisy' in df.columns and df['wer_noisy'].notna().any():
            ax.hist(df['wer_noisy'], bins=20, alpha=0.5, label='含噪音频', color='red', edgecolor='black')
            ax.hist(df['wer_denoised'], bins=20, alpha=0.5, label='降噪后', color='green', edgecolor='black')
            ax.set_xlabel('WER')
            ax.set_ylabel('频数')
            ax.set_title('WER分布对比 (越低越好)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"指标分布图已保存: {save_path}")
        return str(save_path)
    
    def generate_full_report(self, output_root="./output"):
        """
        生成完整的可视化报告
        
        Args:
            output_root: 输出根目录
        """
        output_root = Path(output_root)
        
        print("\n" + "="*60)
        print("生成可视化报告")
        print("="*60)
        
        generated_plots = []
        
        # 1. 算法比较图
        comparison_file = output_root / "algorithm_comparison.csv"
        if comparison_file.exists():
            print("\n生成算法比较图...")
            plot_path = self.plot_algorithm_comparison(str(comparison_file))
            generated_plots.append(plot_path)
        
        # 2. 为每个算法生成MCD/WER分布图
        eval_files = list(output_root.glob("evaluation_*.csv"))
        for eval_file in eval_files:
            alg_name = eval_file.stem.replace("evaluation_", "")
            print(f"\n生成 {alg_name} 的指标分布图...")
            plot_path = self.plot_metrics_distribution(
                str(eval_file), 
                save_name=f"metrics_distribution_{alg_name}.png"
            )
            generated_plots.append(plot_path)
        
        # 3. 生成示例音频的对比图
        clean_dir = Path("./data/clean")
        noisy_dir = Path("./data/noisy")
        denoised_dir = output_root / "denoised"
        
        if clean_dir.exists() and noisy_dir.exists():
            clean_files = sorted(list(clean_dir.glob("*.wav")))
            
            if clean_files:
                # 选择第一个文件作为示例
                clean_file = clean_files[0]
                clean_idx = clean_file.stem.split('_')[1]
                
                # 查找对应的含噪和降噪文件
                noisy_files = list(noisy_dir.glob(f"noisy_{clean_idx}_*.wav"))
                
                if noisy_files:
                    noisy_file = noisy_files[0]
                    
                    # 查找混合算法的降噪结果
                    hybrid_dir = denoised_dir / "hybrid"
                    if hybrid_dir.exists():
                        denoised_file = hybrid_dir / noisy_file.name
                        
                        if denoised_file.exists():
                            print(f"\n生成示例音频对比图...")
                            
                            # 波形图
                            plot_path = self.plot_waveform_comparison(
                                str(clean_file),
                                str(noisy_file),
                                str(denoised_file),
                                save_name="example_waveform.png"
                            )
                            generated_plots.append(plot_path)
                            
                            # 频谱图
                            plot_path = self.plot_spectrogram_comparison(
                                str(clean_file),
                                str(noisy_file),
                                str(denoised_file),
                                save_name="example_spectrogram.png"
                            )
                            generated_plots.append(plot_path)
        
        print("\n" + "="*60)
        print("可视化报告生成完成!")
        print("="*60)
        print(f"\n共生成 {len(generated_plots)} 个图表")
        print(f"保存位置: {self.output_dir}")
        
        for plot in generated_plots:
            print(f"  - {Path(plot).name}")
        
        return generated_plots


if __name__ == "__main__":
    # 生成可视化报告
    visualizer = ResultVisualizer()
    visualizer.generate_full_report()

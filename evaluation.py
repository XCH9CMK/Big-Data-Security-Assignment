"""
评估模块
使用MCD、WER指标评估降噪效果
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import whisper
import jiwer
from tqdm import tqdm


class Evaluator:
    def __init__(self):
        self.whisper_model = None
    
    def compute_mcd(self, file_original, file_reconstructed):
        """
        计算梅尔倒谱失真（MCD）
        
        Args:
            file_original: 原始音频文件路径
            file_reconstructed: 重构音频文件路径
        
        Returns:
            mcd_value: MCD值
        """
        def readmgc(filename):
            sr, x = wavfile.read(filename)
            if x.ndim == 2:
                x = x[:, 0]
            x = x.astype(np.float64)
            
            frame_length = 1024
            hop_length = 256
            
            # 窗口化
            frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
            frames *= pysptk.blackman(frame_length)
            
            # Mel倒谱系数阶数
            order = 25
            alpha = 0.41
            stage = 5
            gamma = -1.0 / stage
            
            mgc = pysptk.mgcep(frames, order, alpha, gamma)
            mgc = mgc.reshape(-1, order + 1)
            return mgc
        
        _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
        
        mgc1 = readmgc(file_original)
        mgc2 = readmgc(file_reconstructed)
        
        # 使用DTW对齐
        distance, path = fastdtw(mgc1, mgc2, dist=euclidean)
        
        pathx = list(map(lambda l: l[0], path))
        pathy = list(map(lambda l: l[1], path))
        x, y = mgc1[pathx], mgc2[pathy]
        
        frames = x.shape[0]
        z = x - y
        s = np.sqrt((z * z).sum(-1)).sum()
        
        mcd_value = _logdb_const * float(s) / float(frames)
        return mcd_value
    
    def compute_wer(self, audio_file, reference_text=None):
        """
        计算错词率（WER）
        
        Args:
            audio_file: 音频文件路径
            reference_text: 参考文本（如果为None，则使用默认文本）
        
        Returns:
            wer_score: WER值
            transcribed_text: 转录文本
        """
        # 加载Whisper模型
        if self.whisper_model is None:
            print("正在加载Whisper模型...")
            self.whisper_model = whisper.load_model("base")
        
        # 默认参考文本
        if reference_text is None:
            reference_text = "This is a test audio file for speech recognition evaluation"
        
        # 加载音频
        audio, _ = librosa.load(audio_file, sr=16000)
        
        # 转录
        result = self.whisper_model.transcribe(audio)
        transcribed_text = result["text"]
        
        # 计算WER
        wer_score = jiwer.wer(reference_text, transcribed_text)
        
        return wer_score, transcribed_text
    
    def compute_pesq(self, clean_file, degraded_file):
        """
        计算PESQ分数（感知语音质量评估）
        需要安装pesq库: pip install pesq
        
        Args:
            clean_file: 干净音频文件
            degraded_file: 降噪后音频文件
        
        Returns:
            pesq_score: PESQ分数
        """
        try:
            from pesq import pesq
            
            # 读取音频
            clean_audio, sr = librosa.load(clean_file, sr=16000)
            degraded_audio, _ = librosa.load(degraded_file, sr=16000)
            
            # 确保长度一致
            min_len = min(len(clean_audio), len(degraded_audio))
            clean_audio = clean_audio[:min_len]
            degraded_audio = degraded_audio[:min_len]
            
            # 计算PESQ
            pesq_score = pesq(sr, clean_audio, degraded_audio, 'wb')  # 'wb' for wideband
            return pesq_score
        
        except ImportError:
            print("警告: pesq库未安装，跳过PESQ计算")
            return None
        except Exception as e:
            print(f"计算PESQ时出错: {e}")
            return None
    
    def compute_stoi(self, clean_audio, degraded_audio, sr=16000):
        """
        计算STOI（短时客观可懂度）
        需要安装pystoi库: pip install pystoi
        
        Args:
            clean_audio: 干净音频信号
            degraded_audio: 降噪后音频信号
            sr: 采样率
        
        Returns:
            stoi_score: STOI分数
        """
        try:
            from pystoi import stoi
            
            # 确保长度一致
            min_len = min(len(clean_audio), len(degraded_audio))
            clean_audio = clean_audio[:min_len]
            degraded_audio = degraded_audio[:min_len]
            
            # 计算STOI
            stoi_score = stoi(clean_audio, degraded_audio, sr, extended=False)
            return stoi_score
        
        except ImportError:
            print("警告: pystoi库未安装，跳过STOI计算")
            return None
        except Exception as e:
            print(f"计算STOI时出错: {e}")
            return None
    
    def evaluate_denoising(self, clean_files, noisy_files, denoised_files, output_csv="evaluation_results.csv"):
        """
        全面评估降噪效果
        
        Args:
            clean_files: 干净音频文件列表
            noisy_files: 含噪音频文件列表
            denoised_files: 降噪后音频文件列表
            output_csv: 结果保存路径
        
        Returns:
            results_df: 评估结果DataFrame
        """
        results = []
        
        print("开始评估降噪效果...")
        for i, (clean_file, noisy_file, denoised_file) in enumerate(tqdm(
            zip(clean_files, noisy_files, denoised_files), 
            total=len(clean_files),
            desc="评估进度"
        )):
            try:
                # 计算MCD
                try:
                    mcd_noisy = self.compute_mcd(clean_file, noisy_file)
                    mcd_denoised = self.compute_mcd(clean_file, denoised_file)
                    mcd_improvement = mcd_noisy - mcd_denoised  # MCD越低越好，所以改善是正值
                except Exception as e:
                    print(f"计算MCD时出错 (文件 {i}): {e}")
                    mcd_noisy = None
                    mcd_denoised = None
                    mcd_improvement = None
                
                # 计算WER
                try:
                    wer_noisy, _ = self.compute_wer(noisy_file)
                    wer_denoised, _ = self.compute_wer(denoised_file)
                    wer_improvement = wer_noisy - wer_denoised  # WER越低越好，所以改善是正值
                except Exception as e:
                    print(f"计算WER时出错 (文件 {i}): {e}")
                    wer_noisy = None
                    wer_denoised = None
                    wer_improvement = None
                
                result = {
                    'file_index': i,
                    'clean_file': clean_file,
                    'noisy_file': noisy_file,
                    'denoised_file': denoised_file,
                    'mcd_noisy': mcd_noisy,
                    'mcd_denoised': mcd_denoised,
                    'mcd_improvement': mcd_improvement,
                    'wer_noisy': wer_noisy,
                    'wer_denoised': wer_denoised,
                    'wer_improvement': wer_improvement
                }
                
                results.append(result)
            
            except Exception as e:
                print(f"评估文件 {i} 时出错: {e}")
                continue
        
        # 创建DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        results_df.to_csv(output_csv, index=False)
        print(f"\n评估结果已保存到: {output_csv}")
        
        # 打印统计信息
        self._print_statistics(results_df)
        
        return results_df
    
    def _print_statistics(self, results_df):
        """打印评估统计信息"""
        print("\n" + "="*60)
        print("评估统计结果")
        print("="*60)
        
        if 'mcd_noisy' in results_df.columns and results_df['mcd_noisy'].notna().any():
            print(f"\nMCD (梅尔倒谱失真) - 越低越好:")
            print(f"  含噪音频平均MCD: {results_df['mcd_noisy'].mean():.2f}")
            print(f"  降噪后平均MCD: {results_df['mcd_denoised'].mean():.2f}")
            if 'mcd_improvement' in results_df.columns:
                print(f"  平均MCD改善: {results_df['mcd_improvement'].mean():.2f} (正值表示改善)")
        
        if 'wer_noisy' in results_df.columns and results_df['wer_noisy'].notna().any():
            print(f"\nWER (错词率) - 越低越好:")
            print(f"  含噪音频平均WER: {results_df['wer_noisy'].mean():.4f}")
            print(f"  降噪后平均WER: {results_df['wer_denoised'].mean():.4f}")
            if 'wer_improvement' in results_df.columns:
                print(f"  平均WER改善: {results_df['wer_improvement'].mean():.4f} (正值表示改善)")
        
        print("="*60)
    
    def evaluate_single_pair(self, clean_file, noisy_file, denoised_file, verbose=True):
        """
        评估单对音频文件
        
        Args:
            clean_file: 干净音频文件
            noisy_file: 含噪音频文件
            denoised_file: 降噪后音频文件
            verbose: 是否打印详细信息
        
        Returns:
            result: 评估结果字典
        """
        # 计算MCD
        try:
            mcd_noisy = self.compute_mcd(clean_file, noisy_file)
            mcd_denoised = self.compute_mcd(clean_file, denoised_file)
            mcd_improvement = mcd_noisy - mcd_denoised
        except Exception as e:
            if verbose:
                print(f"MCD计算失败: {e}")
            mcd_noisy = None
            mcd_denoised = None
            mcd_improvement = None
        
        # 计算WER
        try:
            wer_noisy, _ = self.compute_wer(noisy_file)
            wer_denoised, _ = self.compute_wer(denoised_file)
            wer_improvement = wer_noisy - wer_denoised
        except Exception as e:
            if verbose:
                print(f"WER计算失败: {e}")
            wer_noisy = None
            wer_denoised = None
            wer_improvement = None
        
        result = {
            'mcd_noisy': mcd_noisy,
            'mcd_denoised': mcd_denoised,
            'mcd_improvement': mcd_improvement,
            'wer_noisy': wer_noisy,
            'wer_denoised': wer_denoised,
            'wer_improvement': wer_improvement
        }
        
        if verbose:
            print("\n单文件评估结果:")
            if mcd_noisy is not None:
                print(f"  MCD (含噪): {mcd_noisy:.2f}")
                print(f"  MCD (降噪后): {mcd_denoised:.2f}")
                print(f"  MCD 改善: {mcd_improvement:.2f}")
            if wer_noisy is not None:
                print(f"  WER (含噪): {wer_noisy:.4f}")
                print(f"  WER (降噪后): {wer_denoised:.4f}")
                print(f"  WER 改善: {wer_improvement:.4f}")
        
        return result


if __name__ == "__main__":
    # 测试评估模块
    evaluator = Evaluator()
    print("评估模块已加载")

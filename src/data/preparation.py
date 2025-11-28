"""
数据准备模块
负责下载VCTK数据集、添加各种噪声（交通、人声、机械噪声等）
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import random

# 导入配置和工具函数
try:
    from ..utils.config import DATA_CONFIG, NOISE_CONFIG
    from ..utils.file_utils import get_sample_rate, get_clean_filename, get_noisy_filename
    DEFAULT_SAMPLE_RATE = get_sample_rate()
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000
    NOISE_CONFIG = {'types': ['white', 'pink', 'traffic', 'mechanical', 'babble']}
    def get_clean_filename(idx):
        return f"clean_{idx:04d}.wav"
    def get_noisy_filename(idx, noise_type, snr_db):
        return f"noisy_{idx:04d}_{noise_type}_snr{snr_db:.1f}.wav"


class DataPreparation:
    def __init__(self, data_root="./data"):
        self.data_root = Path(data_root)
        self.vctk_path = self.data_root / "VCTK-Corpus"
        
        # 检查VCTK数据集是否存在
        if (self.vctk_path / "clean").exists() and (self.vctk_path / "noisy").exists():
            # 使用VCTK数据集路径
            self.noisy_path = self.vctk_path / "noisy"
            self.clean_path = self.vctk_path / "clean"
            self.txt_path = self.vctk_path / "txt"
        else:
            # 使用生成数据路径
            self.noisy_path = self.data_root / "noisy"
            self.clean_path = self.data_root / "clean"
            self.txt_path = None
        
        # 创建必要的目录（仅在使用生成数据时）
        if not (self.vctk_path / "clean").exists():
            self.noisy_path.mkdir(parents=True, exist_ok=True)
            self.clean_path.mkdir(parents=True, exist_ok=True)
    
    def generate_noise(self, duration, sr, noise_type='white'):
        """
        生成不同类型的噪声
        
        Args:
            duration: 噪声持续时间（秒）
            sr: 采样率
            noise_type: 噪声类型 ('white', 'pink', 'traffic', 'mechanical', 'babble')
        
        Returns:
            noise: 生成的噪声信号
        """
        samples = int(duration * sr)
        
        if noise_type == 'white':
            # 白噪声
            noise = np.random.randn(samples)
        
        elif noise_type == 'pink':
            # 粉红噪声（1/f噪声）
            white = np.random.randn(samples)
            noise = np.cumsum(white)
            noise = noise - np.mean(noise)
            noise = noise / np.std(noise)
        
        elif noise_type == 'traffic':
            # 交通噪声模拟（低频为主）
            freqs = [50, 100, 150, 200, 300]  # 交通噪声主要频率
            noise = np.zeros(samples)
            for freq in freqs:
                t = np.linspace(0, duration, samples)
                amplitude = np.random.uniform(0.3, 1.0)
                noise += amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
            # 添加随机噪声
            noise += 0.5 * np.random.randn(samples)
        
        elif noise_type == 'mechanical':
            # 机械噪声（周期性脉冲）
            noise = np.random.randn(samples) * 0.3
            pulse_freq = 5  # 5Hz脉冲
            pulse_interval = sr // pulse_freq
            for i in range(0, samples, pulse_interval):
                pulse_length = min(sr // 20, samples - i)
                noise[i:i+pulse_length] += 2 * np.exp(-np.linspace(0, 5, pulse_length))
        
        elif noise_type == 'babble':
            # 人声噪声（多频率混合）
            noise = np.zeros(samples)
            num_voices = random.randint(3, 6)
            for _ in range(num_voices):
                freq = np.random.uniform(80, 300)  # 人声频率范围
                t = np.linspace(0, duration, samples)
                amplitude = np.random.uniform(0.2, 0.8)
                phase_noise = np.cumsum(np.random.randn(samples) * 0.01)
                noise += amplitude * np.sin(2 * np.pi * freq * t + phase_noise)
            # 添加共振峰效果
            noise += 0.3 * np.random.randn(samples)
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # 归一化
        noise = noise / (np.max(np.abs(noise)) + 1e-8)
        return noise
    
    def add_noise(self, clean_audio, sr, snr_db, noise_type='white'):
        """
        向干净音频添加噪声
        
        Args:
            clean_audio: 干净的音频信号
            sr: 采样率
            snr_db: 信噪比（dB）
            noise_type: 噪声类型
        
        Returns:
            noisy_audio: 含噪音频
            noise: 添加的噪声
        """
        duration = len(clean_audio) / sr
        noise = self.generate_noise(duration, sr, noise_type)
        
        # 确保噪声长度与音频相同
        if len(noise) > len(clean_audio):
            noise = noise[:len(clean_audio)]
        else:
            noise = np.pad(noise, (0, len(clean_audio) - len(noise)), 'wrap')
        
        # 计算音频功率
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # 根据SNR调整噪声强度
        snr_linear = 10 ** (snr_db / 10)
        noise_scaling = np.sqrt(signal_power / (noise_power * snr_linear))
        noise = noise * noise_scaling
        
        # 混合信号
        noisy_audio = clean_audio + noise
        
        # 防止削波
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.9
            noise = noise / max_val * 0.9
        
        return noisy_audio, noise
    
    def add_reverb(self, audio, sr, room_size='medium'):
        """
        添加混响效果
        
        Args:
            audio: 输入音频
            sr: 采样率
            room_size: 房间大小 ('small', 'medium', 'large')
        
        Returns:
            reverb_audio: 添加混响后的音频
        """
        # 简单的混响实现（使用延迟和衰减）
        if room_size == 'small':
            delays = [0.01, 0.02, 0.03]
            decays = [0.3, 0.2, 0.1]
        elif room_size == 'medium':
            delays = [0.02, 0.04, 0.06, 0.08]
            decays = [0.4, 0.3, 0.2, 0.1]
        else:  # large
            delays = [0.03, 0.06, 0.09, 0.12, 0.15]
            decays = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        reverb_audio = audio.copy()
        for delay, decay in zip(delays, decays):
            delay_samples = int(delay * sr)
            delayed = np.pad(audio, (delay_samples, 0), 'constant')[:len(audio)]
            reverb_audio += decay * delayed
        
        # 归一化
        reverb_audio = reverb_audio / (np.max(np.abs(reverb_audio)) + 1e-8) * 0.9
        return reverb_audio
    
    def prepare_dataset(self, num_samples=100, snr_range=(-5, 15)):
        """
        准备训练和测试数据集
        
        Args:
            num_samples: 要处理的音频样本数量
            snr_range: SNR范围 (min_snr, max_snr)
        """
        print("准备数据集...")
        
        # 查找VCTK数据集中的wav文件
        if not self.vctk_path.exists():
            print(f"VCTK数据集未找到，请将数据集放置在: {self.vctk_path}")
            print("数据集下载地址: https://datashare.ed.ac.uk/handle/10283/2791")
            print("\n将使用生成的示例数据进行演示...")
            self._create_demo_data(num_samples)
            return
        
        # 递归查找所有wav文件
        wav_files = list(self.vctk_path.rglob("*.wav"))
        
        if not wav_files:
            print("未找到wav文件，使用生成的示例数据...")
            self._create_demo_data(num_samples)
            return
        
        print(f"找到 {len(wav_files)} 个音频文件")
        
        # 随机选择文件
        selected_files = random.sample(wav_files, min(num_samples, len(wav_files)))
        
        noise_types = ['white', 'pink', 'traffic', 'mechanical', 'babble']
        
        for idx, wav_file in enumerate(tqdm(selected_files, desc="处理音频文件")):
            try:
                # 读取音频（使用配置的采样率）
                audio, sr = librosa.load(wav_file, sr=DEFAULT_SAMPLE_RATE)
                
                # 确保音频长度合理
                if len(audio) < sr * 0.5:  # 至少0.5秒
                    continue
                
                # 截取合适长度（2-5秒）
                max_duration = min(5.0, len(audio) / sr)
                if max_duration > 2.0:
                    duration = random.uniform(2.0, max_duration)
                    max_samples = int(duration * sr)
                    if len(audio) > max_samples:
                        start_idx = random.randint(0, len(audio) - max_samples)
                        audio = audio[start_idx:start_idx + max_samples]
                
                # 保存干净音频（使用工具函数生成文件名）
                clean_file = self.clean_path / get_clean_filename(idx)
                sf.write(clean_file, audio, sr)
                
                # 添加不同类型的噪声
                for noise_type in noise_types:
                    snr_db = random.uniform(snr_range[0], snr_range[1])
                    noisy_audio, _ = self.add_noise(audio, sr, snr_db, noise_type)
                    
                    # 随机添加混响
                    if random.random() > 0.5:
                        room_size = random.choice(['small', 'medium', 'large'])
                        noisy_audio = self.add_reverb(noisy_audio, sr, room_size)
                    
                    # 保存含噪音频（使用工具函数生成文件名）
                    noisy_file = self.noisy_path / get_noisy_filename(idx, noise_type, snr_db)
                    sf.write(noisy_file, noisy_audio, sr)
            
            except Exception as e:
                print(f"处理文件 {wav_file} 时出错: {e}")
                continue
        
        print(f"\n数据准备完成!")
        print(f"干净音频: {self.clean_path}")
        print(f"含噪音频: {self.noisy_path}")
    
    def _create_demo_data(self, num_samples=10):
        """
        创建演示数据（当VCTK数据集不可用时）
        """
        print("生成演示音频数据...")
        sr = DEFAULT_SAMPLE_RATE
        noise_types = NOISE_CONFIG.get('types', ['white', 'pink', 'traffic', 'mechanical', 'babble'])
        
        for idx in tqdm(range(num_samples), desc="生成演示数据"):
            # 生成合成语音信号（正弦波组合模拟）
            duration = random.uniform(2.0, 4.0)
            t = np.linspace(0, duration, int(duration * sr))
            
            # 基频和谐波（模拟人声）
            f0 = random.uniform(100, 250)  # 基频
            audio = np.zeros_like(t)
            for harmonic in range(1, 6):
                amplitude = 1.0 / harmonic
                audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
            
            # 添加调制
            modulation = 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
            audio = audio * modulation
            
            # 归一化
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
            
            # 保存干净音频（使用工具函数生成文件名）
            clean_file = self.clean_path / get_clean_filename(idx)
            sf.write(clean_file, audio, sr)
            
            # 为每种噪声类型创建样本
            for noise_type in noise_types:
                snr_db = random.uniform(-5, 15)
                noisy_audio, _ = self.add_noise(audio, sr, snr_db, noise_type)
                
                # 随机添加混响
                if random.random() > 0.5:
                    room_size = random.choice(['small', 'medium', 'large'])
                    noisy_audio = self.add_reverb(noisy_audio, sr, room_size)
                
                # 保存含噪音频（使用工具函数生成文件名）
                noisy_file = self.noisy_path / get_noisy_filename(idx, noise_type, snr_db)
                sf.write(noisy_file, noisy_audio, sr)
        
        print(f"演示数据生成完成! 共 {num_samples} 个样本")


if __name__ == "__main__":
    # 测试数据准备
    data_prep = DataPreparation()
    data_prep.prepare_dataset(num_samples=50)

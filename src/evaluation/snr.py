"""
SNR (Signal-to-Noise Ratio) 指标计算模块
"""

import numpy as np
import librosa

# 导入配置
try:
    from ..utils.config import DATA_CONFIG
    from ..utils.file_utils import get_sample_rate
    DEFAULT_SAMPLE_RATE = get_sample_rate()
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000


class SNRCalculator:
    """SNR计算器"""
    
    def _load_and_align_audio(self, file1, file2, sr=None):
        """
        加载两个音频文件并确保长度一致（辅助方法）
        
        Args:
            file1: 第一个音频文件路径
            file2: 第二个音频文件路径
            sr: 采样率（如果为None，使用配置的默认值）
        
        Returns:
            audio1, audio2, sr: 对齐后的音频和采样率
        """
        if sr is None:
            sr = DEFAULT_SAMPLE_RATE
        audio1, _ = librosa.load(file1, sr=sr)
        audio2, _ = librosa.load(file2, sr=sr)
        
        # 确保长度一致
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        return audio1, audio2, sr
    
    def compute_snr(self, clean_file, degraded_file):
        """
        计算信噪比（SNR）
        
        Args:
            clean_file: 干净音频文件路径
            degraded_file: 降噪后音频文件路径
        
        Returns:
            snr_db: SNR值（dB）
        """
        try:
            clean_audio, degraded_audio, _ = self._load_and_align_audio(clean_file, degraded_file)
            
            # 计算噪声（信号差异）
            noise = clean_audio - degraded_audio
            
            # 计算信号功率和噪声功率
            signal_power = np.mean(clean_audio ** 2)
            noise_power = np.mean(noise ** 2)
            
            # 避免除零
            if noise_power < 1e-10:
                return float('inf')
            
            # 计算SNR (dB)
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            return snr_db
        
        except Exception as e:
            print(f"计算SNR时出错: {e}")
            return None


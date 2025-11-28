"""
维纳滤波降噪算法
"""

import numpy as np
import librosa

# 尝试导入配置，如果失败则使用默认值
try:
    from ..utils.config import STFT_CONFIG, DATA_CONFIG
    DEFAULT_SAMPLE_RATE = DATA_CONFIG['sample_rate']
    DEFAULT_N_FFT = STFT_CONFIG['n_fft']
    DEFAULT_HOP_LENGTH = STFT_CONFIG['hop_length']
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_N_FFT = 512
    DEFAULT_HOP_LENGTH = 128


class WienerFilter:
    """维纳滤波降噪"""
    
    def __init__(self):
        pass
    
    def denoise(self, noisy_audio, sr=None, noise_profile_duration=0.5):
        """
        使用维纳滤波进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率（如果为None，使用配置的默认值）
            noise_profile_duration: 噪声特征估计时长（秒）
        
        Returns:
            denoised: 降噪后的音频
        """
        if sr is None:
            sr = DEFAULT_SAMPLE_RATE
        # STFT参数（使用配置）
        n_fft = DEFAULT_N_FFT
        hop_length = DEFAULT_HOP_LENGTH
        
        # 短时傅里叶变换
        stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power = magnitude ** 2
        
        # 估计噪声功率谱
        noise_frames = int(noise_profile_duration * sr / hop_length)
        noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
        
        # 维纳滤波
        # H = S / (S + N), 其中 S 是信号功率, N 是噪声功率
        signal_power = np.maximum(power - noise_power, 0)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
        
        # 应用滤波器
        magnitude_denoised = magnitude * wiener_gain
        
        # 重构信号
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        denoised = librosa.istft(stft_denoised, hop_length=hop_length)
        
        return denoised


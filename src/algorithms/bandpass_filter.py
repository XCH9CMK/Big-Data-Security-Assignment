"""
带通滤波器降噪算法
"""

from scipy import signal

# 尝试导入配置，如果失败则使用默认值
try:
    from ..utils.config import DATA_CONFIG
    DEFAULT_SAMPLE_RATE = DATA_CONFIG['sample_rate']
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000


class BandPassFilter:
    """带通滤波器（用于语音频段）"""
    
    def __init__(self, lowcut=80, highcut=8000):
        """
        Args:
            lowcut: 低截止频率（Hz）
            highcut: 高截止频率（Hz）
        """
        self.lowcut = lowcut
        self.highcut = highcut
    
    def denoise(self, noisy_audio, sr=None):
        """
        使用带通滤波进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率（如果为None，使用配置的默认值）
        
        Returns:
            denoised: 降噪后的音频
        """
        if sr is None:
            sr = DEFAULT_SAMPLE_RATE
        # 设计巴特沃斯带通滤波器
        nyquist = sr / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # 确保临界频率在有效范围内 (0, 1)
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(5, [low, high], btype='band')
        
        # 应用滤波器
        denoised = signal.filtfilt(b, a, noisy_audio)
        
        return denoised


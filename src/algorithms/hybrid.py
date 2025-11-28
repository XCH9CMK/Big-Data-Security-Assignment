"""
混合降噪算法
结合多种降噪方法
"""

# 尝试导入配置，如果失败则使用默认值
try:
    from ..utils.config import ALGORITHM_CONFIG
except ImportError:
    ALGORITHM_CONFIG = {}

from .spectral_subtraction import SpectralSubtraction
from .wiener_filter import WienerFilter
from .bandpass_filter import BandPassFilter

# 尝试导入配置，如果失败则使用默认值
try:
    from ..utils.config import DATA_CONFIG
    DEFAULT_SAMPLE_RATE = DATA_CONFIG['sample_rate']
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000


class HybridDenoiser:
    """混合降噪器（结合多种方法）"""
    
    def __init__(self):
        # 尝试从配置导入参数，如果失败则使用默认值
        try:
            self.spectral_sub = SpectralSubtraction(**ALGORITHM_CONFIG.get('spectral_subtraction', {}))
            self.bandpass = BandPassFilter(**ALGORITHM_CONFIG.get('bandpass_filter', {}))
        except (ImportError, KeyError):
            # 如果配置不可用，使用默认值
            self.spectral_sub = SpectralSubtraction(alpha=2.0, beta=0.01, adaptive=True)
            self.bandpass = BandPassFilter(lowcut=80, highcut=8000)
        self.wiener = WienerFilter()
    
    def denoise(self, noisy_audio, sr=None):
        """
        使用混合方法进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率（如果为None，使用配置的默认值）
        
        Returns:
            denoised: 降噪后的音频
        """
        if sr is None:
            sr = DEFAULT_SAMPLE_RATE
        # 步骤1: 带通滤波（去除明显的频率外噪声）
        audio = self.bandpass.denoise(noisy_audio, sr)
        
        # 步骤2: 谱减法
        audio = self.spectral_sub.denoise(audio, sr)
        
        # 步骤3: 维纳滤波（精细降噪）
        audio = self.wiener.denoise(audio, sr)
        
        return audio


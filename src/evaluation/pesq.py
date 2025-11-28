"""
PESQ (Perceptual Evaluation of Speech Quality) 指标计算模块
"""

import librosa

# 导入配置
try:
    from ..utils.config import DATA_CONFIG
    from ..utils.file_utils import get_sample_rate
    DEFAULT_SAMPLE_RATE = get_sample_rate()
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000


class PESQCalculator:
    """PESQ计算器"""
    
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
            
            clean_audio, degraded_audio, sr = self._load_and_align_audio(clean_file, degraded_file)
            
            # 计算PESQ
            pesq_score = pesq(sr, clean_audio, degraded_audio, 'wb')  # 'wb' for wideband
            return pesq_score
        
        except ImportError:
            print("警告: pesq库未安装，跳过PESQ计算")
            return None
        except Exception as e:
            print(f"计算PESQ时出错: {e}")
            return None


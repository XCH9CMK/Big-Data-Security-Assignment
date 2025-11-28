"""
WER (Word Error Rate) 指标计算模块
"""

import librosa
import whisper
import jiwer
import torch
import string

# 导入配置
try:
    from ..utils.config import DATA_CONFIG
    from ..utils.file_utils import get_sample_rate
    DEFAULT_SAMPLE_RATE = get_sample_rate()
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000


class WERCalculator:
    """WER计算器"""
    
    def __init__(self, use_gpu=True, whisper_model_size="base"):
        """
        Args:
            use_gpu: 是否使用GPU加速
            whisper_model_size: Whisper模型大小
        """
        self.whisper_model = None
        self.use_gpu = use_gpu
        self.whisper_model_size = whisper_model_size
        self.default_reference_text = "At least 12 persons saw the man with the revolver in the vicinity of the Tipit crime scene, at or immediately after the shooting. By the evening of November 22, five of them had identified Lee Harvey Oswald in police lineups as the man they saw. A sixth did so the next day. Three others subsequently identified Oswald from a photograph. Two witnesses testified that Oswald resembled the man they had seen. One witness felt he was too distant from the gunman to make a positive identification. A taxi driver, William Skoggins, was eating lunch in his cab, which was parked on Patten, facing the southeast corner of Tenth Street and Patten Avenue, a few feet to the north. A police car moving east on 10th at about 10 or 12 miles an hour passed in front of his cab. About 100 feet from the corner, the police car pulled up alongside a man on the sidewalk. This man dressed in a light-colored jacket approached the car."
    
    def _normalize_text_for_wer(self, text):
        """
        标准化文本用于WER计算
        转小写、移除标点符号、移除多余空格
        
        Args:
            text: 原始文本
        
        Returns:
            normalized_text: 标准化后的文本
        """
        # 转小写
        text = text.lower()
        
        # 移除标点符号
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        # 移除多余空格并strip
        text = ' '.join(text.split())
        
        return text
    
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
        # 加载Whisper模型（使用配置的模型大小）
        if self.whisper_model is None:
            print(f"正在加载Whisper模型 ({self.whisper_model_size})...")
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self.whisper_model = whisper.load_model(self.whisper_model_size, device=device)
            print(f"Whisper模型已加载到: {device}")
        
        # 默认参考文本（使用预加载的文本）
        if reference_text is None:
            reference_text = self.default_reference_text
        
        # 加载音频（使用配置的采样率）
        audio, _ = librosa.load(audio_file, sr=DEFAULT_SAMPLE_RATE)
        
        # 转录（使用更高精度参数）
        # beam_size: 使用beam search提高精度（默认5，显式指定确保一致性）
        # temperature: 使用0.0（贪婪解码）获得最准确的结果
        result = self.whisper_model.transcribe(
            audio,
            language="en",  # 指定语言可以提高准确性和速度
            fp16=self.use_gpu and torch.cuda.is_available(),  # GPU上使用fp16加速
            beam_size=5,  # beam search大小，提高精度（默认值，但显式指定）
            temperature=0.0,  # 贪婪解码，获得最准确的结果
            verbose=False  # 不打印详细信息
        )
        transcribed_text = result["text"].strip()
        
        # 标准化文本用于WER计算（转小写、去标点、去多余空格）
        ref_normalized = self._normalize_text_for_wer(reference_text)
        hyp_normalized = self._normalize_text_for_wer(transcribed_text)
        
        # 计算WER
        wer_score = jiwer.wer(ref_normalized, hyp_normalized)
        
        return wer_score, transcribed_text


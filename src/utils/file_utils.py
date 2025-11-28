"""
文件工具模块
提供文件命名、路径处理等工具函数
"""

from pathlib import Path
from .config import DATA_CONFIG, FILE_NAMING_CONFIG


def get_sample_rate():
    """获取配置的采样率"""
    return DATA_CONFIG['sample_rate']


def get_clean_filename(idx):
    """
    生成干净音频文件名
    
    Args:
        idx: 文件索引号
    
    Returns:
        str: 文件名
    """
    return FILE_NAMING_CONFIG['clean_format'].format(
        prefix=FILE_NAMING_CONFIG['clean_prefix'],
        idx=idx
    )


def get_noisy_filename(idx, noise_type, snr_db):
    """
    生成含噪音频文件名
    
    Args:
        idx: 文件索引号
        noise_type: 噪声类型
        snr_db: 信噪比（dB）
    
    Returns:
        str: 文件名
    """
    return FILE_NAMING_CONFIG['noisy_format'].format(
        prefix=FILE_NAMING_CONFIG['noisy_prefix'],
        idx=idx,
        noise_type=noise_type,
        snr_db=snr_db
    )


def parse_noisy_filename(filename):
    """
    解析含噪文件名，提取索引号
    
    Args:
        filename: 文件名（含或不含扩展名）
    
    Returns:
        int or None: 索引号，如果解析失败返回None
    """
    stem = Path(filename).stem
    if stem.startswith('noisy_'):
        try:
            parts = stem.split('_')
            if len(parts) >= 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
    return None


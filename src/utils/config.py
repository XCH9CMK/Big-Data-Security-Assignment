"""
配置文件
存储实验参数配置
"""

# 数据集配置
DATA_CONFIG = {
    'vctk_path': './data/VCTK-Corpus',
    'clean_path': './data/clean',
    'noisy_path': './data/noisy',
    'num_samples': 50,  # 生成的样本数量
    'snr_range': (-5, 15),  # 信噪比范围（dB）
    'sample_rate': 16000,  # 采样率
}

# 噪声类型配置
NOISE_CONFIG = {
    'types': ['white', 'pink', 'traffic', 'mechanical', 'babble'],
    'reverb_prob': 0.5,  # 添加混响的概率
    'room_sizes': ['small', 'medium', 'large'],
}

# 降噪算法配置
ALGORITHM_CONFIG = {
    'spectral_subtraction': {
        'alpha': 2.0,  # 过度减法因子（自适应时的初始值）
        'beta': 0.01,  # 谱底限（自适应时的初始值）
        'adaptive': True,  # 是否启用自适应参数调整
    },
    'wiener_filter': {
        'noise_profile_duration': 0.5,  # 噪声特征估计时长（秒）
    },
    'bandpass_filter': {
        'lowcut': 80,  # 低截止频率（Hz）
        'highcut': 8000,  # 高截止频率（Hz）
    },
    'deep_learning': {
        'n_fft': 512,
        'hop_length': 128,
        'epochs': 100,  # 增加训练轮次以获得更好的模型性能
        'batch_size': 32,
        'learning_rate': 0.001,
    }
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['snr', 'mcd', 'stoi'],  # 使用的评估指标
    'whisper_model': 'base',  # Whisper模型大小
    'max_eval_files': 20,  # 最大评估文件数
}

# 输出配置
OUTPUT_CONFIG = {
    'output_root': './output',
    'denoised_path': './output/denoised',
    'model_path': './output/dl_model.pth',
    'results_path': './output/results',
}

# STFT配置
STFT_CONFIG = {
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'window': 'hann',
}

# 训练配置
TRAINING_CONFIG = {
    'train_ratio': 0.8,  # 训练集比例
    'validation_ratio': 0.1,  # 验证集比例
    'test_ratio': 0.1,  # 测试集比例
    'random_seed': 42,
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'save_plots': True,
    'plot_format': 'png',
}

# 文件命名配置
FILE_NAMING_CONFIG = {
    'clean_prefix': 'clean',
    'noisy_prefix': 'noisy',
    'clean_format': '{prefix}_{idx:04d}.wav',
    'noisy_format': '{prefix}_{idx:04d}_{noise_type}_snr{snr_db:.1f}.wav',
}
"""
MCD (Mel-Cepstral Distortion) 指标计算模块
"""

import numpy as np
import librosa
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import threading


class MCDCalculator:
    """MCD计算器"""
    
    def __init__(self):
        self._audio_cache = {}
        self._mgc_cache = {}
        self._cache_lock = threading.Lock()
        # 预计算窗函数（避免重复计算）
        self._blackman_window = pysptk.blackman(1024)
    
    def _load_audio_for_mcd(self, filename):
        """
        加载音频用于MCD计算（带缓存，线程安全）
        
        Args:
            filename: 音频文件路径
        
        Returns:
            x: 音频数据（已缩放和添加dithering）
        """
        # 快速检查缓存（无锁读取）
        if filename in self._audio_cache:
            return self._audio_cache[filename]  # 直接返回（只读操作）
        
        # 需要加载时再获取锁
        with self._cache_lock:
            # 双重检查（避免重复加载）
            if filename in self._audio_cache:
                return self._audio_cache[filename]
            
            try:
                TARGET_SR = 16000
                y, _ = librosa.load(filename, sr=TARGET_SR)
                
                # 缩放到16位整数范围（-32768到32767）
                # 使用更稳定的缩放方式，避免溢出
                y_max = np.max(np.abs(y))
                if y_max > 1e-10:  # 避免除零
                    # 归一化到[-1, 1]然后缩放到16位范围（向量化操作）
                    y = (y / y_max) * 32767.0
                else:
                    # 如果信号太小，直接缩放
                    y = y * 32768.0
                
                # 添加小的dithering噪声以提高数值稳定性（使用固定种子确保可重复性）
                # 注意：dithering有助于避免量化误差，但应该很小
                # 使用文件名的hash作为种子，确保相同文件每次加载结果一致
                seed = hash(str(filename)) % (2**32)
                rng = np.random.RandomState(seed)
                y = y + rng.randn(len(y)) * 0.5  # 减小dithering幅度，提高稳定性
                
                # 缓存结果
                self._audio_cache[filename] = y
                
                return y
            except Exception as e:
                print(f"警告: 加载音频文件失败 {filename}: {e}")
                raise
    
    def compute_mcd(self, file_original, file_reconstructed):
        """
        计算梅尔倒谱失真（MCD）
        
        Args:
            file_original: 原始音频文件路径
            file_reconstructed: 重构音频文件路径
        
        Returns:
            mcd_value: MCD值
        """
        def get_mgc(x, cache_key=None):
            """计算梅尔倒谱系数（优化版）"""
            # 快速检查缓存（无锁读取）
            if cache_key and cache_key in self._mgc_cache:
                return self._mgc_cache[cache_key]  # 直接返回，不复制
            
            # 需要计算时再获取锁
            if cache_key:
                with self._cache_lock:
                    # 双重检查（避免重复计算）
                    if cache_key in self._mgc_cache:
                        return self._mgc_cache[cache_key]
            
            # 预处理音频
            if x.ndim == 2:
                x = x[:, 0]
            x = x.astype(np.float64, copy=False)  # 如果已经是float64，不复制
            
            # 确保音频长度足够
            frame_length = 1024
            hop_length = 256
            min_length = frame_length
            if len(x) < min_length:
                # 如果音频太短，进行零填充
                x = np.pad(x, (0, min_length - len(x)), mode='constant')
            
            # 帧提取（向量化操作）
            frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
            
            # 使用缓存的窗函数（避免重复计算）
            if frame_length == 1024:
                frames *= self._blackman_window  # 使用缓存的窗函数
            else:
                frames *= pysptk.blackman(frame_length)  # 动态计算（不常见）
            
            # MGC参数（标准设置）
            order = 25
            alpha = 0.41  # 全通常数
            stage = 5
            gamma = -1.0 / stage  # gamma参数
            
            # 计算MGC系数
            mgc = pysptk.mgcep(frames, order, alpha, gamma)
            mgc = mgc.reshape(-1, order + 1)
            
            # 确保MGC系数有效（检查NaN和Inf，仅在必要时处理）
            if not np.all(np.isfinite(mgc)):
                # 如果出现无效值，使用零填充
                mgc = np.nan_to_num(mgc, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 缓存结果
            if cache_key:
                with self._cache_lock:
                    if cache_key not in self._mgc_cache:
                        self._mgc_cache[cache_key] = mgc  # 直接存储，不复制
            
            return mgc
        
        _logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
        s = 0.0
        framesTot = 0
        
        y_clean = self._load_audio_for_mcd(file_original)
        y_enhanced = self._load_audio_for_mcd(file_reconstructed)
        
        mgc1 = get_mgc(y_clean, cache_key=file_original)
        mgc2 = get_mgc(y_enhanced, cache_key=file_reconstructed)
        
        # 使用DTW对齐两个MGC序列
        # 优化：使用适当的radius参数平衡速度和准确性
        # radius=1是默认值，对于大多数情况已经足够准确
        distance, path = fastdtw(mgc1, mgc2, dist=euclidean, radius=1)
        
        # 优化：使用numpy的fromiter或直接解包（最快的方法）
        # 如果path是列表，使用列表推导式转换为numpy数组
        if isinstance(path, list):
            # 使用zip解包然后转换为数组（比循环更快）
            pathx, pathy = zip(*path)
            pathx = np.array(pathx, dtype=np.int32)
            pathy = np.array(pathy, dtype=np.int32)
        else:
            # 如果已经是数组或其他类型，使用循环
            path_len = len(path)
            pathx = np.empty(path_len, dtype=np.int32)
            pathy = np.empty(path_len, dtype=np.int32)
            for i, (px, py) in enumerate(path):
                pathx[i] = px
                pathy[i] = py
        
        # 使用高级索引（比循环更快）
        x_aligned = mgc1[pathx]
        y_aligned = mgc2[pathy]
        
        frames = x_aligned.shape[0]
        framesTot += frames
        
        # 计算欧氏距离（完全向量化操作，优化内存访问）
        z = x_aligned - y_aligned
        # 使用更稳定的计算方式：先计算平方和，再开方
        squared_distances = np.sum(z * z, axis=1)
        # 避免数值不稳定：确保平方和不为负（向量化操作）
        squared_distances = np.maximum(squared_distances, 0.0)
        s += np.sum(np.sqrt(squared_distances))
        
        # 避免除零错误
        if framesTot == 0:
            return float('inf')
        
        MCD_value = _logdb_const * float(s) / float(framesTot)
        return MCD_value
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._audio_cache.clear()
            self._mgc_cache.clear()


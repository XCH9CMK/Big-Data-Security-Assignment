"""
语音降噪算法模块
实现多种降噪算法：谱减法、维纳滤波、深度学习方法等
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fftpack import fft, ifft
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


class SpectralSubtraction:
    """谱减法降噪"""
    
    def __init__(self, alpha=2.0, beta=0.01):
        """
        Args:
            alpha: 过度减法因子
            beta: 谱底限
        """
        self.alpha = alpha
        self.beta = beta
    
    def denoise(self, noisy_audio, sr=16000, noise_profile_duration=0.5):
        """
        使用谱减法进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率
            noise_profile_duration: 噪声特征估计时长（秒）
        
        Returns:
            denoised: 降噪后的音频
        """
        # STFT参数
        n_fft = 512
        hop_length = 128
        
        # 短时傅里叶变换
        stft = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 估计噪声频谱（使用前段作为噪声样本）
        noise_frames = int(noise_profile_duration * sr / hop_length)
        noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # 谱减法
        magnitude_denoised = magnitude - self.alpha * noise_profile
        
        # 应用谱底限
        magnitude_denoised = np.maximum(magnitude_denoised, self.beta * magnitude)
        
        # 重构信号
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        denoised = librosa.istft(stft_denoised, hop_length=hop_length)
        
        return denoised


class WienerFilter:
    """维纳滤波降噪"""
    
    def __init__(self):
        pass
    
    def denoise(self, noisy_audio, sr=16000, noise_profile_duration=0.5):
        """
        使用维纳滤波进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率
            noise_profile_duration: 噪声特征估计时长（秒）
        
        Returns:
            denoised: 降噪后的音频
        """
        # STFT参数
        n_fft = 512
        hop_length = 128
        
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
    
    def denoise(self, noisy_audio, sr=16000):
        """
        使用带通滤波进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率
        
        Returns:
            denoised: 降噪后的音频
        """
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


class SimpleDeepDenoiser(nn.Module):
    """简单的深度学习降噪模型（基于U-Net架构）"""
    
    def __init__(self, n_fft=512):
        super(SimpleDeepDenoiser, self).__init__()
        self.n_fft = n_fft
        freq_bins = n_fft // 2 + 1
        
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Linear(freq_bins, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(512, freq_bins),
            nn.Sigmoid()  # 输出增益掩码
        )
    
    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # 瓶颈
        b = self.bottleneck(e3)
        
        # 解码
        d3 = self.decoder3(b)
        d2 = self.decoder2(d3)
        mask = self.decoder1(d2)
        
        return mask


class DeepLearningDenoiser:
    """深度学习降噪器"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_fft = 512
        self.hop_length = 128
        self.model = SimpleDeepDenoiser(n_fft=self.n_fft).to(self.device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已从 {model_path} 加载")
    
    def train(self, clean_files, noisy_files, epochs=50, batch_size=32, save_path="model_weights.pth"):
        """
        训练降噪模型
        
        Args:
            clean_files: 干净音频文件列表
            noisy_files: 含噪音频文件列表
            epochs: 训练轮数
            batch_size: 批次大小
            save_path: 模型保存路径
        """
        print(f"开始训练模型 (设备: {self.device})...")
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 随机打乱文件顺序
            indices = np.random.permutation(len(clean_files))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_clean_specs = []
                batch_noisy_specs = []
                
                for idx in batch_indices:
                    try:
                        # 加载音频
                        clean_audio, _ = librosa.load(clean_files[idx], sr=16000)
                        noisy_audio, _ = librosa.load(noisy_files[idx], sr=16000)
                        
                        # 确保长度一致
                        min_len = min(len(clean_audio), len(noisy_audio))
                        clean_audio = clean_audio[:min_len]
                        noisy_audio = noisy_audio[:min_len]
                        
                        # STFT
                        clean_stft = librosa.stft(clean_audio, n_fft=self.n_fft, hop_length=self.hop_length)
                        noisy_stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
                        
                        clean_mag = np.abs(clean_stft).T
                        noisy_mag = np.abs(noisy_stft).T
                        
                        batch_clean_specs.append(clean_mag)
                        batch_noisy_specs.append(noisy_mag)
                    
                    except Exception as e:
                        continue
                
                if not batch_clean_specs:
                    continue
                
                # 将列表转换为张量（每个样本的帧数可能不同，需要填充或截断）
                max_frames = max([spec.shape[0] for spec in batch_noisy_specs])
                
                for j in range(len(batch_clean_specs)):
                    if batch_clean_specs[j].shape[0] < max_frames:
                        pad_width = ((0, max_frames - batch_clean_specs[j].shape[0]), (0, 0))
                        batch_clean_specs[j] = np.pad(batch_clean_specs[j], pad_width, mode='constant')
                        batch_noisy_specs[j] = np.pad(batch_noisy_specs[j], pad_width, mode='constant')
                    else:
                        batch_clean_specs[j] = batch_clean_specs[j][:max_frames]
                        batch_noisy_specs[j] = batch_noisy_specs[j][:max_frames]
                
                # 转换为张量
                clean_specs = torch.FloatTensor(np.array(batch_clean_specs)).to(self.device)
                noisy_specs = torch.FloatTensor(np.array(batch_noisy_specs)).to(self.device)
                
                # 计算理想掩码
                ideal_mask = clean_specs / (noisy_specs + 1e-8)
                ideal_mask = torch.clamp(ideal_mask, 0, 1)
                
                # 前向传播
                optimizer.zero_grad()
                
                # 对每一帧进行预测
                batch_loss = 0
                for frame_idx in range(max_frames):
                    predicted_mask = self.model(noisy_specs[:, frame_idx, :])
                    loss = criterion(predicted_mask, ideal_mask[:, frame_idx, :])
                    batch_loss += loss
                
                batch_loss = batch_loss / max_frames
                
                # 反向传播
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # 保存模型
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到 {save_path}")
    
    def denoise(self, noisy_audio, sr=16000):
        """
        使用深度学习模型进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率
        
        Returns:
            denoised: 降噪后的音频
        """
        self.model.eval()
        
        with torch.no_grad():
            # STFT
            stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 转置以便逐帧处理
            magnitude_frames = magnitude.T  # (time, freq)
            
            # 预测掩码
            masks = []
            for frame in magnitude_frames:
                frame_tensor = torch.FloatTensor(frame).unsqueeze(0).to(self.device)
                mask = self.model(frame_tensor)
                masks.append(mask.cpu().numpy())
            
            masks = np.array(masks).squeeze()  # (time, freq)
            masks = masks.T  # (freq, time)
            
            # 应用掩码
            magnitude_denoised = magnitude * masks
            
            # 重构信号
            stft_denoised = magnitude_denoised * np.exp(1j * phase)
            denoised = librosa.istft(stft_denoised, hop_length=self.hop_length)
        
        return denoised


class HybridDenoiser:
    """混合降噪器（结合多种方法）"""
    
    def __init__(self):
        self.spectral_sub = SpectralSubtraction(alpha=2.0, beta=0.01)
        self.wiener = WienerFilter()
        self.bandpass = BandPassFilter(lowcut=80, highcut=8000)
    
    def denoise(self, noisy_audio, sr=16000):
        """
        使用混合方法进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率
        
        Returns:
            denoised: 降噪后的音频
        """
        # 步骤1: 带通滤波（去除明显的频率外噪声）
        audio = self.bandpass.denoise(noisy_audio, sr)
        
        # 步骤2: 谱减法
        audio = self.spectral_sub.denoise(audio, sr)
        
        # 步骤3: 维纳滤波（精细降噪）
        audio = self.wiener.denoise(audio, sr)
        
        return audio


if __name__ == "__main__":
    # 测试降噪算法
    print("降噪算法模块已加载")

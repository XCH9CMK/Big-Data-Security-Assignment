"""
深度学习降噪算法
基于U-Net架构的谱掩码预测
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

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
        # 使用配置的STFT参数
        self.n_fft = DEFAULT_N_FFT
        self.hop_length = DEFAULT_HOP_LENGTH
        self.model = SimpleDeepDenoiser(n_fft=self.n_fft).to(self.device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已从 {model_path} 加载")
    
    def train(self, clean_files, noisy_files, epochs=50, batch_size=32, learning_rate=0.001, save_path="model_weights.pth"):
        """
        训练降噪模型
        
        Args:
            clean_files: 干净音频文件列表
            noisy_files: 含噪音频文件列表
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            save_path: 模型保存路径
        """
        print(f"开始训练模型 (设备: {self.device})...")
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # 添加学习率调度器，每20个epoch降低学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
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
                        # 加载音频（使用配置的采样率）
                        clean_audio, _ = librosa.load(clean_files[idx], sr=DEFAULT_SAMPLE_RATE)
                        noisy_audio, _ = librosa.load(noisy_files[idx], sr=DEFAULT_SAMPLE_RATE)
                        
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
            
            # 早停机制和最佳模型保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
            
            if patience_counter >= patience:
                print(f"早停触发：损失在 {patience} 个epoch内未改善")
                print(f"最佳损失: {best_loss:.6f}")
                break
        
        if patience_counter < patience:
            torch.save(self.model.state_dict(), save_path)
        
        print(f"模型已保存到 {save_path}")
    
    def denoise(self, noisy_audio, sr=None):
        """
        使用深度学习模型进行降噪
        
        Args:
            noisy_audio: 含噪音频
            sr: 采样率（如果为None，使用配置的默认值）
        
        Returns:
            denoised: 降噪后的音频
        """
        if sr is None:
            sr = DEFAULT_SAMPLE_RATE
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


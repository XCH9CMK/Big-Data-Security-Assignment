@echo off
chcp 65001 >nul
echo ====================================================
echo 安装实验所需依赖包
echo ====================================================
echo.

echo 正在安装核心依赖...
pip install numpy scipy pandas tqdm

echo.
echo 正在安装音频处理库...
pip install librosa soundfile

echo.
echo 正在安装深度学习框架...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo 正在安装语音评估库...
pip install pysptk fastdtw openai-whisper jiwer

echo.
echo 正在安装可选库...
pip install pesq pystoi
if errorlevel 1 (
    echo 注意: 可选库安装失败，但不影响主要功能
)

echo.
echo ====================================================
echo 依赖包安装完成!
echo ====================================================
echo.
pause

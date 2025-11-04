@echo off
chcp 65001 >nul
echo ====================================================
echo 语音降噪与增强实验 - 一键运行脚本
echo ====================================================
echo.

echo [1/4] 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)
echo ✓ Python环境正常
echo.

echo [2/4] 检查依赖包...
python -c "import librosa, soundfile, torch" 2>nul
if errorlevel 1 (
    echo 警告: 部分依赖包未安装
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 依赖包已安装
)
echo.

echo [3/4] 运行快速测试...
python quick_test.py
if errorlevel 1 (
    echo 警告: 快速测试失败，但将继续运行主实验
)
echo.

echo [4/4] 运行完整实验...
echo 这可能需要15-20分钟，请耐心等待...
echo.
python main.py --num_samples 30 --max_eval 20
if errorlevel 1 (
    echo 错误: 实验运行失败
    pause
    exit /b 1
)

echo.
echo ====================================================
echo 实验完成!
echo ====================================================
echo.
echo 结果保存在 output\ 目录
echo 主要文件:
echo   - output\algorithm_comparison.csv (算法比较)
echo   - output\denoised\ (降噪音频)
echo   - output\evaluation_*.csv (详细评估)
echo.
echo 按任意键退出...
pause >nul

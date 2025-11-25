"""
实验状态查看程序
查看当前实验进度，不进行任何操作
"""

import json
from pathlib import Path
from datetime import datetime
import argparse


def view_experiment_status(output_root="./output"):
    """查看实验状态"""
    progress_file = Path(output_root) / "experiment_progress.json"
    
    print("📊 语音降噪实验状态查看")
    print("="*60)
    
    if not progress_file.exists():
        print("❌ 没有找到实验进度文件")
        print(f"📁 查找位置: {progress_file}")
        print("\n💡 提示:")
        print("   • 如果还未开始实验，请运行: python pausable_experiment.py")
        print("   • 如果使用了自定义输出目录，请使用 --output_root 参数")
        return
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取进度文件失败: {str(e)}")
        return
    
    # 显示基本信息
    status = progress_data.get('status', 'unknown')
    status_icon = {
        'not_started': '⏸️',
        'running': '🔄',
        'completed': '✅',
        'unknown': '❓'
    }.get(status, '❓')
    
    print(f"📋 实验状态: {status_icon} {status.upper()}")
    print(f"🕐 开始时间: {progress_data.get('start_time', 'Unknown')}")
    
    if 'last_update' in progress_data:
        print(f"📅 最后更新: {progress_data['last_update']}")
    
    if 'end_time' in progress_data:
        print(f"🏁 结束时间: {progress_data['end_time']}")
    
    # 显示实验配置
    config = progress_data.get('experiment_config', {})
    if config:
        print(f"\n⚙️ 实验配置:")
        print(f"   • 总样本数: {config.get('num_samples', 'N/A')}")
        print(f"   • 目标算法: {config.get('algorithm', 'N/A')}")
        print(f"   • 评估文件数: {config.get('max_eval', 'N/A')}")
        print(f"   • 批处理大小: {config.get('batch_size', 'N/A')} (每批暂停一次)")
    
    # 显示算法进度
    print(f"\n🔧 算法处理进度:")
    completed_algs = progress_data.get('completed_algorithms', [])
    alg_progress = progress_data.get('algorithm_progress', {})
    current_alg = progress_data.get('current_algorithm')
    
    if not alg_progress:
        print("   • 暂无处理记录")
    else:
        total_processed = 0
        for alg, count in alg_progress.items():
            is_completed = alg in completed_algs
            is_current = alg == current_alg
            
            if is_completed:
                status_icon = "✅"
            elif is_current:
                status_icon = "🔄"
            else:
                status_icon = "⏸️"
            
            print(f"   • {alg}: {count} 个文件 {status_icon}")
            total_processed += count
        
        print(f"\n📈 统计信息:")
        print(f"   • 总处理文件: {total_processed}")
        print(f"   • 已完成算法: {len(completed_algs)}")
        
        if current_alg and current_alg not in completed_algs:
            print(f"   • 当前算法: {current_alg}")
    
    # 显示输出文件
    output_path = Path(output_root)
    if output_path.exists():
        print(f"\n📁 输出文件:")
        
        # 检查降噪文件
        denoised_path = output_path / "denoised"
        if denoised_path.exists():
            alg_dirs = list(denoised_path.iterdir())
            if alg_dirs:
                print(f"   • 降噪音频: {len(alg_dirs)} 个算法目录")
                for alg_dir in alg_dirs:
                    if alg_dir.is_dir():
                        file_count = len(list(alg_dir.glob("*.wav")))
                        print(f"     - {alg_dir.name}: {file_count} 个文件")
        
        # 检查评估文件
        eval_files = list(output_path.glob("evaluation_*.csv"))
        if eval_files:
            print(f"   • 评估报告: {len(eval_files)} 个CSV文件")
            for eval_file in eval_files:
                print(f"     - {eval_file.name}")
        
        # 检查对比文件
        comparison_file = output_path / "algorithm_comparison.csv"
        if comparison_file.exists():
            print(f"   • 算法对比: algorithm_comparison.csv")
    
    # 显示下一步操作建议
    print(f"\n💡 下一步操作:")
    if status == 'completed':
        print("   • 实验已完成，可以查看结果文件")
        print("   • 如需重新实验，请运行: python resume_experiment.py")
    elif status == 'running':
        print("   • 实验已暂停，恢复运行: python resume_experiment.py")
        print("   • 重新开始实验: python resume_experiment.py (选择重新开始)")
    elif status == 'not_started':
        print("   • 开始新实验: python pausable_experiment.py")
    else:
        print("   • 状态未知，建议检查进度文件")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='查看语音降噪实验状态')
    parser.add_argument('--output_root', type=str, default='./output', help='输出根目录')
    
    args = parser.parse_args()
    
    view_experiment_status(args.output_root)


if __name__ == "__main__":
    main()
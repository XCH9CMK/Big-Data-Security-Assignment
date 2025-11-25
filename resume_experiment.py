"""
实验恢复程序
用于恢复之前暂停的语音降噪实验
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse


def load_progress(output_root="./output"):
    """加载实验进度"""
    progress_file = Path(output_root) / "experiment_progress.json"
    
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载进度文件失败: {str(e)}")
        return None


def display_progress_info(progress_data):
    """显示实验进度信息"""
    print("="*70)
    print("📋 实验进度信息")
    print("="*70)
    
    # 基本信息
    print(f"🕐 开始时间: {progress_data.get('start_time', 'Unknown')}")
    if 'last_update' in progress_data:
        print(f"📅 最后更新: {progress_data['last_update']}")
    print(f"📊 实验状态: {progress_data.get('status', 'Unknown')}")
    
    # 实验配置
    config = progress_data.get('experiment_config', {})
    if config:
        print(f"\n⚙️ 实验配置:")
        print(f"   • 总样本数: {config.get('num_samples', 'N/A')}")
        print(f"   • 算法: {config.get('algorithm', 'N/A')}")
        print(f"   • 评估文件数: {config.get('max_eval', 'N/A')}")
        print(f"   • 批处理大小: {config.get('batch_size', 'N/A')}")
    
    # 算法进度
    print(f"\n🔧 算法进度:")
    completed_algs = progress_data.get('completed_algorithms', [])
    alg_progress = progress_data.get('algorithm_progress', {})
    
    if not alg_progress:
        print("   • 暂无进度记录")
    else:
        total_processed = 0
        for alg, count in alg_progress.items():
            status = "✅ 已完成" if alg in completed_algs else "🔄 进行中"
            print(f"   • {alg}: {count} 个文件 {status}")
            total_processed += count
        
        print(f"\n📈 总计处理: {total_processed} 个文件")
        print(f"✅ 已完成算法: {len(completed_algs)}")
    
    print("="*70)


def ask_resume_confirmation():
    """询问用户是否恢复实验"""
    print("\n🤔 选择操作:")
    print("1. 恢复实验 (继续之前的进度)")
    print("2. 重新开始 (清除进度，从头开始)")
    print("3. 取消")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        if choice == '1':
            return 'resume'
        elif choice == '2':
            return 'restart'
        elif choice == '3':
            return 'cancel'
        else:
            print("❌ 无效选择，请输入 1、2 或 3")


def clear_progress(output_root="./output"):
    """清除实验进度"""
    progress_file = Path(output_root) / "experiment_progress.json"
    
    try:
        if progress_file.exists():
            progress_file.unlink()
            print(f"✅ 已清除进度文件: {progress_file}")
        else:
            print("ℹ️ 没有找到进度文件")
        return True
    except Exception as e:
        print(f"❌ 清除进度文件失败: {str(e)}")
        return False


def construct_resume_command(progress_data, output_root="./output"):
    """构造恢复实验的命令"""
    config = progress_data.get('experiment_config', {})
    
    # 基本命令
    cmd_parts = ["python", "pausable_experiment.py", "--resume"]
    
    # 添加配置参数
    if 'num_samples' in config:
        cmd_parts.extend(["--num_samples", str(config['num_samples'])])
    
    if 'algorithm' in config:
        cmd_parts.extend(["--algorithm", config['algorithm']])
    
    if 'max_eval' in config:
        cmd_parts.extend(["--max_eval", str(config['max_eval'])])
    
    if 'batch_size' in config:
        cmd_parts.extend(["--batch_size", str(config['batch_size'])])
    
    if output_root != "./output":
        cmd_parts.extend(["--output_root", output_root])
    
    return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description='恢复语音降噪实验')
    parser.add_argument('--output_root', type=str, default='./output', help='输出根目录')
    parser.add_argument('--auto', action='store_true', help='自动恢复，不询问确认')
    
    args = parser.parse_args()
    
    print("🔄 语音降噪实验恢复程序")
    print("="*50)
    
    # 加载进度信息
    progress_data = load_progress(args.output_root)
    
    if not progress_data:
        print("❌ 没有找到有效的实验进度文件")
        print("💡 请确保之前运行过 pausable_experiment.py")
        print(f"📁 查找位置: {Path(args.output_root) / 'experiment_progress.json'}")
        return
    
    # 显示进度信息
    display_progress_info(progress_data)
    
    # 检查实验状态
    status = progress_data.get('status', 'unknown')
    if status == 'completed':
        print("\n✅ 实验已完成，无需恢复")
        print("💡 如需重新运行，请删除进度文件或使用 --restart 参数")
        return
    
    if status == 'not_started':
        print("\n⚠️ 实验尚未开始")
        print("💡 请先运行 pausable_experiment.py 开始实验")
        return
    
    # 询问用户操作
    if args.auto:
        action = 'resume'
        print("\n🚀 自动恢复模式")
    else:
        action = ask_resume_confirmation()
    
    if action == 'resume':
        # 构造恢复命令
        resume_cmd = construct_resume_command(progress_data, args.output_root)
        
        print(f"\n🔄 恢复实验...")
        print(f"📝 执行命令: {resume_cmd}")
        print("\n" + "="*50)
        
        # 执行恢复命令
        os.system(resume_cmd)
        
    elif action == 'restart':
        # 清除进度并重新开始
        if clear_progress(args.output_root):
            config = progress_data.get('experiment_config', {})
            
            # 构造新实验命令（不带 --resume）
            cmd_parts = ["python", "pausable_experiment.py"]
            
            if 'num_samples' in config:
                cmd_parts.extend(["--num_samples", str(config['num_samples'])])
            
            if 'algorithm' in config:
                cmd_parts.extend(["--algorithm", config['algorithm']])
            
            if 'max_eval' in config:
                cmd_parts.extend(["--max_eval", str(config['max_eval'])])
            
            if 'batch_size' in config:
                cmd_parts.extend(["--batch_size", str(config['batch_size'])])
            
            if args.output_root != "./output":
                cmd_parts.extend(["--output_root", args.output_root])
            
            new_cmd = " ".join(cmd_parts)
            
            print(f"\n🚀 重新开始实验...")
            print(f"📝 执行命令: {new_cmd}")
            print("\n" + "="*50)
            
            # 执行新实验命令
            os.system(new_cmd)
        
    else:  # cancel
        print("\n❌ 操作已取消")


if __name__ == "__main__":
    main()
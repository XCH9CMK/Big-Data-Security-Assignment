"""
数据路径验证脚本
检查VCTK数据集是否正确放置
"""

from pathlib import Path
from data_preparation import DataPreparation

def test_data_paths():
    print("🔍 数据路径验证")
    print("=" * 50)
    
    # 初始化数据准备模块
    try:
        data_prep = DataPreparation(data_root="./data")
        
        print(f"📁 数据根目录: {data_prep.data_root}")
        print(f"📁 VCTK路径: {data_prep.vctk_path}")
        print(f"📁 干净音频路径: {data_prep.clean_path}")
        print(f"📁 含噪音频路径: {data_prep.noisy_path}")
        
        # 检查路径是否存在
        paths_status = {
            "VCTK目录": data_prep.vctk_path.exists(),
            "干净音频目录": data_prep.clean_path.exists(),
            "含噪音频目录": data_prep.noisy_path.exists(),
        }
        
        print("\n📊 路径状态检查:")
        for path_name, exists in paths_status.items():
            status = "✅" if exists else "❌"
            print(f"   {path_name}: {status}")
        
        # 统计文件数量
        if data_prep.clean_path.exists():
            clean_count = len(list(data_prep.clean_path.glob("*.wav")))
            print(f"\n📈 干净音频文件数: {clean_count}")
        
        if data_prep.noisy_path.exists():
            noisy_count = len(list(data_prep.noisy_path.glob("*.wav")))
            print(f"📈 含噪音频文件数: {noisy_count}")
            
        if hasattr(data_prep, 'txt_path') and data_prep.txt_path and data_prep.txt_path.exists():
            txt_count = len(list(data_prep.txt_path.glob("*.txt")))
            print(f"📈 参考文本文件数: {txt_count}")
        
        # 验证是否可以开始实验
        if all(paths_status.values()) and clean_count > 0 and noisy_count > 0:
            print(f"\n✅ 数据验证通过！可以开始实验")
            return True
        else:
            print(f"\n❌ 数据验证失败，请检查数据放置")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程出错: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_paths()
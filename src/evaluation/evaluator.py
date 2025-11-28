"""
评估器主模块
整合各种评估指标的计算和结果统计
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from .mcd import MCDCalculator
from .wer import WERCalculator
from .snr import SNRCalculator
from .pesq import PESQCalculator


class Evaluator:
    """评估器主类"""
    
    def __init__(self, use_gpu=True, whisper_model_size="base", n_jobs=1):
        """
        初始化评估器
        
        Args:
            use_gpu: 是否使用GPU加速
            whisper_model_size: Whisper模型大小 ("tiny", "base", "small", "medium", "large")
            n_jobs: 并行处理的线程数（1=顺序处理，-1=使用所有CPU核心，>1=指定数量）
        """
        self.use_gpu = use_gpu
        self.whisper_model_size = whisper_model_size
        
        # 确定并行线程数（仅用于MCD计算）
        if n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        elif n_jobs > 0:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
        
        # 初始化各个计算器
        self.mcd_calculator = MCDCalculator()
        self.wer_calculator = WERCalculator(use_gpu=use_gpu, whisper_model_size=whisper_model_size)
        self.snr_calculator = SNRCalculator()
        self.pesq_calculator = PESQCalculator()
        
        # 默认参考文本
        self.default_reference_text = self.wer_calculator.default_reference_text
    
    def _get_reference_text_from_file(self, clean_file_path):
        """
        从VCTK数据集的txt文件中获取参考文本
        
        Args:
            clean_file_path: 干净音频文件路径
        
        Returns:
            str: 参考文本，如果找不到则返回None
        """
        clean_path = Path(clean_file_path)
        
        # 尝试多种方式匹配txt文件
        txt_candidates = []
        
        # 方式1: 如果clean文件本身就是VCTK格式（如p232_001.wav）
        if '_' in clean_path.stem and not clean_path.name.startswith('clean_'):
            # 直接匹配txt文件
            vctk_txt_path = clean_path.parent.parent / "txt"
            txt_candidates.append(vctk_txt_path / f"{clean_path.stem}.txt")
        
        # 方式2: 如果clean文件是clean_0000.wav格式，尝试找到原始VCTK文件
        if clean_path.name.startswith('clean_'):
            vctk_txt_path = clean_path.parent.parent / "txt"
            if vctk_txt_path.exists():
                # 尝试提取索引号
                try:
                    idx = int(clean_path.stem.replace('clean_', ''))
                    # 列出所有txt文件，按索引匹配（不准确，但可以尝试）
                    txt_files = sorted(list(vctk_txt_path.glob("*.txt")))
                    if idx < len(txt_files):
                        txt_candidates.append(txt_files[idx])
                except:
                    pass
        
        # 方式3: 直接匹配文件名（去掉clean_前缀，添加.txt）
        base_name = clean_path.stem.replace('clean_', '')
        vctk_txt_path = clean_path.parent.parent / "txt"
        if vctk_txt_path.exists():
            # 尝试p232_001格式
            txt_candidates.append(vctk_txt_path / f"{base_name}.txt")
            # 尝试直接匹配
            txt_candidates.append(vctk_txt_path / clean_path.name.replace('.wav', '.txt'))
        
        # 尝试读取第一个存在的txt文件
        for txt_file in txt_candidates:
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            return text
                except Exception as e:
                    continue
        
        return None
    
    def evaluate_denoising(self, clean_files, noisy_files, denoised_files, output_csv="evaluation_results.csv", compute_snr=False):
        """
        评估降噪效果（MCD和WER，可选SNR）
        
        Args:
            clean_files: 干净音频文件列表
            noisy_files: 含噪音频文件列表
            denoised_files: 降噪后音频文件列表
            output_csv: 结果保存路径
            compute_snr: 是否计算SNR（默认False，因为需要额外计算时间）
        
        Returns:
            results_df: 评估结果DataFrame
        """
        results = []
        
        print("开始评估降噪效果...")
        print(f"评估指标: MCD, WER" + (", SNR" if compute_snr else ""))
        
        # 预先加载Whisper模型（避免在循环中重复检查）
        # 模型会在第一次调用 compute_wer 时自动加载
        
        self.mcd_calculator.clear_cache()
        
        print("计算MCD指标...")
        mcd_results = [None] * len(clean_files)
        
        def compute_mcd_pair(args):
            """计算一对文件的MCD（用于并行处理）"""
            i, clean_file, noisy_file, denoised_file = args
            try:
                # MCD值越小越好，表示更接近clean
                mcd_noisy = self.mcd_calculator.compute_mcd(clean_file, noisy_file)
                mcd_denoised = self.mcd_calculator.compute_mcd(clean_file, denoised_file)
                # improvement = mcd_noisy - mcd_denoised
                # 如果improvement > 0，说明denoised更接近clean（降噪有效）
                # 如果improvement < 0，说明denoised反而更远离clean（降噪效果差）
                mcd_improvement = mcd_noisy - mcd_denoised
                
                # 调试：如果improvement异常，打印警告
                if mcd_improvement < -0.5:  # 如果improvement明显为负（降低阈值以捕获更多异常）
                    print(f"\n警告: 文件 {i} 的MCD improvement异常 ({mcd_improvement:.2f})")
                    print(f"  clean: {Path(clean_file).name}")
                    print(f"  noisy: {Path(noisy_file).name}")
                    print(f"  denoised: {Path(denoised_file).name}")
                    print(f"  MCD(clean,noisy)={mcd_noisy:.2f}, MCD(clean,denoised)={mcd_denoised:.2f}")
                    print(f"  说明: 降噪后的MCD反而更高，可能文件不匹配或降噪效果差")
                
                return i, {
                    'mcd_noisy': mcd_noisy,
                    'mcd_denoised': mcd_denoised,
                    'mcd_improvement': mcd_improvement
                }
            except Exception as e:
                print(f"\n计算MCD时出错 (文件 {i}): {e}")
                return i, {
                    'mcd_noisy': None,
                    'mcd_denoised': None,
                    'mcd_improvement': None
                }
        
        if self.n_jobs > 1:
            max_workers = min(self.n_jobs, len(clean_files))
            print(f"使用 {max_workers} 个线程并行计算MCD（共 {len(clean_files)} 个文件）...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    (i, clean_file, noisy_file, denoised_file)
                    for i, (clean_file, noisy_file, denoised_file) in enumerate(
                        zip(clean_files, noisy_files, denoised_files)
                    )
                ]
                
                futures = {executor.submit(compute_mcd_pair, task): task for task in tasks}
                
                with tqdm(total=len(tasks), desc="MCD计算", mininterval=1.0) as pbar:
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            i, result = future.result(timeout=120)
                            mcd_results[i] = result
                            pbar.update(1)
                        except FutureTimeoutError:
                            print(f"\n警告: 文件 {task[0]} 的MCD计算超时，跳过")
                            mcd_results[task[0]] = {
                                'mcd_noisy': None,
                                'mcd_denoised': None,
                                'mcd_improvement': None
                            }
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n计算MCD时出错 (文件 {task[0]}): {e}")
                            mcd_results[task[0]] = {
                                'mcd_noisy': None,
                                'mcd_denoised': None,
                                'mcd_improvement': None
                            }
                            pbar.update(1)
        else:
            # 顺序处理
            for i, (clean_file, noisy_file, denoised_file) in enumerate(tqdm(
                zip(clean_files, noisy_files, denoised_files), 
                total=len(clean_files),
                desc="MCD计算"
            )):
                _, result = compute_mcd_pair((i, clean_file, noisy_file, denoised_file))
                mcd_results[i] = result
        
        self.mcd_calculator.clear_cache()
        
        print("计算WER指标...")
        all_audio_files = []
        reference_texts = []
        
        for idx, (clean_file, noisy_file, denoised_file) in enumerate(zip(clean_files, noisy_files, denoised_files)):
            ref_text = self._get_reference_text_from_file(clean_file)
            if ref_text is None:
                ref_text = self.default_reference_text
                if idx == 0:
                    print("警告: 部分文件未找到对应的txt参考文本，将使用默认文本")
            
            all_audio_files.extend([noisy_file, denoised_file])
            reference_texts.extend([ref_text, ref_text])
        
        wer_results = [None] * len(all_audio_files)
        
        for idx, (audio_file, ref_text) in enumerate(tqdm(zip(all_audio_files, reference_texts), total=len(all_audio_files), desc="WER计算")):
            try:
                wer_score, _ = self.wer_calculator.compute_wer(audio_file, reference_text=ref_text)
                wer_results[idx] = wer_score
            except Exception as e:
                print(f"\n计算WER时出错 (文件 {audio_file}): {e}")
                wer_results[idx] = None
        
        for i, (clean_file, noisy_file, denoised_file) in enumerate(
            zip(clean_files, noisy_files, denoised_files)
        ):
            try:
                mcd_data = mcd_results[i]
                wer_noisy = wer_results[i * 2]
                wer_denoised = wer_results[i * 2 + 1]
                
                if wer_noisy is not None and wer_denoised is not None:
                    wer_improvement = wer_noisy - wer_denoised
                else:
                    wer_improvement = None
                
                snr_noisy = None
                snr_denoised = None
                snr_improvement = None
                if compute_snr:
                    try:
                        snr_noisy = self.snr_calculator.compute_snr(clean_file, noisy_file)
                        snr_denoised = self.snr_calculator.compute_snr(clean_file, denoised_file)
                        if snr_noisy is not None and snr_denoised is not None:
                            snr_improvement = snr_denoised - snr_noisy
                    except Exception as e:
                        print(f"\n计算SNR时出错 (文件 {i}): {e}")
                
                result = {
                    'file_index': i,
                    'clean_file': clean_file,
                    'noisy_file': noisy_file,
                    'denoised_file': denoised_file,
                    'mcd_noisy': mcd_data['mcd_noisy'],
                    'mcd_denoised': mcd_data['mcd_denoised'],
                    'mcd_improvement': mcd_data['mcd_improvement'],
                    'wer_noisy': wer_noisy,
                    'wer_denoised': wer_denoised,
                    'wer_improvement': wer_improvement
                }
                
                if compute_snr:
                    result['snr_noisy'] = snr_noisy
                    result['snr_denoised'] = snr_denoised
                    result['snr_improvement'] = snr_improvement
                
                results.append(result)
            
            except Exception as e:
                print(f"\n评估文件 {i} 时出错: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n评估结果已保存到: {output_csv}")
        
        self._print_statistics(results_df)
        
        return results_df
    
    def _sanitize_inf_values(self, series):
        """将inf值替换为NaN"""
        return series.replace([np.inf, -np.inf], np.nan)
    
    def _calculate_metric_stats(self, results_df, metric_name, metric_type='lower_better', format_str='.2f'):
        """
        计算指标统计信息
        
        Args:
            results_df: 结果DataFrame
            metric_name: 指标名称（如'mcd', 'wer', 'snr'）
            metric_type: 'lower_better' 或 'higher_better'
            format_str: 格式化字符串
        
        Returns:
            dict: 包含统计信息的字典，如果指标不存在则返回None
        """
        noisy_col = f'{metric_name}_noisy'
        denoised_col = f'{metric_name}_denoised'
        improvement_col = f'{metric_name}_improvement'
        
        if noisy_col not in results_df.columns or not results_df[noisy_col].notna().any():
            return None
        
        stats = {}
        
        if metric_name == 'snr':
            noisy_series = self._sanitize_inf_values(results_df[noisy_col])
            denoised_series = self._sanitize_inf_values(results_df[denoised_col])
        else:
            noisy_series = results_df[noisy_col]
            denoised_series = results_df[denoised_col]
        
        if noisy_series.notna().any():
            stats['noisy_mean'] = noisy_series.mean()
            stats['denoised_mean'] = denoised_series.mean()
            
            if improvement_col in results_df.columns:
                if metric_name == 'snr':
                    improvement_series = self._sanitize_inf_values(results_df[improvement_col])
                else:
                    improvement_series = results_df[improvement_col]
                
                if improvement_series.notna().any():
                    stats['improvement_mean'] = improvement_series.mean()
        
        stats['metric_type'] = metric_type
        stats['format_str'] = format_str
        return stats
    
    def _print_statistics(self, results_df):
        """打印评估统计信息"""
        print("\n" + "="*60)
        print("评估统计结果")
        print("="*60)
        
        # MCD统计
        mcd_stats = self._calculate_metric_stats(results_df, 'mcd', 'lower_better', '.2f')
        if mcd_stats:
            print(f"\nMCD (梅尔倒谱失真) - 越低越好:")
            print(f"  含噪音频平均MCD: {mcd_stats['noisy_mean']:.2f}")
            print(f"  降噪后平均MCD: {mcd_stats['denoised_mean']:.2f}")
            if 'improvement_mean' in mcd_stats:
                print(f"  平均MCD改善: {mcd_stats['improvement_mean']:.2f} (正值表示改善)")
        
        # WER统计
        wer_stats = self._calculate_metric_stats(results_df, 'wer', 'lower_better', '.4f')
        if wer_stats:
            print(f"\nWER (错词率) - 越低越好:")
            print(f"  含噪音频平均WER: {wer_stats['noisy_mean']:.4f}")
            print(f"  降噪后平均WER: {wer_stats['denoised_mean']:.4f}")
            if 'improvement_mean' in wer_stats:
                print(f"  平均WER改善: {wer_stats['improvement_mean']:.4f} (正值表示改善)")
        
        # SNR统计
        snr_stats = self._calculate_metric_stats(results_df, 'snr', 'higher_better', '.2f')
        if snr_stats:
            print(f"\nSNR (信噪比) - 越高越好:")
            print(f"  含噪音频平均SNR: {snr_stats['noisy_mean']:.2f} dB")
            print(f"  降噪后平均SNR: {snr_stats['denoised_mean']:.2f} dB")
            if 'improvement_mean' in snr_stats:
                print(f"  平均SNR改善: {snr_stats['improvement_mean']:.2f} dB (正值表示改善)")
        
        print("="*60)
    
    def generate_comparison_report(self, all_results, output_csv="algorithm_comparison.csv"):
        """
        生成算法比较报告
        
        Args:
            all_results: 字典，键为算法名称，值为评估结果DataFrame
            output_csv: 比较报告保存路径
        
        Returns:
            comparison_df: 比较结果DataFrame
        """
        if not all_results:
            return None
        
        print("\n" + "="*60)
        print("算法性能比较")
        print("="*60)
        
        comparison = []
        
        for alg_name, results_df in all_results.items():
            metrics = {'Algorithm': alg_name}
            
            # 使用统一的统计计算方法
            for metric_name in ['mcd', 'wer', 'snr']:
                stats = self._calculate_metric_stats(results_df, metric_name)
                if stats:
                    metrics[f'Avg_{metric_name.upper()}_Denoised'] = stats['denoised_mean']
                    if 'improvement_mean' in stats:
                        metrics[f'Avg_{metric_name.upper()}_Improvement'] = stats['improvement_mean']
            
            comparison.append(metrics)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\n")
        print(comparison_df.to_string(index=False))
        
        # 保存比较结果
        if output_csv:
            comparison_df.to_csv(output_csv, index=False)
            print(f"\n比较结果已保存到: {output_csv}")
        
        return comparison_df
    
    def evaluate_single_pair(self, clean_file, noisy_file, denoised_file, verbose=True, compute_snr=False):
        """
        评估单对音频文件
        
        Args:
            clean_file: 干净音频文件
            noisy_file: 含噪音频文件
            denoised_file: 降噪后音频文件
            verbose: 是否打印详细信息
            compute_snr: 是否计算SNR（默认False）
        
        Returns:
            result: 评估结果字典
        """
        # 计算MCD
        try:
            mcd_noisy = self.mcd_calculator.compute_mcd(clean_file, noisy_file)
            mcd_denoised = self.mcd_calculator.compute_mcd(clean_file, denoised_file)
            mcd_improvement = mcd_noisy - mcd_denoised
        except Exception as e:
            if verbose:
                print(f"MCD计算失败: {e}")
            mcd_noisy = None
            mcd_denoised = None
            mcd_improvement = None
        
        # 计算WER
        try:
            wer_noisy, _ = self.wer_calculator.compute_wer(noisy_file)
            wer_denoised, _ = self.wer_calculator.compute_wer(denoised_file)
            wer_improvement = wer_noisy - wer_denoised
        except Exception as e:
            if verbose:
                print(f"WER计算失败: {e}")
            wer_noisy = None
            wer_denoised = None
            wer_improvement = None
        
        result = {
            'mcd_noisy': mcd_noisy,
            'mcd_denoised': mcd_denoised,
            'mcd_improvement': mcd_improvement,
            'wer_noisy': wer_noisy,
            'wer_denoised': wer_denoised,
            'wer_improvement': wer_improvement
        }
        
        # 可选：计算SNR
        if compute_snr:
            try:
                snr_noisy = self.snr_calculator.compute_snr(clean_file, noisy_file)
                snr_denoised = self.snr_calculator.compute_snr(clean_file, denoised_file)
                if snr_noisy is not None and snr_denoised is not None:
                    snr_improvement = snr_denoised - snr_noisy
                else:
                    snr_improvement = None
                result['snr_noisy'] = snr_noisy
                result['snr_denoised'] = snr_denoised
                result['snr_improvement'] = snr_improvement
            except Exception as e:
                if verbose:
                    print(f"SNR计算失败: {e}")
                result['snr_noisy'] = None
                result['snr_denoised'] = None
                result['snr_improvement'] = None
        
        if verbose:
            print("\n单文件评估结果:")
            if mcd_noisy is not None:
                print(f"  MCD (含噪): {mcd_noisy:.2f}")
                print(f"  MCD (降噪后): {mcd_denoised:.2f}")
                print(f"  MCD 改善: {mcd_improvement:.2f}")
            if wer_noisy is not None:
                print(f"  WER (含噪): {wer_noisy:.4f}")
                print(f"  WER (降噪后): {wer_denoised:.4f}")
                print(f"  WER 改善: {wer_improvement:.4f}")
            if compute_snr and 'snr_noisy' in result and result['snr_noisy'] is not None:
                print(f"  SNR (含噪): {result['snr_noisy']:.2f} dB")
                print(f"  SNR (降噪后): {result['snr_denoised']:.2f} dB")
                if result['snr_improvement'] is not None:
                    print(f"  SNR 改善: {result['snr_improvement']:.2f} dB")
        
        return result


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sleep-EDF 数据预处理模块 - 升级版

融合 file_matcher.py 的模块化设计 和 data_preprocessing.py 的完整功能

核心功能：
1. 文件匹配：自动识别并匹配 EEG 数据文件和标签文件
2. 标签解析：灵活解析多种标签格式，筛选有效标签（R/1/2/3）
3. 信号滤波：50Hz 陷波滤波 + 0.5-30Hz 带通滤波
4. 帧分割：按 30 秒/帧切分，与标签一一对应
5. 数据保存：保存为 NumPy 格式，便于后续特征提取

作者：SEU-BME 竞赛团队
日期：2026-03-19
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import signal
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ====================== 配置类 ======================
class Config:
    """配置类 - 集中管理所有参数"""
    
    # 路径配置
    DATA_PATH = r"..\人工智能生物医学大数据赛题2—数据集\Train_set"
    SAVE_PATH = r"..\data\processed"
    
    # 信号处理参数
    FS = 100  # 采样率 (Hz)
    FRAME_DURATION = 30  # 帧长度 (秒)
    FRAME_POINTS = FS * FRAME_DURATION  # 每帧数据点数
    
    # 滤波参数
    NOTCH_FREQ = 50.0  # 陷波频率 (Hz)
    NOTCH_Q = 27.5  # 陷波品质因数
    BANDPASS_LOW = 0.5  # 带通低频 (Hz)
    BANDPASS_HIGH = 30.0  # 带通高频 (Hz)
    BANDPASS_ORDER = 5  # 滤波器阶数
    
    # 标签映射
    LABEL_MAP = {
        "Sleep stage R": "R", "Sleep stage 1": "1", 
        "Sleep stage 2": "2", "Sleep stage 3": "3",
        "R": "R", "1": "1", "2": "2", "3": "3",
        "REM": "R", "N1": "1", "N2": "2", "N3": "3"
    }
    
    # 有效标签（竞赛要求：仅 R/1/2/3）
    VALID_LABELS = ['R', '1', '2', '3']


# ====================== 文件匹配类 ======================
class FileMatcher:
    """
    文件匹配类 - 负责 EEG 数据文件与标签文件的匹配
    """
    
    def __init__(self, data_path: str):
        """
        初始化文件匹配器
        
        参数:
            data_path: 数据根目录路径
        """
        self.data_path = data_path
        self.eeg_files = []  # EEG 数据文件列表
        self.label_files = []  # 标签文件列表
        self.matched_pairs = []  # 匹配成功的文件对
        self.unmatched_files = []  # 未匹配的文件
        
        logger.info(f"初始化文件匹配器，数据路径: {data_path}")
    
    def scan_files(self) -> None:
        """
        扫描目录，分类 EEG 数据文件和标签文件
        """
        logger.info("开始扫描文件...")
        
        # 遍历所有子目录
        for root, dirs, files in os.walk(self.data_path):
            for file_name in files:
                # 过滤条件：仅保留 txt 文件
                if not file_name.endswith('.txt'):
                    continue
                
                # 过滤条件：排除滤波后的冗余文件
                if 'filtered' in file_name.lower():
                    continue
                
                # 分类文件
                if 'EEGFpz_Cz' in file_name:
                    self.eeg_files.append(os.path.join(root, file_name))
                elif 'Hypnogram' in file_name:
                    self.label_files.append(os.path.join(root, file_name))
        
        logger.info(f"扫描完成：找到 {len(self.eeg_files)} 个 EEG 文件，{len(self.label_files)} 个标签文件")
    
    def extract_match_key(self, file_path: str) -> Optional[str]:
        """
        从文件路径中提取匹配键
        
        参数:
            file_path: 文件完整路径
            
        返回:
            匹配键 (如 "ST7011J0_Part1") 或 None
        """
        try:
            file_name = os.path.basename(file_path)
            
            # 提取被试 ID（下划线分割第一个部分）
            subject_id = file_name.split('_')[0]
            
            # 兼容两种格式：Part1 或 Part_1
            part_match = re.search(r'Part[_ ]*(\d+)', file_name)
            if part_match:
                part_num = part_match.group(1)
                return f"{subject_id}_Part{part_num}"
            else:
                logger.warning(f"未找到 Part 编号: {file_name}")
                return None
                
        except Exception as e:
            logger.error(f"文件名解析失败: {file_path} | 错误: {e}")
            return None
    
    def match_files(self) -> None:
        """
        匹配 EEG 文件和标签文件
        """
        logger.info("开始匹配文件...")
        
        # 构建标签文件字典
        label_dict = {}
        for label_path in self.label_files:
            key = self.extract_match_key(label_path)
            if key:
                label_dict[key] = label_path
        
        # 遍历 EEG 文件，查找匹配的标签
        for eeg_path in self.eeg_files:
            key = self.extract_match_key(eeg_path)
            if key and key in label_dict:
                self.matched_pairs.append({
                    'key': key,
                    'eeg_path': eeg_path,
                    'label_path': label_dict[key]
                })
            else:
                self.unmatched_files.append(eeg_path)
        
        logger.info(f"匹配完成：成功 {len(self.matched_pairs)} 对，失败 {len(self.unmatched_files)} 个")
        
        if self.unmatched_files:
            logger.warning(f"未匹配文件示例: {self.unmatched_files[:3]}")


# ====================== 标签解析类 ======================
class LabelParser:
    """
    标签解析类 - 负责解析和清洗标签文件
    """
    
    def __init__(self):
        """初始化标签解析器"""
        self.label_map = Config.LABEL_MAP
        self.valid_labels = Config.VALID_LABELS
        logger.info("初始化标签解析器")
    
    def parse_label_file(self, label_path: str) -> Optional[pd.DataFrame]:
        """
        解析标签文件
        
        参数:
            label_path: 标签文件路径
            
        返回:
            标签 DataFrame 或 None
        """
        try:
            # 读取所有行
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # 调试：打印前3行
            logger.debug(f"标签文件格式 - {os.path.basename(label_path)}:")
            for i, line in enumerate(lines[:3]):
                logger.debug(f"  第{i+1}行: {line}")
            
            # 解析每一行
            label_rows = []
            for line_num, line in enumerate(lines, 1):
                try:
                    # 跳过表头行
                    if any(keyword in line.lower() for keyword in 
                           ['onset', 'start', 'end', 'duration', 'label', 'description']):
                        continue
                    
                    # 用正则分割任意空白符/逗号
                    parts = re.split(r'[\s,]+', line)
                    parts = [p for p in parts if p]  # 过滤空字符串
                    
                    # 解析时间和标签
                    if len(parts) >= 4:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        duration = float(parts[2])
                        label_str = ' '.join(parts[3:])  # 标签可能包含空格
                    elif len(parts) == 3:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        duration = end_time - start_time
                        label_str = parts[2]
                    else:
                        logger.warning(f"第{line_num}行格式异常: {line}")
                        continue
                    
                    # 清理标签字符串
                    label_str = label_str.strip().replace('"', '').replace("'", "")
                    
                    label_rows.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'label': label_str
                    })
                    
                except Exception as e:
                    logger.warning(f"第{line_num}行解析失败: {line} | 错误: {e}")
                    continue
            
            return pd.DataFrame(label_rows)
            
        except Exception as e:
            logger.error(f"读取标签文件失败: {label_path} | 错误: {e}")
            return None
    
    def filter_valid_labels(self, label_df: pd.DataFrame) -> pd.DataFrame:
        """
        筛选有效标签（仅保留 R/1/2/3）
        
        参数:
            label_df: 原始标签 DataFrame
            
        返回:
            清洗后的标签 DataFrame
        """
        if label_df.empty:
            return label_df
        
        # 定义有效标签判断函数
        def is_valid_label(lbl: str) -> bool:
            lbl_lower = lbl.lower()
            return any(keyword in lbl_lower for keyword in ['r', '1', '2', '3', 'rem', 'n1', 'n2', 'n3'])
        
        # 筛选有效行
        valid_df = label_df[label_df['label'].apply(is_valid_label)].copy()
        
        # 映射标签
        def map_label(lbl: str) -> Optional[str]:
            # 精确匹配
            if lbl in self.label_map:
                return self.label_map[lbl]
            
            # 模糊匹配
            lbl_lower = lbl.lower()
            if 'r' in lbl_lower or 'rem' in lbl_lower:
                return 'R'
            elif '1' in lbl_lower or 'n1' in lbl_lower:
                return '1'
            elif '2' in lbl_lower or 'n2' in lbl_lower:
                return '2'
            elif '3' in lbl_lower or 'n3' in lbl_lower:
                return '3'
            return None
        
        valid_df['label'] = valid_df['label'].apply(map_label)
        
        # 移除映射失败的标签
        valid_df = valid_df.dropna(subset=['label'])
        
        logger.info(f"标签清洗: 原始 {len(label_df)} 行 -> 有效 {len(valid_df)} 行")
        
        return valid_df


# ====================== 信号处理类 ======================
class SignalProcessor:
    """
    信号处理类 - 负责滤波和帧分割
    """
    
    def __init__(self):
        """初始化信号处理器"""
        self.fs = Config.FS
        self.frame_points = Config.FRAME_POINTS
        logger.info("初始化信号处理器")
    
    def filter_signal(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        对脑电信号进行滤波
        
        处理流程：
        1. 50Hz 陷波滤波（去除工频干扰）
        2. 0.5-30Hz 带通滤波（保留睡眠相关脑电波）
        
        参数:
            eeg_data: 原始 EEG 信号
            
        返回:
            滤波后的 EEG 信号
        """
        if len(eeg_data) == 0:
            return eeg_data
        
        # 1. 50Hz 陷波滤波
        b, a = signal.iirnotch(Config.NOTCH_FREQ, Config.NOTCH_Q, self.fs)
        eeg_notch = signal.filtfilt(b, a, eeg_data)
        
        # 2. 0.5-30Hz 带通滤波
        b, a = signal.butter(
            Config.BANDPASS_ORDER,
            [Config.BANDPASS_LOW, Config.BANDPASS_HIGH],
            btype='bandpass',
            fs=self.fs
        )
        eeg_filtered = signal.filtfilt(b, a, eeg_notch)
        
        return eeg_filtered
    
    def split_into_frames(self, eeg_data: np.ndarray, 
                         label_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 EEG 信号按 30 秒/帧分割
        
        参数:
            eeg_data: 滤波后的 EEG 信号
            label_df: 清洗后的标签 DataFrame
            
        返回:
            frames: EEG 帧数组 (n_frames, 3000)
            labels: 标签数组 (n_frames,)
        """
        frames = []
        labels = []
        
        if len(eeg_data) == 0 or label_df.empty:
            return np.array(frames), np.array(labels)
        
        for _, row in label_df.iterrows():
            try:
                start_sec = float(row['start_time'])
                end_sec = float(row['end_time'])
                current_label = row['label']
                
                # 计算数据点索引
                start_idx = int(np.floor(start_sec * self.fs))
                end_idx = int(np.floor(end_sec * self.fs))
                
                # 边界检查
                if start_idx < 0 or end_idx > len(eeg_data):
                    continue
                
                segment_duration = end_sec - start_sec
                
                # 处理长标签段（按 30 秒切分）
                if segment_duration >= Config.FRAME_DURATION:
                    for sub_start in np.arange(start_sec, end_sec, Config.FRAME_DURATION):
                        sub_end = sub_start + Config.FRAME_DURATION
                        if sub_end > end_sec:
                            break
                        
                        sub_start_idx = int(np.floor(sub_start * self.fs))
                        sub_end_idx = int(np.floor(sub_end * self.fs))
                        
                        if sub_end_idx - sub_start_idx == self.frame_points:
                            frame = eeg_data[sub_start_idx:sub_end_idx]
                            frames.append(frame)
                            labels.append(current_label)
                
                # 刚好 30 秒的段
                elif segment_duration == Config.FRAME_DURATION:
                    if end_idx - start_idx == self.frame_points:
                        frame = eeg_data[start_idx:end_idx]
                        frames.append(frame)
                        labels.append(current_label)
                        
            except Exception as e:
                logger.warning(f"帧分割失败: {e}")
                continue
        
        return np.array(frames), np.array(labels)


# ====================== 主处理类 ======================
class DataPreprocessor:
    """
    数据预处理主类 - 整合所有功能
    """
    
    def __init__(self, data_path: str = None, save_path: str = None):
        """
        初始化数据预处理器
        
        参数:
            data_path: 数据根目录路径
            save_path: 保存路径
        """
        self.data_path = data_path or Config.DATA_PATH
        self.save_path = save_path or Config.SAVE_PATH
        
        # 初始化子模块
        self.file_matcher = FileMatcher(self.data_path)
        self.label_parser = LabelParser()
        self.signal_processor = SignalProcessor()
        
        # 存储处理结果
        self.processed_data = []
        
        logger.info("=" * 60)
        logger.info("数据预处理器初始化完成")
        logger.info(f"数据路径: {self.data_path}")
        logger.info(f"保存路径: {self.save_path}")
        logger.info("=" * 60)
    
    def process_all(self) -> None:
        """
        执行完整的数据预处理流程
        """
        start_time = datetime.now()
        logger.info("开始完整的数据预处理流程...")
        
        # 步骤 1: 扫描和匹配文件
        self.file_matcher.scan_files()
        self.file_matcher.match_files()
        
        if not self.file_matcher.matched_pairs:
            logger.error("没有匹配成功的文件对，处理终止")
            return
        
        # 步骤 2: 处理每一对匹配的文件
        for idx, pair in enumerate(self.file_matcher.matched_pairs, 1):
            try:
                logger.info(f"\n处理第 {idx}/{len(self.file_matcher.matched_pairs)} 组: {pair['key']}")
                
                # 读取 EEG 数据
                eeg_data = np.loadtxt(pair['eeg_path'])
                logger.info(f"  EEG 数据: {len(eeg_data)} 点 ({len(eeg_data)/Config.FS:.1f} 秒)")
                
                # 解析标签
                label_df = self.label_parser.parse_label_file(pair['label_path'])
                if label_df is None or label_df.empty:
                    logger.warning(f"  标签文件为空，跳过")
                    continue
                
                # 清洗标签
                clean_label_df = self.label_parser.filter_valid_labels(label_df)
                if clean_label_df.empty:
                    logger.warning(f"  没有有效标签，跳过")
                    continue
                
                # 滤波
                filtered_eeg = self.signal_processor.filter_signal(eeg_data)
                logger.info(f"  滤波完成")
                
                # 帧分割
                frames, labels = self.signal_processor.split_into_frames(filtered_eeg, clean_label_df)
                logger.info(f"  帧分割: {len(frames)} 帧")
                
                if len(frames) > 0:
                    # 统计标签分布
                    label_counts = pd.Series(labels).value_counts().to_dict()
                    logger.info(f"  标签分布: {label_counts}")
                    
                    # 存储结果
                    self.processed_data.append({
                        'key': pair['key'],
                        'eeg_path': pair['eeg_path'],
                        'label_path': pair['label_path'],
                        'frames': frames,
                        'labels': labels,
                        'frame_count': len(frames)
                    })
                
            except Exception as e:
                logger.error(f"处理失败 {pair['key']}: {e}")
                continue
        
        # 步骤 3: 保存结果
        self.save_results()
        
        # 打印统计信息
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info("数据预处理完成!")
        logger.info(f"总耗时: {elapsed_time:.1f} 秒")
        logger.info(f"处理文件组: {len(self.processed_data)}/{len(self.file_matcher.matched_pairs)}")
        
        total_frames = sum(item['frame_count'] for item in self.processed_data)
        logger.info(f"总帧数: {total_frames}")
        
        # 统计所有标签
        all_labels = []
        for item in self.processed_data:
            all_labels.extend(item['labels'].tolist())
        
        if all_labels:
            label_counts = pd.Series(all_labels).value_counts()
            logger.info(f"总标签分布:\n{label_counts}")
        
        logger.info("=" * 60)
    
    def save_results(self) -> None:
        """
        保存预处理后的数据
        """
        if not self.processed_data:
            logger.warning("没有可保存的数据")
            return
        
        # 创建保存目录
        os.makedirs(self.save_path, exist_ok=True)
        
        # 合并所有帧和标签
        all_frames = np.concatenate([item['frames'] for item in self.processed_data])
        all_labels = np.concatenate([item['labels'] for item in self.processed_data])
        
        # 保存为 NumPy 文件
        frames_path = os.path.join(self.save_path, 'eeg_frames.npy')
        labels_path = os.path.join(self.save_path, 'frame_labels.npy')
        
        np.save(frames_path, all_frames)
        np.save(labels_path, all_labels)
        
        logger.info(f"\n数据已保存:")
        logger.info(f"  帧数据: {frames_path} (shape: {all_frames.shape})")
        logger.info(f"  标签: {labels_path} (shape: {all_labels.shape})")


# ====================== 主函数 ======================
def main():
    """
    主函数 - 演示数据预处理流程
    """
    print("=" * 60)
    print("Sleep-EDF 数据预处理 - 升级版")
    print("=" * 60)
    
    # 创建预处理器并执行
    preprocessor = DataPreprocessor()
    preprocessor.process_all()
    
    print("\n" + "=" * 60)
    print("预处理完成！请查看日志文件 preprocessing.log 了解详细信息")
    print("=" * 60)


if __name__ == "__main__":
    main()

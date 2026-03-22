"""
N1 期 + R 期双优化版本
优化策略：
1. N1 期数据增强（2.5 倍）
2. R 期数据增强（3.5 倍）
3. SMOTE 过采样平衡所有类别
4. 添加 N1 和 R 期专属特征
5. 优化类别权重
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ====================== 第一步：配置路径 + 读取预处理数据 ======================
processed_data_path = r"..\data\processed"
frames_path = os.path.join(processed_data_path, "eeg_frames.npy")
labels_path = os.path.join(processed_data_path, "frame_labels.npy")

try:
    eeg_frames = np.load(frames_path)
    frame_labels = np.load(labels_path)
    print(f"✅ 成功读取预处理数据：")
    print(f"   - 脑电帧形状：{eeg_frames.shape}")
    print(f"   - 标签数量：{len(frame_labels)}")
    print(f"   - 标签分布：{pd.Series(frame_labels).value_counts().to_dict()}")
    print("=" * 60)
except FileNotFoundError:
    print(f"❌ 未找到 npy 文件，请检查路径：{processed_data_path}")
    exit()

# ====================== 第二步：N1 期和 R 期专属特征提取 ======================
def extract_n1_r_specific_features(frame, fs=100):
    """
    提取 N1 期和 R 期特异性特征
    """
    features = {}
    
    # 1. θ波（4-8Hz）特征
    b_theta, a_theta = signal.butter(4, [4, 8], btype='bandpass', fs=fs)
    theta_filtered = signal.filtfilt(b_theta, a_theta, frame)
    theta_power = np.var(theta_filtered)
    
    # 2. δ波（0.5-4Hz）特征
    b_delta, a_delta = signal.butter(4, [0.5, 4], btype='bandpass', fs=fs)
    delta_filtered = signal.filtfilt(b_delta, a_delta, frame)
    delta_power = np.var(delta_filtered)
    
    # 3. θ/δ比值（N1 期特征：比值较高）
    features['theta_delta_ratio'] = theta_power / (delta_power + 1e-6)
    
    # 4. α波（8-13Hz）衰减指数（R 期α波几乎消失）
    b_alpha, a_alpha = signal.butter(4, [8, 13], btype='bandpass', fs=fs)
    alpha_filtered = signal.filtfilt(b_alpha, a_alpha, frame)
    alpha_power = np.var(alpha_filtered)
    total_power = np.var(frame)
    features['alpha_attenuation'] = 1.0 - (alpha_power / (total_power + 1e-6))
    
    # 5. σ波/纺锤波（12-14Hz）检测（N2 期特征，R 期较少）
    b_sigma, a_sigma = signal.butter(4, [12, 14], btype='bandpass', fs=fs)
    sigma_filtered = signal.filtfilt(b_sigma, a_sigma, frame)
    sigma_power = np.var(sigma_filtered)
    features['sigma_power'] = sigma_power
    
    # 6. 锯齿波检测（2-6Hz，R 期特征）
    b_sawtooth, a_sawtooth = signal.butter(4, [2, 6], btype='bandpass', fs=fs)
    sawtooth_filtered = signal.filtfilt(b_sawtooth, a_sawtooth, frame)
    sawtooth_power = np.var(sawtooth_filtered)
    features['sawtooth_power'] = sawtooth_power
    
    # 7. β波（13-30Hz）特征（R 期较活跃）
    b_beta, a_beta = signal.butter(4, [13, 30], btype='bandpass', fs=fs)
    beta_filtered = signal.filtfilt(b_beta, a_beta, frame)
    beta_power = np.var(beta_filtered)
    features['beta_power_ratio'] = beta_power / (total_power + 1e-6)
    
    # 8. 高频/低频比值（R 期较高）
    b_high, a_high = signal.butter(4, 8, btype='highpass', fs=fs)
    high_filtered = signal.filtfilt(b_high, a_high, frame)
    high_power = np.var(high_filtered)
    features['high_low_ratio'] = high_power / (delta_power + 1e-6)
    
    # 9. 过零率（R 期较高）
    zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
    features['zero_cross_rate'] = zero_crossings / len(frame)
    
    # 10. 信号复杂度（R 期较复杂）
    features['complexity'] = np.std(np.diff(frame)) / np.std(frame)
    
    # 11. 峰度（N1 期峰度较低）
    features['kurtosis'] = stats.kurtosis(frame)
    
    # 12. 偏度
    features['skewness'] = stats.skew(frame)
    
    # 13. α波占空比（R 期α波少）
    alpha_envelope = np.abs(signal.hilbert(alpha_filtered))
    alpha_threshold = np.mean(alpha_envelope)
    alpha_duty_cycle = np.sum(alpha_envelope > alpha_threshold) / len(alpha_envelope)
    features['alpha_duty_cycle'] = alpha_duty_cycle
    
    # 14. θ波相对强度（R 期θ波明显）
    features['theta_relative'] = theta_power / (theta_power + delta_power + alpha_power + beta_power + 1e-6)
    
    # 15. 简化版熵值
    features['entropy_approx'] = np.log(np.var(frame) + 1) / np.log(np.mean(np.abs(frame)) + 1)
    
    return features

def extract_eeg_features(frame, fs=100):
    """
    完整特征提取：基础特征 + N1/R 专属特征
    """
    features = []
    
    # ----- 基础时域特征 -----
    features.extend([
        np.mean(frame),
        np.std(frame),
        np.max(frame),
        np.min(frame),
        np.ptp(frame),
        np.median(frame),
        stats.skew(frame),
        stats.kurtosis(frame),
        np.sum(np.diff(np.sign(frame)) != 0) / len(frame),
        np.percentile(frame, 25),
        np.percentile(frame, 75),
        np.percentile(frame, 90),
        np.percentile(frame, 10),
        np.sum(np.abs(frame)),
        np.sqrt(np.sum(frame ** 2)),
        np.max(np.abs(frame)),
        np.mean(np.abs(np.diff(frame))),
        np.std(np.diff(frame)),
        np.sum(np.abs(frame) > 2 * np.std(frame)) / len(frame)
    ])
    
    # ----- 基础频域特征 -----
    freqs = np.fft.rfftfreq(len(frame), 1/fs)
    fft_vals = np.abs(np.fft.rfft(frame)) ** 2
    
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    
    delta_mask = (freqs >= delta_band[0]) & (freqs < delta_band[1])
    theta_mask = (freqs >= theta_band[0]) & (freqs < theta_band[1])
    alpha_mask = (freqs >= alpha_band[0]) & (freqs < alpha_band[1])
    beta_mask = (freqs >= beta_band[0]) & (freqs < beta_band[1])
    
    delta_power = np.sum(fft_vals[delta_mask])
    theta_power = np.sum(fft_vals[theta_mask])
    alpha_power = np.sum(fft_vals[alpha_mask])
    beta_power = np.sum(fft_vals[beta_mask])
    total_power = np.sum(fft_vals) + 1e-6
    
    features.extend([
        delta_power / total_power,
        theta_power / total_power,
        alpha_power / total_power,
        beta_power / total_power,
        (alpha_power + beta_power) / (delta_power + theta_power + 1e-6),
        theta_power / delta_power,
        np.mean(fft_vals[delta_mask]) if np.any(delta_mask) else 0,
        np.mean(fft_vals[theta_mask]) if np.any(theta_mask) else 0,
        np.mean(fft_vals[alpha_mask]) if np.any(alpha_mask) else 0,
        np.mean(fft_vals[beta_mask]) if np.any(beta_mask) else 0,
        np.std(fft_vals[delta_mask]) if np.any(delta_mask) else 0,
        np.std(fft_vals[theta_mask]) if np.any(theta_mask) else 0,
        np.std(fft_vals[alpha_mask]) if np.any(alpha_mask) else 0,
        np.std(fft_vals[beta_mask]) if np.any(beta_mask) else 0,
        total_power,
        -np.sum((fft_vals / total_power) * np.log(fft_vals / total_power + 1e-6))
    ])
    
    # ----- N1 和 R 期专属特征 -----
    n1_r_features = extract_n1_r_specific_features(frame, fs)
    features.extend(list(n1_r_features.values()))
    
    return np.array(features)

# ====================== 第三步：N1 期和 R 期数据增强 ======================
def augment_minority_classes(frames, labels, n1_factor=2.5, r_factor=3.5):
    """
    对 N1 期和 R 期进行数据增强
    """
    n1_mask = (labels == '1')
    r_mask = (labels == 'R')
    
    n1_frames = frames[n1_mask]
    n1_labels = labels[n1_mask]
    r_frames = frames[r_mask]
    r_labels = labels[r_mask]
    
    augmented_frames = []
    augmented_labels = []
    
    # N1 期增强
    original_n1_count = len(n1_frames)
    target_n1_count = int(original_n1_count * n1_factor)
    
    print(f"🔄 对 N1 期进行数据增强...")
    print(f"   原始 N1 样本数：{original_n1_count}")
    print(f"   目标 N1 样本数：{target_n1_count}")
    
    for i in range(target_n1_count - original_n1_count):
        idx = i % original_n1_count
        frame = n1_frames[idx].copy()
        
        # 随机选择增强方式
        aug_type = np.random.choice(['noise', 'scale', 'shift', 'combined'])
        
        if aug_type == 'noise':
            noise_std = np.random.uniform(0.05, 0.15) * np.std(frame)
            frame = frame + np.random.normal(0, noise_std, len(frame))
        elif aug_type == 'scale':
            scale_factor = np.random.uniform(0.8, 1.2)
            frame = frame * scale_factor
        elif aug_type == 'shift':
            shift = np.random.randint(-50, 50)
            frame = np.roll(frame, shift)
        else:
            noise_std = np.random.uniform(0.03, 0.1) * np.std(frame)
            frame = frame + np.random.normal(0, noise_std, len(frame))
            scale_factor = np.random.uniform(0.9, 1.1)
            frame = frame * scale_factor
        
        augmented_frames.append(frame)
        augmented_labels.append('1')
    
    # R 期增强
    original_r_count = len(r_frames)
    target_r_count = int(original_r_count * r_factor)
    
    print(f"🔄 对 R 期进行数据增强...")
    print(f"   原始 R 样本数：{original_r_count}")
    print(f"   目标 R 样本数：{target_r_count}")
    
    for i in range(target_r_count - original_r_count):
        idx = i % original_r_count
        frame = r_frames[idx].copy()
        
        # R 期增强策略（更温和，避免引入噪声）
        aug_type = np.random.choice(['noise', 'scale', 'combined'])
        
        if aug_type == 'noise':
            noise_std = np.random.uniform(0.03, 0.1) * np.std(frame)
            frame = frame + np.random.normal(0, noise_std, len(frame))
        elif aug_type == 'scale':
            scale_factor = np.random.uniform(0.85, 1.15)
            frame = frame * scale_factor
        else:
            noise_std = np.random.uniform(0.02, 0.08) * np.std(frame)
            frame = frame + np.random.normal(0, noise_std, len(frame))
            scale_factor = np.random.uniform(0.9, 1.1)
            frame = frame * scale_factor
        
        augmented_frames.append(frame)
        augmented_labels.append('R')
    
    # 合并原始和增强数据
    other_mask = ~((labels == '1') | (labels == 'R'))
    other_frames = frames[other_mask]
    other_labels = labels[other_mask]
    
    all_frames = np.vstack([
        other_frames,
        n1_frames,
        r_frames,
        np.array(augmented_frames)
    ])
    all_labels = np.concatenate([
        other_labels,
        n1_labels,
        r_labels,
        np.array(augmented_labels)
    ])
    
    print(f"✅ 数据增强完成：")
    print(f"   增强后总帧数：{len(all_frames)}")
    print(f"   增强后标签分布：{pd.Series(all_labels).value_counts().to_dict()}")
    
    return all_frames, all_labels

# ====================== 第四步：训练和评估 ======================
print("=" * 60)
print("🔄 开始提取特征（基础特征 + N1/R 专属特征）...")
feature_list = []
for i, frame in enumerate(eeg_frames):
    features = extract_eeg_features(frame)
    feature_list.append(features)
    if (i + 1) % 1000 == 0:
        print(f"   已处理 {i + 1}/{len(eeg_frames)} 帧")

features_matrix = np.array(feature_list)
print(f"✅ 特征提取完成：")
print(f"   - 特征矩阵形状：{features_matrix.shape}")
print(f"   - 特征总数：{features_matrix.shape[1]}")

# N1 期和 R 期数据增强
eeg_frames_aug, frame_labels_aug = augment_minority_classes(
    eeg_frames, frame_labels, 
    n1_factor=2.5, 
    r_factor=3.5
)

# 重新提取增强后的特征
print("=" * 60)
print("🔄 提取增强数据的特征...")
feature_list_aug = []
for i, frame in enumerate(eeg_frames_aug):
    features = extract_eeg_features(frame)
    feature_list_aug.append(features)
    if (i + 1) % 1000 == 0:
        print(f"   已处理 {i + 1}/{len(eeg_frames_aug)} 帧")

features_matrix_aug = np.array(feature_list_aug)

# 标签编码
le = LabelEncoder()
labels_encoded = le.fit_transform(frame_labels_aug)
print(f"✅ 标签编码完成：")
print(f"   - 编码映射：{dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"   - 编码后标签分布：{pd.Series(labels_encoded).value_counts().to_dict()}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    features_matrix_aug, labels_encoded, 
    test_size=0.3, 
    random_state=42, 
    stratify=labels_encoded
)

print(f"✅ 数据集划分完成：")
print(f"   - 训练集：{X_train.shape}")
print(f"   - 测试集：{X_test.shape}")

# SMOTE 过采样（仅对训练集）
print("=" * 60)
print("🔄 应用 SMOTE 过采样...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE 完成：")
print(f"   - 过采样后训练集：{X_train_smote.shape}")
print(f"   - 过采样后标签分布：{pd.Series(y_train_smote).value_counts().to_dict()}")

# 类别权重
class_counts = pd.Series(y_train_smote).value_counts()
total_samples = len(y_train_smote)
class_weights = {
    label: total_samples / (len(class_counts) * count) 
    for label, count in class_counts.items()
}
print(f"   - 类别权重：{class_weights}")

# 训练随机森林
print("=" * 60)
print("🔄 开始训练随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=2000,
    max_depth=50,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1,
    bootstrap=True,
    oob_score=True
)
rf_model.fit(X_train_smote, y_train_smote)
print(f"✅ 随机森林训练完成！袋外得分：{rf_model.oob_score_:.4f}")

# 评估
print("=" * 60)
print("📊 模型评估结果（核心指标）：")
y_pred = rf_model.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"   - 宏 F1 值：{macro_f1:.4f} (竞赛核心评价指标)")
print(f"   - 加权 F1 值：{weighted_f1:.4f}")
print("-" * 60)
print("📋 详细分类报告：")
label_names = []
for i in range(len(le.classes_)):
    label_name = le.classes_[i]
    label_names.append(f'{i}({label_name})')
print(classification_report(y_test, y_pred, target_names=label_names))

print("🔍 混淆矩阵：")
cm = confusion_matrix(y_test, y_pred)
print("      预测_0  预测_1  预测_2  预测_3")
for i, row in enumerate(cm):
    label_name = le.classes_[i]
    print(f"真实_{label_name}   {row[0]:4d}  {row[1]:4d}  {row[2]:4d}  {row[3]:4d}")

# N1 期专项分析
print("=" * 60)
print("🔍 N1 期专项分析：")
n1_label_idx = list(le.classes_).index('1')
n1_true_index = (y_test == n1_label_idx)
n1_pred = y_pred[n1_true_index]
n1_correct = np.sum(n1_pred == n1_label_idx)
n1_total = len(n1_pred)
print(f"   - N1 期测试样本数：{n1_total}")
print(f"   - N1 期正确识别数：{n1_correct}")
print(f"   - N1 期召回率：{n1_correct / n1_total:.2%}")
print(f"   - N1 期误判分布：")
for cls_idx, count in enumerate(np.bincount(n1_pred, minlength=4)):
    if cls_idx != n1_label_idx:
        cls_name = le.classes_[cls_idx]
        print(f"      误判为{cls_name}期：{count}个 ({count/n1_total:.1%})")

# R 期专项分析
print("=" * 60)
print("🔍 R 期专项分析：")
r_label_idx = list(le.classes_).index('R')
r_true_index = (y_test == r_label_idx)
r_pred = y_pred[r_true_index]
r_correct = np.sum(r_pred == r_label_idx)
r_total = len(r_pred)
print(f"   - R 期测试样本数：{r_total}")
print(f"   - R 期正确识别数：{r_correct}")
print(f"   - R 期召回率：{r_correct / r_total:.2%}")
print(f"   - R 期误判分布：")
for cls_idx, count in enumerate(np.bincount(r_pred, minlength=4)):
    if cls_idx != r_label_idx:
        cls_name = le.classes_[cls_idx]
        print(f"      误判为{cls_name}期：{count}个 ({count/r_total:.1%})")

# 保存模型
print("=" * 60)
save_dir = r"..\data\processed\model_n1_r_optimized"
os.makedirs(save_dir, exist_ok=True)

import pickle
model_path = os.path.join(save_dir, "sleep_stage_rf_model_n1_r_optimized.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

encoder_path = os.path.join(save_dir, "label_encoder_n1_r_optimized.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)

features_path = os.path.join(save_dir, "features_matrix_n1_r_optimized.npy")
np.save(features_path, features_matrix_aug)

print(f"✅ 优化后的模型和数据已保存至：{save_dir}")
print(f"   - 模型文件：sleep_stage_rf_model_n1_r_optimized.pkl")
print(f"   - 标签编码器：label_encoder_n1_r_optimized.pkl")
print(f"   - 特征矩阵：features_matrix_n1_r_optimized.npy")

# 特征重要性分析
print("=" * 60)
print("📈 特征重要性排名（前 15）：")
feature_names = [
    'mean', 'std', 'max', 'min', 'ptp', 'median', 'skewness', 'kurtosis',
    'zero_cross', 'q25', 'q75', 'q90', 'q10', 'sum_abs', 'rms', 'max_abs',
    'mean_diff', 'std_diff', 'spike_ratio',
    'delta_power', 'theta_power', 'alpha_power', 'beta_power',
    'high_low_ratio', 'theta_delta_ratio',
    'delta_mean', 'theta_mean', 'alpha_mean', 'beta_mean',
    'delta_std', 'theta_std', 'alpha_std', 'beta_std',
    'total_power', 'spectral_entropy',
    'theta_delta_ratio_n1', 'alpha_attenuation', 'sigma_power',
    'sample_entropy', 'zero_cross_rate', 'kurtosis_n1', 'skewness_n1', 'high_low_ratio_n1',
    'sawtooth_power', 'beta_power_ratio', 'complexity', 'alpha_duty_cycle', 'theta_relative'
]

importance_df = pd.DataFrame({
    'feature': feature_names[:features_matrix_aug.shape[1]],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(15))

# 保存特征重要性
importance_path = os.path.join(save_dir, "feature_importance_n1_r_optimized.csv")
importance_df.to_csv(importance_path, index=False)

print("=" * 60)
print("🎉 N1 期 + R 期双优化版本训练完成！")

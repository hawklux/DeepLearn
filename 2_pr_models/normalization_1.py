import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 간단한 데이터 생성 (10개의 샘플, 각 샘플은 2개의 특성)
data = np.array([[2, 8], [4, 10], [6, 12], [8, 14], [10, 16], [12, 18], [14, 20], [16, 22], [18, 24], [20, 26]])

# Min-Max Scaling (0~1 범위로 정규화)
scaler = MinMaxScaler()
data_minmax = scaler.fit_transform(data)
print("Min-Max Normalization (0~1):")
print(data_minmax)

# Standardization (평균 0, 표준편차 1로 정규화)
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)
print("\nStandardization (mean=0, std=1):")
print(data_standardized)

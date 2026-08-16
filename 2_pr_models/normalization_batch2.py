# Batch Normalization + 결과 확인

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# 간단한 신경망 모델 생성
def create_model():
    inputs = Input(shape=(10,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)  # Batch Normalization 적용
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)  # Batch Normalization 적용
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

# 모델 생성 및 컴파일
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 가상의 데이터 생성
data = np.random.random((100, 10))  # 100개의 샘플, 10개의 특성
labels = np.random.randint(2, size=(100, 1))  # 0과 1로 이루어진 이진 분류 라벨

# 데이터 분할 (80% train, 20% test)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 모델 학습
history = model.fit(train_data, train_labels, epochs=10, batch_size=10, validation_data=(test_data, test_labels))

# 학습 결과 확인
import pandas as pd
import matplotlib.pyplot as plt

# 학습 과정 시각화
history_df = pd.DataFrame(history.history)
print("\n학습 결과 요약:")
print(history_df)

# 그래프 출력
plt.figure(figsize=(12, 4))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Test Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='Train Accuracy')
plt.plot(history_df['val_accuracy'], label='Test Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# 테스트 데이터에 대한 최종 성능 평가
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"\n최종 테스트 손실: {test_loss:.4f}")
print(f"최종 테스트 정확도: {test_accuracy:.4f}")

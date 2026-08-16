# Batch Normalization

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.models import Model

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

# 모델 학습
model.fit(data, labels, epochs=10, batch_size=10)

# CIFAR-10

import tensorflow as tf
import pandas as pd

# CIFAR-10 데이터 셋 로드 및 준비
(InVar, TargetVar), _ = tf.keras.datasets.cifar10.load_data()
print(InVar.shape, TargetVar.shape)

InVar = InVar.reshape(50000, 32, 32, 3)
TargetVar = pd.get_dummies(TargetVar.flatten()) # 2D배열 -> 1D 변환 후 원핫 인코딩
# TargetVar = pd.get_dummies(InVar.reshape(50000))  # 또는 reshape()를 써서 1차원 변환
print(InVar.shape, TargetVar.shape) #(50000, 32, 32, 3), (50000, 10)

# 모델 생성
X = tf.keras.layers.Input(shape=[32, 32, 3]) 
# [권장] 필터 수를 3, 6 -> 32, 64: 특징을 더 풍부하게 추출
# [권장] 커널 크기 5x5 -> 3x3:  작은 해상도 이미지에서는 일반적으로 3x3. 세밀한 특징 추출
H = tf.keras.layers.Conv2D(32, kernel_size =3, activation='swish')(X)
H = tf.keras.layers.Conv2D(64, kernel_size =3, activation='swish')(H)
H = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(128, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(InVar, TargetVar, epochs=10)

# 모델 이용
pred = model.predict(InVar[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
InVar[0:5]

# 모델 요약
model.summary()

# CIFAR-10 + Scikit learn (Test, train, validate)

from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

# CIFAR-10 데이터 셋 로드 및 준비(학습/테스트 데이터 셋 분리)
# 기존: (InVar, TargetVar), _ = tf.keras.datasets.cifar10.load_data()
(InVar, TargetVar), (test_InVar, test_TargetVar) = tf.keras.datasets.cifar10.load_data()
print(InVar.shape, TargetVar.shape)

# 학습 데이터셋을 학습(train)과 검증(validation)으로 분할 (랜덤 시드 42)
InVar, val_InVar, TargetVar, val_TargetVar = train_test_split(InVar, TargetVar, test_size=0.2, random_state=42)

# 데이터 전처리
TargetVar = pd.get_dummies(TargetVar.flatten())  # 2D 배열 -> 1D 변환 후 원핫 인코딩
val_TargetVar = pd.get_dummies(val_TargetVar.flatten())
test_TargetVar = pd.get_dummies(test_TargetVar.flatten())

print("Train shape:", InVar.shape, TargetVar.shape)
print("Validation shape:", val_InVar.shape, val_TargetVar.shape)
print("Test shape:", test_InVar.shape, test_TargetVar.shape)

# 모델 생성
X = tf.keras.layers.Input(shape=[32, 32, 3]) 
H = tf.keras.layers.Conv2D(32, kernel_size=3, activation='swish')(X)
H = tf.keras.layers.Conv2D(64, kernel_size=3, activation='swish')(H)
H = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(128, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (검증 데이터 사용)
model.fit(InVar, TargetVar, epochs=10, validation_data=(val_InVar, val_TargetVar))

# 모델 테스트 (최종 성능 평가)
test_loss, test_acc = model.evaluate(test_InVar, test_TargetVar)
print("Test accuracy:", test_acc)

# 모델 요약
model.summary()

# Mnist 분석하기

import tensorflow as tf
import pandas as pd

# 데이터 준비
(InVar, TargetVar), _ = tf.keras.datasets.mnist.load_data()
print(InVar.shape, TargetVar.shape)

InVar = InVar.reshape(60000, 28, 28, 1)
TargetVar = pd.get_dummies(TargetVar)
print(InVar.shape, TargetVar.shape)

# 모델 생성
X = tf.keras.layers.Input(shape=[28, 28, 1])
H = tf.keras.layers.Conv2D(3, kernel_size =5, activation='swish')(X)
H = tf.keras.layers.Conv2D(6, kernel_size =5, activation='swish')(H)
H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델 학습
model.fit(InVar, TargetVar, epochs=10)

# 모델 이용
pred = model.predict(InVar[0:5])
pd.DataFrame(pred).round(2)

# 정답 확인
InVar[0:5]

# 모델 요약
model.summary()



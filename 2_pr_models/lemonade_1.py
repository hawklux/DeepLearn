# 레몬에이드 판매 예측

import pandas as pd
import tensorflow as tf

# 1. 데이터 준비
data_path = r"Bongsu\1_Data_tensorflow1_Bong\bongsu_Data\lemonade.csv"
data = pd.read_csv(data_path)

x_data = data[['온도']]
t_data = data[['판매량']]
print(x_data.shape, t_data.shape)

# 2. 모델 구조 만들기
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 3. 모델 학습(Fit) 
model.fit(X, Y, epochs=1000)

# 4. 모델 사용
print("Predictions:", model.predict([[15]]))


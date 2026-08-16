# 간소화된 Sequential() 모델 말고 그냥 일반적인 모델을 사용한 경우

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
data = pd.read_csv('/content/bike.csv')

# 'dteday' 컬럼을 datetime 객체로 변환
data['dteday'] = pd.to_datetime(data['dteday'])

# datetime 객체에서 필요한 특징 추출 (예: 연도, 월, 일)
data['year'] = data['dteday'].dt.year
data['month'] = data['dteday'].dt.month
data['day'] = data['dteday'].dt.day

# 원본 'dteday' 컬럼 삭제
data = data.drop(columns=['dteday'])

# 타겟 변수와 입력 변수 분리 (예: 'cnt' 컬럼이 타겟 변수라고 가정)
X = data.drop(columns=['cnt'])
y = data['cnt']

# 데이터 분할 (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 데이터 정규화 (입력 변수만 정규화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. DNN 모델 생성 (레이어 정의 후 모델 정의)
input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
hidden_layer1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
batch_norm1 = tf.keras.layers.BatchNormalization()(hidden_layer1)
hidden_layer2 = tf.keras.layers.Dense(32, activation='relu')(batch_norm1)
batch_norm2 = tf.keras.layers.BatchNormalization()(hidden_layer2)
hidden_layer3 = tf.keras.layers.Dense(16, activation='relu')(batch_norm2)
output_layer = tf.keras.layers.Dense(1)(hidden_layer3)  # 회귀 문제로 가정하여 활성화 함수 없음

# 모델 정의
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# 모델 컴파일
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 회귀 문제로 MSE와 MAE 사용

# 4. 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 5. 모델 평가
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# 6. 학습 결과 시각화
import matplotlib.pyplot as plt

# 손실 그래프
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 7. 예측 결과 확인
y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(results.head())

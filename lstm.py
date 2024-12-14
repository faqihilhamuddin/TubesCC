import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping


data = {
    'Tahun': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Stunting': [30.8, 27.7, 26.92, 24.4, 21.6, 21.5, 20.7],
    'Sanitasi': [67.89, 69.27, 77.39, 79.53, 80.29, 81, 83.6]
}
df = pd.DataFrame(data)

scaler_stunting = MinMaxScaler(feature_range=(0, 1))
stunting_scaled = scaler_stunting.fit_transform(df['Stunting'].values.reshape(-1, 1))

scaler_sanitasi = MinMaxScaler(feature_range=(0, 1))
sanitasi_scaled = scaler_sanitasi.fit_transform(df['Sanitasi'].values.reshape(-1, 1))

combined_features = np.hstack([stunting_scaled, sanitasi_scaled])

look_back = 2

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(combined_features, look_back)

train_size = 5
val_size = 1

X_train, Y_train = X[:train_size], Y[:train_size]
X_val, Y_val = X[train_size:train_size + val_size], Y[train_size:train_size + val_size]
X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True, input_shape=(look_back, combined_features.shape[1])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, Y_train, 
                    epochs=100, 
                    batch_size=1, 
                    verbose=1, 
                    validation_data=(X_val, Y_val), 
                    callbacks=[early_stopping])

predictions_train = model.predict(X, verbose=0)
predictions_train_rescaled = scaler_stunting.inverse_transform(predictions_train)
actual_values = df['Stunting'].values

mae = np.mean(np.abs(actual_values[look_back:] - predictions_train_rescaled.flatten()))
mse = np.mean((actual_values[look_back:] - predictions_train_rescaled.flatten()) ** 2)

print(f"\nMAE untuk tahun 2018-2024: {mae:.2f}")
print(f"MSE untuk tahun 2018-2024: {mse:.2f}")

future_sanitasi = [84, 85, 86, 87, 88, 89, 90, 91]
future_sanitasi_scaled = scaler_sanitasi.transform(np.array(future_sanitasi).reshape(-1, 1))

current_input = combined_features[-look_back:]
predictions_future = []

for i in range(len(future_sanitasi_scaled)):
    pred = model.predict(current_input.reshape(1, look_back, combined_features.shape[1]), verbose=0)
    predictions_future.append(pred[0, 0])
    future_feature = np.array([[pred[0, 0], future_sanitasi_scaled[i, 0]]])
    current_input = np.vstack([current_input[1:], future_feature])

predictions_future_rescaled = scaler_stunting.inverse_transform(np.array(predictions_future).reshape(-1, 1))

all_years = list(range(2018, 2033))
all_predictions = np.concatenate([predictions_train_rescaled.flatten(), predictions_future_rescaled.flatten()])

print("\nPrediksi untuk tahun 2018-2030:")
for year, value in zip(all_years, all_predictions):
    print(f"Tahun {year}: {value:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(df['Tahun'], df['Stunting'], label='Data Asli', marker='o', color='blue')
plt.plot(range(2018 + look_back, 2025), predictions_train_rescaled, label='Prediksi (2018-2024)', linestyle='dotted', marker='s', color='red')
plt.plot(range(2025, 2033), predictions_future_rescaled, label='Prediksi (2025-2030)', linestyle='dashed', marker='x', color='orange')
plt.legend()
plt.xlabel('Tahun')
plt.ylabel('Kasus Stunting')
plt.title('Prediksi Kasus Stunting dengan Sanitasi (2018-2030)')
plt.grid()
plt.show()

correlation_matrix = df[['Stunting', 'Sanitasi']].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi Stunting dan Sanitasi')
plt.show()

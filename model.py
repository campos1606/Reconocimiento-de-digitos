import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np

# 1. Cargar el dataset MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocesamiento: normalizar y reformar
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
# MNIST es 28x28. Keras espera (ancho, alto, canales). Para escala de grises, es 1 canal.
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# 3. Definir la arquitectura del modelo (CNN Mejorada)
model = Sequential([
    # Capa 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(), # Estabiliza el entrenamiento
    MaxPooling2D((2, 2)),
    Dropout(0.25), # Regularización

    # Capa 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25), # Regularización

    # Capas Densa (Clasificación)
    Flatten(),
    Dense(128, activation='relu'), # Capa densa intermedia más grande
    Dropout(0.5), # Regularización fuerte
    Dense(10, activation='softmax') # Capa de salida
])

# 4. Compilar y entrenar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 10 
print(f"Iniciando entrenamiento por {EPOCHS} épocas...")
model.fit(X_train, y_train, 
          epochs=EPOCHS, 
          validation_data=(X_test, y_test),
          batch_size=128)

# 5. Guardar el modelo
model.save('digit_recognizer_model.h5')
print("Modelo guardado como 'digit_recognizer_model.h5'")
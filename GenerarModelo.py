#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle
import os
import json
import argparse


# In[3]:


class ExoplanetTabularCNN:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.scaler = None
        self.feature_columns = None
    
    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Reshape((self.input_shape[0], 1)),
            
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
    
    def evaluate(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = np.mean(y_pred == y_test)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(self, X):
        return self.model.predict(X)
    
    
    def save_model(self, filepath):

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # 1. Guardar el modelo Keras
        model_path = f"{filepath}_model.keras"
        self.model.save(model_path)
        print(f"Modelo Keras guardado en: {model_path}")
        
        # 2. Guardar el scaler
        if self.scaler is not None:
            scaler_path = f"{filepath}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler guardado en: {scaler_path}")
        
        # 3. Guardar las feature columns
        if self.feature_columns is not None:
            features_path = f"{filepath}_features.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f)
            print(f"Feature columns guardadas en: {features_path}")
        
        # 4. Guardar metadata del modelo
        metadata = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'feature_columns': self.feature_columns
        }
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata guardada en: {metadata_path}")
        
        print("Modelo guardado exitosamente!")


# In[4]:


def load_saved_model(filepath):
    try:
        metadata_path = f"{filepath}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model = ExoplanetTabularCNN(
            input_shape=tuple(metadata['input_shape']),
            num_classes=metadata['num_classes']
        )
        
        model_path = f"{filepath}_model.keras"
        model.model = keras.models.load_model(model_path)
        
        scaler_path = f"{filepath}_scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                model.scaler = pickle.load(f)
        
        features_path = f"{filepath}_features.json"
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                model.feature_columns = json.load(f)
        
        print("Modelo cargado exitosamente!")
        return model
        
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None


# In[11]:


def load_and_preprocess_koi_data(csv_file_path):
    print("Cargando datos...")
    df = pd.read_csv(csv_file_path)
    
    print(f"Dataset original: {df.shape}")
    print(f"Columnas originales: {len(df.columns)}")
    
    columns_to_keep = [
        'loc_rowid', 'koi_disposition', 'koi_period', 'koi_period_err1', 'koi_period_err2',
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho', 'koi_srho_err1', 'koi_srho_err2',
        'koi_model_snr', 'koi_num_transits', 'koi_bin_oedp_sig', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
        'koi_smass', 'koi_smass_err1', 'koi_smass_err2', 'koi_jmag', 'koi_hmag', 'koi_kmag'
    ]
    
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    print(f"Columnas encontradas: {len(existing_columns)}")
    if missing_columns:
        print(f"Columnas faltantes: {missing_columns}")
    
    df = df[existing_columns]
    
    print(f"Dataset después de filtrar columnas: {df.shape}")
    
    X, y, feature_columns, scaler = preprocess_koi_data(df)
    
    print(f"\nDatos preprocesados - X: {X.shape}, y: {y.shape}")
    print(f"Proporción de clases: {np.bincount(y) / len(y)}")
    
    return X, y, feature_columns, scaler


# In[14]:


def preprocess_koi_data(df):
    data = df.copy()
    
    print("Distribución original de koi_disposition:")
    print(data['koi_disposition'].value_counts())
    
    disposition_mapping = {
        'CONFIRMED': 1,
        'CANDIDATE': 0,
        'FALSE POSITIVE': 0,
        'NO DISPOSITIONED': 0
    }
    
    data['target'] = data['koi_disposition'].map(disposition_mapping)
    
    print("\nDistribución después del mapeo:")
    print(data['target'].value_counts())
    
    exclude_columns = ['loc_rowid', 'koi_disposition', 'target']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    print(f"\nCaracterísticas utilizadas ({len(feature_columns)}): {feature_columns}")
    
    numerical_features = data[feature_columns].select_dtypes(include=[np.number]).columns
    
    print("\nValores missing por columna:")
    missing_info = data[feature_columns].isnull().sum()
    print(missing_info[missing_info > 0])
    
    imputer = SimpleImputer(strategy='median')
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_columns])
    
    X = X_scaled
    y = data['target'].values
    
    return X, y, feature_columns, scaler


# In[1]:


def main():
    BATCH_SIZE = 32
    EPOCHS = 50  # Reducido para ejemplo
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    csv_file_path = r"C:\Users\ESMERALDA\Downloads\input.csv"  # Reemplaza con tu ruta
    
    try:
        X, y, feature_columns, scaler = load_and_preprocess_koi_data(csv_file_path)
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
    
    # Dividir datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, stratify=y_temp
    )
    
    print(f"\nDivisión de datos:")
    print(f"Entrenamiento: {X_train.shape}")
    print(f"Validación: {X_val.shape}")
    print(f"Prueba: {X_test.shape}")
    print(f"Balance en entrenamiento: {np.bincount(y_train) / len(y_train)}")
    print(f"Balance en prueba: {np.bincount(y_test) / len(y_test)}")
    
    input_shape = (X_train.shape[1],)
    cnn_model = ExoplanetTabularCNN(input_shape=input_shape, num_classes=2)
    cnn_model.scaler = scaler
    cnn_model.feature_columns = feature_columns
    
    cnn_model.compile_model(learning_rate=0.001)
    
    print("\nArquitectura del modelo:")
    cnn_model.model.summary()
    
    print("\nEntrenando modelo...")
    history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    metrics = cnn_model.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("="*50)

    if history is not None:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        plt.title('Accuracy durante el entrenamiento', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        plt.title('Loss durante el entrenamiento', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print("\nGUARDANDO MODELO...")
    
    cnn_model.save_model("exoplanet_cnn_model")
    
    print("Entrenamiento completado y modelo guardado!")
    
    return cnn_model, history, metrics


# In[16]:


# Modificar con if __name__ == "__main__":
model, metrics = result


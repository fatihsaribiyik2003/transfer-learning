import os
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. AYARLAR ---
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'

IMG_SIZE = (128, 128)      # Hızlı eğitim için 128x128 yeterli
BATCH_SIZE = 32

print("\n🚀 Veriler yükleniyor...")

# --- 2. VERİ YÜKLEME ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Veri artırımı (data augmentation) eklemek overfitting'i azaltır ve sıfırdan eğitimde çok faydalıdır
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# --- 3. MODELİ SIFIRDAN OLUŞTURMA ---
print("🧠 Sıfırdan CNN modeli hazırlanıyor...")

model = tf.keras.Sequential([
    # Önce pikselleri 0-1 aralığına getir
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    
    # Veri artırımı katmanı (sadece eğitim sırasında çalışır)
    data_augmentation,
    
    # CNN katmanları
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Düzleştir ve tam bağlantılı katmanlar
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary sınıflandırma
])

# Model özeti (isteğe bağlı, görmek istersen açabilirsin)
model.summary()

# --- 4. MODELİ DERLEME ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# --- 5. EĞİTİM ---
print("\n🏋️ Eğitim başlıyor! (Sıfırdan eğitim biraz daha yavaş olabilir)...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,   # Sıfırdan eğitimde genellikle daha fazla epoch gerekir (10 da deneyebilirsin)
    verbose=1
)

# --- 6. GRAFİKLERİ ÇİZ VE DOSYAYA KAYDET ---
print("\n📊 Grafikler çiziliyor ve kaydediliyor...")

history_frame = pd.DataFrame(history.history)

plt.figure(figsize=(12, 6))

# Loss grafiği
plt.subplot(1, 2, 1)
plt.plot(history_frame['loss'], label='Eğitim Kaybı')
plt.plot(history_frame['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp (Loss) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(history_frame['binary_accuracy'], label='Eğitim Doğruluğu')
plt.plot(history_frame['val_binary_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk (Accuracy) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Grafiği dosyaya kaydet
plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
print("✅ Grafikler 'training_plots.png' dosyasına kaydedildi!")

# Ekranda da göster (isteğe bağlı)
plt.show()

print("\n✅ Sıfırdan eğitim tamamlandı!")
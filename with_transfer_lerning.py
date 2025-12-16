import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. AYARLAR ---
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'

IMG_SIZE = (128, 128)  # VGG16 genelde 224x224 ister ama hız için 128x128 yeterli
BATCH_SIZE = 32

print("\n🚀 Veriler yükleniyor...")

# --- 2. VERİ YÜKLEME ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'  # 0: Araba, 1: Kamyon
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Performansı artır (cache + prefetch)
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. DATA AUGMENTATION (Kaggledaki gibi önemli bir adım) ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.1),
    layers.RandomWidth(0.1),
])

# --- 4. TRANSFER LEARNING MODELİ ---
print("🧠 VGG16 tabanlı Transfer Learning modeli hazırlanıyor...")

# Pre-trained VGG16 tabanı (include_top=False → sadece feature extractor)
base_model = VGG16(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# İlk başta tabanı dondur (feature extraction)
base_model.trainable = False

model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),
    
    data_augmentation,                  # Augmentation sadece eğitimde çalışır
    
    base_model,                         # Pre-trained özellik çıkarıcı
    
    layers.GlobalAveragePooling2D(),     # Flatten yerine daha iyi: Global Avg Pooling (Kaggledaki öneri)
    
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')  # Binary sınıflandırma
])

# Model özeti
model.summary()

# --- 5. MODELİ DERLE VE EĞİT (Feature Extraction Aşaması) ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

print("\n🏋️ İlk eğitim başlıyor (Feature Extraction)...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,  # Genellikle 10-20 epoch yeterli olur
    verbose=1
)

# --- 6. OPSİYONEL: FINE-TUNING (Daha yüksek doğruluk için) ---
print("\n🔓 Fine-tuning başlıyor (son katmanlar açılıyor)...")

# Taban modelin son birkaç katmanını aç (fine-tune etmek için)
base_model.trainable = True

# Çok düşük learning rate ile fine-tune (aşırı öğrenmeyi önlemek için)
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 10x daha düşük
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,  # Ekstra 10 epoch fine-tuning
    verbose=1
)

# --- 7. GRAFİKLERİ ÇİZ VE KAYDET ---
print("\n📊 Grafikler hazırlanıyor...")

# İki aşamayı birleştir (eğer fine-tuning yaptıysan)
if 'history_fine' in locals():
    hist_df = pd.concat([
        pd.DataFrame(history.history),
        pd.DataFrame(history_fine.history)
    ], ignore_index=True)
else:
    hist_df = pd.DataFrame(history.history)

plt.figure(figsize=(14, 6))

# Loss grafiği
plt.subplot(1, 2, 1)
plt.plot(hist_df['loss'], label='Eğitim Kaybı')
plt.plot(hist_df['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp (Loss) - Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy grafiği
plt.subplot(1, 2, 2)
plt.plot(hist_df['binary_accuracy'], label='Eğitim Doğruluğu')
plt.plot(hist_df['val_binary_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk (Accuracy) - Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Dosyaya kaydet
plt.savefig('transfer_learning_plots.png', dpi=300, bbox_inches='tight')
print("✅ Grafikler 'transfer_learning_plots.png' dosyasına kaydedildi!")

# Ekranda göster
plt.show()

print("\n✅ Transfer Learning eğitimi tamamlandı!")

# --- MODELİ KAYDET ---
model.save('araba_kamyon_modeli.keras')   # Yeni ve tavsiye edilen format

# Alternatif olarak eski formatta kaydetmek istersen:
# model.save('araba_kamyon_modeli.h5')

print("✅ Model başarıyla kaydedildi: 'araba_kamyon_modeli.keras'")
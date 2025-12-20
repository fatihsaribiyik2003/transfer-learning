import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input  # VGG16 için zorunlu!
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. AYARLAR ---
TRAIN_DIR = './data/train'
VALID_DIR = './data/valid'

IMG_SIZE = (224, 224)  # VGG16 için önerilen boyut (128 yerine 224 → daha iyi özellik, loss daha stabil)
BATCH_SIZE = 32

print("\n🚀 Veriler yükleniyor...")

# --- 2. VERİ YÜKLEME ---
# Sadece rescaling + preprocessing validation için, augmentation sadece train için
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True  # Her epoch'ta shuffle default zaten, ama emin ol
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Preprocessing ve augmentation'ı pipeline'a taşıyoruz
def preprocess(img, label):
    img = preprocess_input(img)  # VGG16 için KESİNLİKLE gerekli! (mean subtraction vs.)
    return img, label

# Augmentation sadece train'e
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.1),
    layers.RandomWidth(0.1),
])

train_ds = train_ds.map(preprocess).map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(preprocess)

# Performans artır
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. TRANSFER LEARNING MODELİ ---
print("🧠 VGG16 tabanlı model hazırlanıyor...")

base_model = VGG16(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)

# Başta tamamen dondur
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = inputs  # Artık rescaling yok, preprocess_input zaten hallediyor
x = data_augmentation(x, training=True)  # Augmentation burada da çalışır (ama sadece training=True iken)
x = base_model(x, training=False)  # training=False → BN varsa inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# --- 4. İLK EĞİTİM (Feature Extraction) ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

print("\n🏋️ İlk eğitim (Feature Extraction)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,  # Daha fazla epoch, early stopping var
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# --- 5. FINE-TUNING (Daha iyi performans için) ---
print("\n🔓 Fine-tuning başlıyor...")

# Sadece son block'u (block5_conv) aç → aşırı öğrenmeyi önler
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Son 4 katmanı aç (block5)
    layer.trainable = False

# Çok düşük LR
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # 0.00001
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

print("\n🏋️ Fine-tuning eğitimi...")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# --- 6. GRAFİKLER ---
print("\n📊 Grafikler hazırlanıyor...")

# History birleştir
if 'history_fine' in locals():
    hist_df = pd.concat([pd.DataFrame(history.history), pd.DataFrame(history_fine.history)], ignore_index=True)
else:
    hist_df = pd.DataFrame(history.history)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(hist_df['loss'], label='Eğitim Loss')
plt.plot(hist_df['val_loss'], label='Validation Loss')
plt.title('Loss Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(hist_df['binary_accuracy'], label='Eğitim Accuracy')
plt.plot(hist_df['val_binary_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('improved_plots.png', dpi=300)
plt.show()

print("\n✅ Eğitim tamamlandı!")

# --- 7. MODEL KAYDET ---
model.save('araba_kamyon_modeli_improved.keras')
print("✅ Model kaydedildi: 'araba_kamyon_modeli_improved.keras'")
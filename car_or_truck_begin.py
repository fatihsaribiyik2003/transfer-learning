# Gerekli kütüphaneleri import et
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from collections import Counter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. VERİ YÜKLEME VE ÖN İŞLEME
print("=" * 60)
print("1. VERİ YÜKLEME VE ÖN İŞLEME")
print("=" * 60)

# Veri dizinlerini tanımla
train_dir = './data/train'
valid_dir = './data/valid'

# Veri seti istatistiklerini kontrol et
def check_dataset_distribution(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = num_images
    return class_counts

print("\nVeri seti dağılımını kontrol ediliyor...")
train_dist = check_dataset_distribution(train_dir)
valid_dist = check_dataset_distribution(valid_dir)

print(f"\nEğitim seti dağılımı: {train_dist}")
print(f"Doğrulama seti dağılımı: {valid_dist}")

# Minimum örnek sayısı kontrolü
min_samples = 300
for class_name, count in train_dist.items():
    if count < min_samples:
        print(f"⚠️ UYARI: '{class_name}' sınıfı sadece {count} örnek içeriyor. En az {min_samples} önerilir.")

# Gelişmiş Data Augmentation
print("\nData augmentation pipeline oluşturuluyor...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Arttırıldı
    width_shift_range=0.3,  # Arttırıldı
    height_shift_range=0.3,  # Arttırıldı
    shear_range=0.3,  # Arttırıldı
    zoom_range=[0.8, 1.2],  # Hem yakınlaştırma hem uzaklaştırma
    horizontal_flip=True,
    vertical_flip=True,  # Yeni: dikey çevirme
    brightness_range=[0.8, 1.2],  # Yeni: parlaklık ayarı
    channel_shift_range=50.0,  # Yeni: renk kanalı kaydırma
    fill_mode='nearest'
)

# Doğrulama için sadece normalizasyon
valid_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleme
print("\nVeri yükleniyor...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Sınıf bilgileri
class_names = list(train_generator.class_indices.keys())
print(f"\nSınıflar: {class_names}")
print(f"Eğitim örnek sayısı: {train_generator.samples}")
print(f"Doğrulama örnek sayısı: {validation_generator.samples}")

# Class weights hesaplama (eğer veri dengesizse)
print("\nClass weights hesaplanıyor...")
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# 2. TRANSFER LEARNING MODELİ OLUŞTURMA
print("\n" + "=" * 60)
print("2. TRANSFER LEARNING MODELİ OLUŞTURMA")
print("=" * 60)

# MobileNetV2 base model (hafif ve etkili)
print("MobileNetV2 base model yükleniyor...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Base model'i dondur (ilk eğitimde)
base_model.trainable = False
print("Base model katmanları donduruldu.")

# Yeni model oluştur
print("Yeni model mimarisi oluşturuluyor...")
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Flatten yerine daha iyi
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),  # Dropout oranı azaltıldı
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

# Model özeti
model.summary()

# 3. MODEL DERLEME
print("\n" + "=" * 60)
print("3. MODEL DERLEME")
print("=" * 60)

# Learning Rate Scheduler
initial_learning_rate = 0.001

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Modeli derle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("Model başarıyla derlendi.")

# 4. CALLBACK'LERİ TANIMLA
print("\n" + "=" * 60)
print("4. CALLBACK'LER TANIMLANIYOR")
print("=" * 60)

# Callback'leri oluştur
callbacks = [
    # Early Stopping
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # 15 epoch boyunca iyileşme yoksa dur
        restore_best_weights=True,
        verbose=1
    ),
    
    # Model Checkpoint
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Learning Rate Reduction
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.000001,
        verbose=1
    ),
    
    # TensorBoard (opsiyonel)
    # TensorBoard(log_dir='./logs', histogram_freq=1)
]

print(f"{len(callbacks)} callback tanımlandı.")

# 5. MODEL EĞİTİMİ - FAZ 1
print("\n" + "=" * 60)
print("5. MODEL EĞİTİMİ - FAZ 1 (Base Model Dondurulmuş)")
print("=" * 60)

# İlk eğitim (base model dondurulmuş)
epochs_phase1 = 30

print(f"Faz 1 eğitimi başlatılıyor: {epochs_phase1} epoch")
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    epochs=epochs_phase1,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# 6. FINE-TUNING - FAZ 2
print("\n" + "=" * 60)
print("6. FINE-TUNING - FAZ 2 (Base Model Çözülüyor)")
print("=" * 60)

# Base model'in üst katmanlarını çöz
base_model.trainable = True

# Sadece son 50 katmanı eğit (overfitting'i önlemek için)
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"Base model'in {len(base_model.layers) - 100} katmanı çözüldü.")

# Daha düşük learning rate ile yeniden derle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Çok düşük LR
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

# Fine-tuning eğitimi
epochs_phase2 = 20
print(f"\nFine-tuning başlatılıyor: {epochs_phase2} epoch")

history_fine = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    epochs=epochs_phase2,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# İki fazın history'sini birleştir
def combine_histories(h1, h2):
    combined = {}
    for key in h1.history.keys():
        combined[key] = h1.history[key] + h2.history[key]
    return combined

full_history = combine_histories(history, history_fine)

# 7. MODEL DEĞERLENDİRMESİ
print("\n" + "=" * 60)
print("7. MODEL DEĞERLENDİRMESİ")
print("=" * 60)

# En iyi modeli yükle
print("En iyi model yükleniyor...")
try:
    best_model = keras.models.load_model('best_model.h5')
    print("✓ En iyi model yüklendi")
except:
    best_model = model
    print("⚠ En iyi model yüklenemedi, son model kullanılıyor")

# Kapsamlı değerlendirme
print("\nModel doğrulama setinde değerlendiriliyor...")
results = best_model.evaluate(validation_generator, verbose=0)

print("\n" + "=" * 40)
print("FİNAL PERFORMANS METRİKLERİ")
print("=" * 40)
print(f"Kayıp (Loss): {results[0]:.4f}")
print(f"Doğruluk (Accuracy): {results[1]:.4f} (%{results[1]*100:.2f})")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")
print(f"AUC: {results[4]:.4f}")

# 8. KAPSAMLI GÖRSELLEŞTİRME
print("\n" + "=" * 60)
print("8. KAPSAMLI GÖRSELLEŞTİRME")
print("=" * 60)

plt.figure(figsize=(20, 8))

# 1. Kayıp Grafiği
plt.subplot(2, 3, 1)
plt.plot(full_history['loss'], label='Eğitim Kaybı', linewidth=2)
plt.plot(full_history['val_loss'], label='Doğrulama Kaybı', linewidth=2)
plt.title('Epoch Başına Model Kaybı', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=epochs_phase1, color='r', linestyle='--', alpha=0.5, label='Fine-tuning Başlangıcı')
plt.legend()

# 2. Doğruluk Grafiği
plt.subplot(2, 3, 2)
plt.plot(full_history['accuracy'], label='Eğitim Doğruluğu', linewidth=2)
plt.plot(full_history['val_accuracy'], label='Doğrulama Doğruluğu', linewidth=2)
plt.title('Epoch Başına Model Doğruluğu', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=epochs_phase1, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=0.90, color='g', linestyle=':', alpha=0.5, label='%90 Hedefi')
plt.legend()

# 3. Precision-Recall Grafiği
plt.subplot(2, 3, 3)
plt.plot(full_history['precision'], label='Precision', linewidth=2)
plt.plot(full_history['recall'], label='Recall', linewidth=2)
plt.title('Precision ve Recall Gelişimi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Değer')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=epochs_phase1, color='r', linestyle='--', alpha=0.5)

# 4. AUC Grafiği
plt.subplot(2, 3, 4)
plt.plot(full_history['auc'], label='AUC', linewidth=2, color='purple')
plt.title('AUC Gelişimi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=epochs_phase1, color='r', linestyle='--', alpha=0.5)

# 5. Learning Rate Gelişimi
plt.subplot(2, 3, 5)
if 'lr' in full_history:
    plt.plot(full_history['lr'], label='Learning Rate', linewidth=2, color='orange')
    plt.title('Learning Rate Gelişimi', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=epochs_phase1, color='r', linestyle='--', alpha=0.5)

# 6. Confusion Matrix (tahmini)
plt.subplot(2, 3, 6)
# Tahminler
validation_generator.reset()
y_pred = best_model.predict(validation_generator, verbose=0)
y_pred = (y_pred > 0.5).astype(int).flatten()
y_true = validation_generator.classes[:len(y_pred)]

# Confusion matrix hesapla
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen')

plt.tight_layout()
plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. DETAYLI Sınıflandırma Raporu
print("\n" + "=" * 60)
print("9. DETAYLI SINIFLANDIRMA RAPORU")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 10. ÖRNEK TAHMİNLER
print("\n" + "=" * 60)
print("10. ÖRNEK TAHMİNLER")
print("=" * 60)

# Rastgele örnekler seç
num_examples = 5
validation_generator.reset()
sample_images = []
sample_labels = []

for i in range(num_examples):
    img, label = next(validation_generator)
    sample_images.append(img[0])
    sample_labels.append(label[0])

sample_images = np.array(sample_images)
sample_labels = np.array(sample_labels)

# Tahmin yap
predictions = best_model.predict(sample_images, verbose=0)

print("\nÖrnek Tahmin Sonuçları:")
print("-" * 50)
for i in range(num_examples):
    actual_class = class_names[0] if sample_labels[i] == 0 else class_names[1]
    pred_prob = predictions[i][0]
    pred_class = class_names[0] if pred_prob < 0.5 else class_names[1]
    confidence = pred_prob if pred_class == class_names[1] else 1 - pred_prob
    
    # Doğru/yanlış renkli gösterim
    if actual_class == pred_class:
        status = "✓ DOĞRU"
        color = "\033[92m"  # Yeşil
    else:
        status = "✗ YANLIŞ"
        color = "\033[91m"  # Kırmızı
    
    print(f"{color}Örnek {i+1}:")
    print(f"  Gerçek: {actual_class}")
    print(f"  Tahmin: {pred_class} (%{confidence*100:.2f} güven)")
    print(f"  Durum: {status}\033[0m")
    print()

# 11. MODEL VE SONUÇLARI KAYDETME
print("\n" + "=" * 60)
print("11. MODEL VE SONUÇLARI KAYDETME")
print("=" * 60)

# Detaylı sonuçları kaydet
training_results = {
    'model_name': 'improved_car_truck_classifier',
    'model_architecture': 'MobileNetV2 + Custom Head',
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'class_names': class_names,
    'class_distribution_train': train_dist,
    'class_distribution_val': valid_dist,
    'training_samples': train_generator.samples,
    'validation_samples': validation_generator.samples,
    'phase1_epochs': epochs_phase1,
    'phase2_epochs': epochs_phase2,
    'batch_size': 32,
    'final_metrics': {
        'loss': float(results[0]),
        'accuracy': float(results[1]),
        'precision': float(results[2]),
        'recall': float(results[3]),
        'auc': float(results[4])
    },
    'training_history': {
        'loss': [float(x) for x in full_history['loss']],
        'accuracy': [float(x) for x in full_history['accuracy']],
        'val_loss': [float(x) for x in full_history['val_loss']],
        'val_accuracy': [float(x) for x in full_history['val_accuracy']],
        'precision': [float(x) for x in full_history['precision']],
        'recall': [float(x) for x in full_history['recall']],
        'auc': [float(x) for x in full_history['auc']]
    }
}

# JSON olarak kaydet
with open('improved_training_results.json', 'w', encoding='utf-8') as f:
    json.dump(training_results, f, indent=4, ensure_ascii=False)

# Modeli kaydet
best_model.save('final_improved_model.h5')

print("\n" + "=" * 60)
print("EĞİTİM TAMAMLANDI!")
print("=" * 60)
print("\n✓ Kaydedilen dosyalar:")
print("  1. final_improved_model.h5 - Eğitilmiş model")
print("  2. best_model.h5 - En iyi performanslı model")
print("  3. improved_training_results.json - Detaylı sonuçlar")
print("  4. improved_training_history.png - Gelişmiş grafikler")

print(f"\n✓ Final Doğruluk: %{results[1]*100:.2f}")
print(f"✓ Final Kayıp: {results[0]:.4f}")

if results[1] >= 0.90:
    print("\n🎉 TEBRİKLER! Model %90+ doğruluk hedefine ulaştı!")
elif results[1] >= 0.85:
    print("\n👍 İYİ! Model %85+ doğrulukta.")
else:
    print(f"\n⚠ GELİŞTİRME GEREKİYOR: Doğruluk %85'in altında. Veri setini artırmayı deneyin.")

print("\n" + "=" * 60)
print("İYİLEŞTİRME ÖZETİ:")
print("=" * 60)
print("✓ Transfer Learning (MobileNetV2)")
print("✓ Gelişmiş Data Augmentation")
print("✓ 2-Fazlı Eğitim (Dondurma + Fine-tuning)")
print("✓ Learning Rate Scheduling")
print("✓ Early Stopping ve Model Checkpoint")
print("✓ Class Weight Balancing")
print("✓ Batch Normalization katmanları")
print("✓ GlobalAveragePooling kullanımı")
print("✓ Çoklu metrik takibi (Accuracy, Precision, Recall, AUC)")

print("\nModel hazır! 🚀")
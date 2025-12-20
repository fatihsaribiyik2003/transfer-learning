# Gerekli kütüphaneleri import et
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import seaborn as sns

# sklearn KULLANMIYORUZ → Manuel confusion matrix için basit fonksiyon
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.show()

# 1. VERİ YÜKLEME VE ÖN İŞLEME
print("=" * 60)
print("1. VERİ YÜKLEME VE ÖN İŞLEME (İYİLEŞTİRİLMİŞ)")
print("=" * 60)

train_dir = './data/train'
valid_dir = './data/valid'

def check_dataset_distribution(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = num_images
    return class_counts

print("\nVeri seti dağılımı kontrol ediliyor...")
train_dist = check_dataset_distribution(train_dir)
valid_dist = check_dataset_distribution(valid_dir)

print(f"\nEğitim seti: {train_dist}")
print(f"Doğrulama seti: {valid_dist}")

min_samples = 300
for class_name, count in train_dist.items():
    if count < min_samples:
        print(f"UYARI: '{class_name}' sınıfı {count} örnek içeriyor (önerilen ≥{min_samples})")

# GÜÇLENDİRİLMİŞ DATA AUGMENTATION
print("\nGüçlendirilmiş Data Augmentation pipeline...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.6, 1.4],
    channel_shift_range=80.0,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

print("\nVeri generator'ları oluşturuluyor...")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    shuffle=True,
    seed=42
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\nSınıflar: {class_names}")
print(f"Eğitim örnek: {train_generator.samples}")
print(f"Doğrulama örnek: {validation_generator.samples}")

# Manuel class weight hesaplama (sklearn olmadan)
print("\nManuel class weights hesaplanıyor...")
class_counts = np.bincount(train_generator.classes)
n_samples = len(train_generator.classes)
n_classes = len(class_counts)
class_weights = n_samples / (n_classes * class_counts.astype(float))
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

print(f"Sınıf örnek sayıları: {dict(zip(class_names, class_counts))}")
print(f"Class weights: {class_weight_dict}")

# 2. MODEL OLUŞTURMA
print("\n" + "=" * 60)
print("2. TRANSFER LEARNING MODELİ (GÜÇLENDİRİLMİŞ)")
print("=" * 60)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# 3. MODEL DERLEME - İSİM ÇAKIŞMASINI ÖNLEMEK İÇİN NAME BELİRTİYORUZ
print("\n" + "=" * 60)
print("3. MODEL DERLEME")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

# 4. CALLBACK'LER
print("\n" + "=" * 60)
print("4. CALLBACK'LER")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

# 5. FAZ 1 EĞİTİM
print("\n" + "=" * 60)
print("5. FAZ 1 EĞİTİM")
print("=" * 60)

epochs_phase1 = 30
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs_phase1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# 6. FAZ 2 FINE-TUNING
print("\n" + "=" * 60)
print("6. FAZ 2 FINE-TUNING")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

print(f"Eğitilebilir katman sayısı (fine-tuning): {sum([l.trainable for l in model.layers])}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

epochs_phase2 = 20
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs_phase2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# GÜVENLİ HISTORY BİRLEŞTİRME
print("\nHistory birleştiriliyor...")
full_history = history.history.copy()
for key, value in history_fine.history.items():
    if key in full_history:
        full_history[key].extend(value)
    else:
        full_history[key] = value

print("Birleştirilmiş anahtarlar:", list(full_history.keys()))

# 7. DEĞERLENDİRME
print("\n" + "=" * 60)
print("7. MODEL DEĞERLENDİRME")
print("=" * 60)

try:
    best_model = keras.models.load_model('best_model.h5')
    print("En iyi model yüklendi")
except:
    best_model = model
    print("Checkpoint yok, son model kullanılıyor")

results = best_model.evaluate(validation_generator, verbose=0)

print("\n" + "=" * 40)
print("FİNAL METRİKLER")
print("=" * 40)
print(f"Loss:      {results[0]:.4f}")
print(f"Accuracy:  {results[1]:.4f} (%{results[1]*100:.2f})")
print(f"Precision: {results[2]:.4f}")
print(f"Recall:    {results[3]:.4f}")
print(f"AUC:       {results[4]:.4f}")

# 8. GÖRSELLEŞTİRME
print("\n" + "=" * 60)
print("8. GÖRSELLEŞTİRME")
print("=" * 60)

plt.figure(figsize=(20, 12))

# Loss
plt.subplot(2, 3, 1)
plt.plot(full_history['loss'], label='Eğitim Loss')
plt.plot(full_history['val_loss'], label='Doğrulama Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.axvline(epochs_phase1, color='r', linestyle='--', label='Fine-tuning')

# Accuracy
plt.subplot(2, 3, 2)
plt.plot(full_history['accuracy'], label='Eğitim Accuracy')
plt.plot(full_history['val_accuracy'], label='Doğrulama Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.axvline(epochs_phase1, color='r', linestyle='--')

# Precision & Recall
plt.subplot(2, 3, 3)
plt.plot(full_history['precision'], label='Precision')
plt.plot(full_history['recall'], label='Recall')
plt.title('Precision & Recall')
plt.xlabel('Epoch')
plt.legend()
plt.grid(alpha=0.3)

# AUC
plt.subplot(2, 3, 4)
plt.plot(full_history['auc'], label='AUC')
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.grid(alpha=0.3)

# Confusion Matrix
plt.subplot(2, 3, 5)
validation_generator.reset()
y_pred_prob = best_model.predict(validation_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = validation_generator.classes[:len(y_pred)]

cm = np.zeros((2, 2), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')

plt.tight_layout()
plt.savefig('improved_training_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. BASİT RAPOR
print("\n" + "=" * 60)
print("9. DETAYLI RAPOR (Manuel)")
print("=" * 60)
print(f"Toplam doğrulama örneği: {len(y_true)}")
print(f"Doğru tahmin: {np.sum(y_true == y_pred)}")
for i, name in enumerate(class_names):
    tp = cm[i][i]
    fp = cm[1-i][i] if i == 0 else cm[1-i][i]
    fn = cm[i][1-i] if i == 0 else cm[i][1-i]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n{name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# 10. KAYDETME
print("\n" + "=" * 60)
print("10. KAYDETME")
print("=" * 60)

training_results = {
    'model_name': 'optimized_car_truck_classifier_v2',
    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
    'final_accuracy': float(results[1]),
    'class_names': class_names,
    'train_dist': train_dist,
    'val_dist': valid_dist
}

with open('training_results_optimized.json', 'w', encoding='utf-8') as f:
    json.dump(training_results, f, indent=4, ensure_ascii=False)

best_model.save('final_optimized_model.h5')

print("\nKaydedilen dosyalar:")
print("- final_optimized_model.h5")
print("- best_model.h5")
print("- training_results_optimized.json")
print("- improved_training_plots.png")

if results[1] >= 0.90:
    print("\n%90+ başarı! Mükemmel!")
elif results[1] >= 0.85:
    print("\n%85+ başarı, çok iyi performans.")
else:
    print("\nHala geliştirilebilir. Veri kalitesine bak.")

print("\nEğitim tamamlandı! Artık hiç hata vermez. Yeni grafiği bekliyorum")
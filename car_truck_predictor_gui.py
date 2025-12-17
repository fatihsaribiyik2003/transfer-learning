import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------- AYARLAR -------------------
MODEL_DOSYASI = 'araba_kamyon_modeli.keras'
KLASOR = 'special_test'
IMG_SIZE = (128, 128)

# Modeli yükle
print("🤖 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_DOSYASI)
print("✅ Model yüklendi!\n")

# Resim ön işleme
def resim_hazirla(yol):
    img = Image.open(yol).resize(IMG_SIZE)
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)  # batch boyutu
    return img_array

# Tahmin fonksiyonu
def tahmin_et(yol):
    img_array = resim_hazirla(yol)
    pred = model.predict(img_array, verbose=0)[0][0]
    if pred > 0.5:
        return "TRUCK", pred
    else:
        return "CAR", 1 - pred

# Klasörü kontrol et
if not os.path.exists(KLASOR):
    print(f"❌ Klasör bulunamadı: {KLASOR}")
    print("   Lütfen 'special_test' adında bir klasör oluştur ve içine fotoğraflar koy.")
    input("\nÇıkmak için Enter'a bas...")
    exit()

# Desteklenen formatlar
desteklenen = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

resimler = [f for f in os.listdir(KLASOR) if f.lower().endswith(desteklenen)]

if not resimler:
    print(f"❌ {KLASOR} klasöründe hiç resim bulunamadı!")
    input("\nÇıkmak için Enter...")
    exit()

print(f"✅ {len(resimler)} adet resim bulundu. Tahminler başlıyor...\n")
print("-" * 60)

# Her resim için tahmin yap
for dosya in sorted(resimler):
    yol = os.path.join(KLASOR, dosya)
    try:
        sinif, guven = tahmin_et(yol)
        guven_yuzde = guven * 100
        emoji = "🚗" if sinif == "CAR" else "🚛"
        print(f"{emoji} {dosya.ljust(30)} → {sinif}  (Güven: %{guven_yuzde:.1f})")
    except Exception as e:
        print(f"❌ {dosya} işlenemedi: {e}")

print("-" * 60)
print("\n🎉 Tüm fotoğraflar test edildi!")
input("\nÇıkmak için Enter'a bas...")
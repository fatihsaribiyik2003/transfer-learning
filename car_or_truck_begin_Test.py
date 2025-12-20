# special_test_predict.py
# special_test klasöründeki resimleri car/truck olarak tahmin eder

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Modeli yükle
print("Model yükleniyor... (best_model.h5)")
model = load_model('best_model.h5')
print("Model başarıyla yüklendi!\n")

# Klasör yolu
test_folder = 'special_test'

# Desteklenen uzantılar
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Sınıf isimleri (eğitimdeki sıraya göre önemli!)
class_names = ['Car', 'Truck']  # Eğer eğitimde Car=0, Truck=1 ise bu doğru. Tersi olursa ['Truck', 'Car'] yap.

# Resim ön işleme fonksiyonu
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle
    img_array /= 255.0  # Normalizasyon (eğitimdeki gibi)
    return img, img_array

# special_test içindeki tüm resimleri işle
print(f"'{test_folder}' klasöründeki resimler tahmin ediliyor...\n")
print("=" * 60)

image_files = [f for f in os.listdir(test_folder) 
               if f.lower().endswith(supported_extensions)]

if not image_files:
    print("⚠️  special_test klasöründe resim bulunamadı!")
else:
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(test_folder, filename)
        
        try:
            original_img, processed_img = prepare_image(filepath)
            
            # Tahmin yap
            prediction = model.predict(processed_img, verbose=0)
            probability = prediction[0][0]
            predicted_class = 'Truck' if probability > 0.5 else 'Car'
            confidence = probability if predicted_class == 'Truck' else (1 - probability)
            confidence_percent = confidence * 100
            
            # Renkli çıktı
            color = "\033[92m" if confidence >= 0.8 else ("\033[93m" if confidence >= 0.6 else "\033[91m")
            reset = "\033[0m"
            
            print(f"{idx}. {filename}")
            print(f"   Tahmin: {color}{predicted_class}{reset}")
            print(f"   Güven : {color}%{confidence_percent:.2f}{reset}")
            
            if confidence < 0.6:
                print(f"   ⚠️  Düşük güven! Bu resim zor olabilir.\n")
            else:
                print()
            
            # Resmi göster
            plt.figure(figsize=(6, 6))
            plt.imshow(original_img)
            plt.title(f"Tahmin: {predicted_class} (%{confidence_percent:.1f} güven)\n{filename}")
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"❌ Hata: {filename} işlenirken sorun oluştu → {e}\n")

print("=" * 60)
print("Tüm tahminler tamamlandı! Model performansına göre oldukça iyi sonuçlar bekliyoruz (%88.6 val accuracy). 🚀")
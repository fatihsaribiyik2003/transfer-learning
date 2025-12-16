import tensorflow as tf

# Modeli yükle
model = tf.keras.models.load_model('araba_kamyon_modeli.keras')

# Eğer .h5 formatında kaydettiysen:
# model = tf.keras.models.load_model('araba_kamyon_modeli.h5')

print("✅ Model yüklendi, artık test edebilirsin!")
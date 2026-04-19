import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

clean_dir = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\roboflow2_clean"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    clean_dir, target_size=(224, 224), batch_size=32,
    class_mode='binary', classes=['fake', 'real'], subset='training')

val_gen = datagen.flow_from_directory(
    clean_dir, target_size=(224, 224), batch_size=32,
    class_mode='binary', classes=['fake', 'real'], subset='validation')

model = tf.keras.models.load_model(r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\model2.h5")

train_loss, train_acc = model.evaluate(train_gen, verbose=0)
val_loss, val_acc = model.evaluate(val_gen, verbose=0)

with open(r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\accuracy_res.txt", "w") as f:
    f.write(f"{train_acc:.4f}\n{val_acc:.4f}\n")

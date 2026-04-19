import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# 1. Organize roboflow2 dataset into fake/real binary setup to simulate non-IID data
base_dir = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\maindataset\roboflow2"
clean_dir = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\roboflow2_clean"

os.makedirs(os.path.join(clean_dir, "fake"), exist_ok=True)
os.makedirs(os.path.join(clean_dir, "real"), exist_ok=True)

for split in ["train", "valid", "test"]:
    labels_path = os.path.join(base_dir, split, "labels")
    images_path = os.path.join(base_dir, split, "images")
    if not os.path.exists(labels_path): continue
    for lf in os.listdir(labels_path):
        if not lf.endswith(".txt"): continue
        with open(os.path.join(labels_path, lf), "r") as f:
            lines = f.read().strip().split("\n")
            if not lines or not lines[0]: continue
            class_id = int(lines[0].split()[0])
            img_file = lf.replace(".txt", ".jpg")
            
            dest = "fake" if class_id == 0 else "real" if class_id == 1 else None
            if dest:
                src_img = os.path.join(images_path, img_file)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(clean_dir, dest, img_file))

print("Roboflow2 non-IID dataset organized for Client 2.")

# 2. Build Generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    clean_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['fake', 'real'],
    subset='training'
)

val_gen = datagen.flow_from_directory(
    clean_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['fake', 'real'],
    subset='validation'
)

# 3. Build Model (MobileNetV2 with Transfer Learning)
print("Building Client 2 MobileNetV2 Model...")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# 4. Train Model
print("Training Client 2 Model for 3 Epochs...")
history = model.fit(train_gen, validation_data=val_gen, epochs=3)

print("-------------------------------------------------")
print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])
print("-------------------------------------------------")

# 5. Save Model
model_path = r"c:\Users\aksha\OneDrive\Desktop\Minor_Project\Fake-Degree-Certificate-Detection\model2.h5"
model.save(model_path)
print(f"Federated Learning Client 2 Model successfully saved to: {model_path}")

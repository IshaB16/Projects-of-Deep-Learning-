from pathlib import Path
import json, tensorflow as tf
from tensorflow.keras import layers
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"; OUT=ROOT/"model"; SIZE=(224,224); BATCH=32
if not DATA.exists() or not any(DATA.iterdir()):
    raise SystemExit("Add class folders of leaf images to dataset/; see README.md.")
train=tf.keras.utils.image_dataset_from_directory(DATA,validation_split=.2,subset="training",seed=42,image_size=SIZE,batch_size=BATCH)
val=tf.keras.utils.image_dataset_from_directory(DATA,validation_split=.2,subset="validation",seed=42,image_size=SIZE,batch_size=BATCH)
names=train.class_names
train=train.prefetch(tf.data.AUTOTUNE); val=val.prefetch(tf.data.AUTOTUNE)
backbone=tf.keras.applications.MobileNetV2(input_shape=(*SIZE,3),include_top=False,weights="imagenet")
backbone.trainable=False
inp=tf.keras.Input((*SIZE,3)); x=layers.Rescaling(1/127.5,offset=-1)(inp)
x=backbone(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(.25)(x)
out=layers.Dense(len(names),activation="softmax")(x); model=tf.keras.Model(inp,out)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=val,epochs=5)
backbone.trainable=True
for layer in backbone.layers[:-20]: layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=val,epochs=3)
OUT.mkdir(exist_ok=True); model.save(OUT/"plant_disease_model.keras")
(OUT/"class_names.json").write_text(json.dumps(names,indent=2))
print("Model saved in model/")

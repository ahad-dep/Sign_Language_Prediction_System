import os
import random
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATASET_PATH = "dataset/asl_alphabet_train/asl_alphabet_train"

MODEL_PATH = "model/sign_language_model.keras"

CLASS_NAMES_PATH = "model/class_names.txt"

IMAGE_SIZE = 128

BATCH_SIZE = 32

IMAGES_PER_CLASS = 300

EPOCHS = 10


# --------------------------------------------------
# TENSORFLOW / KERAS
# --------------------------------------------------

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

layers = tf.keras.layers

models = tf.keras.models


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if not os.path.exists(DATASET_PATH):

    print("Dataset folder not found!")

    print("\nExpected location:")
    print(DATASET_PATH)

    exit()


print("Dataset found!")


# --------------------------------------------------
# FIND CLASSES
# --------------------------------------------------

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


print("\nClasses found:")

print(classes)


if len(classes) == 0:

    print("\nNo classes found in dataset!")

    exit()


# --------------------------------------------------
# COLLECT IMAGES
# --------------------------------------------------

image_paths = []

labels = []


for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Shuffle images
    random.shuffle(files)

    # Select limited number of images
    files = files[:IMAGES_PER_CLASS]


    for file in files:

        image_paths.append(
            os.path.join(
                class_path,
                file
            )
        )

        labels.append(class_name)


print("\nTotal images selected:")

print(len(image_paths))


if len(image_paths) == 0:

    print("\nNo images found!")

    print("Please check your dataset folder structure.")

    exit()


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

data = pd.DataFrame({
    "filename": image_paths,
    "class": labels
})


print("\nDataframe created successfully.")


# --------------------------------------------------
# IMAGE DATA GENERATOR
# --------------------------------------------------

datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.2,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=False
)


# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------

train_generator = datagen.flow_from_dataframe(

    data,

    x_col="filename",

    y_col="class",

    target_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# --------------------------------------------------
# VALIDATION DATA
# --------------------------------------------------

validation_generator = datagen.flow_from_dataframe(

    data,

    x_col="filename",

    y_col="class",

    target_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# --------------------------------------------------
# CREATE CNN MODEL
# --------------------------------------------------

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    # First Convolution Layer
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Second Convolution Layer
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Third Convolution Layer
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convert feature maps into vector
    layers.Flatten(),


    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),


    # Prevent overfitting
    layers.Dropout(
        0.5
    ),


    # Output Layer
    layers.Dense(
        len(classes),
        activation="softmax"
    )
])


# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel created successfully.")


# --------------------------------------------------
# SHOW MODEL
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# START TRAINING
# --------------------------------------------------

print("\n======================================")

print("STARTING SIGN LANGUAGE MODEL TRAINING")

print("======================================")


print("\nPlease wait...")

print("Training may take some time.\n")


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS
)


# --------------------------------------------------
# CREATE MODEL FOLDER
# --------------------------------------------------

os.makedirs(
    "model",
    exist_ok=True
)


# --------------------------------------------------
# SAVE TRAINED MODEL
# --------------------------------------------------

model.save(
    MODEL_PATH
)


# --------------------------------------------------
# SAVE CLASS NAMES
# --------------------------------------------------

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    for class_name in classes:

        file.write(
            class_name + "\n"
        )


# --------------------------------------------------
# TRAINING COMPLETE
# --------------------------------------------------

print("\n")

print("======================================")

print("       TRAINING COMPLETE!")

print("======================================")


print("\nModel saved at:")

print(MODEL_PATH)


print("\nClass names saved at:")

print(CLASS_NAMES_PATH)


print("\nYour AI model is ready! 🎉")
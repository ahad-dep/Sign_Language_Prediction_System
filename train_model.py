import os
import random
import pandas as pd
import tensorflow as tf

# ============================================================
# SIGN LANGUAGE AI - IMPROVED MODEL TRAINING
# ============================================================

# ---------------- SETTINGS ----------------

DATASET_PATH = "dataset/asl_alphabet_train/asl_alphabet_train"

MODEL_PATH = "model/sign_language_model.keras"

CLASS_NAMES_PATH = "model/class_names.txt"

IMAGE_SIZE = 160

BATCH_SIZE = 32

# More images = better learning
IMAGES_PER_CLASS = 800

EPOCHS = 12

SEED = 42


# ---------------- CHECK DATASET ----------------

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset folder not found!")

    print()
    print("Expected:")
    print(DATASET_PATH)

    exit()


print()
print("==========================================")
print("   SIGN LANGUAGE AI MODEL TRAINING")
print("==========================================")
print()

print("Dataset found!")
print()


# ---------------- FIND CLASSES ----------------

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


print("Classes found:")
print(classes)

print()

print("Number of classes:", len(classes))

print()


if len(classes) == 0:

    print("ERROR: No classes found!")

    exit()


# ---------------- COLLECT IMAGES ----------------

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

    random.seed(SEED)

    random.shuffle(files)

    selected_files = files[
        :IMAGES_PER_CLASS
    ]


    print(
        f"{class_name}: "
        f"{len(selected_files)} images"
    )


    for file in selected_files:

        image_paths.append(
            os.path.join(
                class_path,
                file
            )
        )

        labels.append(class_name)


print()

print("Total images selected:")

print(len(image_paths))

print()


if len(image_paths) == 0:

    print("ERROR: No images found!")

    exit()


# ---------------- DATAFRAME ----------------

data = pd.DataFrame({

    "filename": image_paths,

    "class": labels

})


print("Dataframe created successfully.")

print()


# ============================================================
# DATA AUGMENTATION
# ============================================================

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.2,

    rotation_range=15,

    width_shift_range=0.12,

    height_shift_range=0.12,

    zoom_range=0.15,

    shear_range=0.10,

    brightness_range=[
        0.8,
        1.2
    ],

    horizontal_flip=True,

    fill_mode="nearest"

)


# ============================================================
# TRAINING DATA
# ============================================================

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

    shuffle=True,

    seed=SEED

)


# ============================================================
# VALIDATION DATA
# ============================================================

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


# ============================================================
# MOBILE NET V2
# ============================================================

print()

print("Loading MobileNetV2 AI model...")

print()


base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),

    include_top=False,

    weights="imagenet"

)


# Freeze the original ImageNet layers

base_model.trainable = False


# ============================================================
# CREATE MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(classes),
        activation="softmax"
    )

])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


print()

print("Model created successfully!")

print()

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.00001

    )

]


# ============================================================
# START TRAINING
# ============================================================

print()

print("==========================================")
print("       STARTING AI TRAINING")
print("==========================================")

print()

print("Images:", len(image_paths))

print("Classes:", len(classes))

print("Image size:", IMAGE_SIZE)

print("Epochs:", EPOCHS)

print()

print("Training may take some time.")

print("Please do not close VS Code.")

print()


history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks

)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


model.save(
    MODEL_PATH
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    for class_name in classes:

        file.write(
            class_name + "\n"
        )


# ============================================================
# FINAL RESULT
# ============================================================

final_accuracy = history.history[
    "accuracy"
][-1]

final_val_accuracy = history.history[
    "val_accuracy"
][-1]


print()

print("==========================================")
print("       TRAINING COMPLETE!")
print("==========================================")

print()

print(
    f"Training Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{final_val_accuracy * 100:.2f}%"
)

print()

print("Model saved at:")

print(MODEL_PATH)

print()

print("Class names saved at:")

print(CLASS_NAMES_PATH)

print()

print("Your improved AI model is ready!")

print("==========================================")
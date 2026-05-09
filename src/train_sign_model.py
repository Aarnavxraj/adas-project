import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import yaml

if __name__ == "__main__":
    # Load config (run from project root)
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)["training"]

    TRAIN_DIR  = cfg["train_dir"]
    TEST_DIR   = cfg["test_dir"]
    MODEL_PATH = cfg["model_path"]
    IMG_SIZE   = tuple(cfg["img_size"])
    BATCH_SIZE = cfg["batch_size"]
    EPOCHS     = cfg["epochs"]

    print("Train path exists:", os.path.exists(TRAIN_DIR))
    print("Test  path exists:", os.path.exists(TEST_DIR))

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        brightness_range=[0.7, 1.3],
        horizontal_flip=False
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print(f"Train samples  : {train_generator.samples}")
    print(f"Test  samples  : {test_generator.samples}")
    print(f"Num classes    : {train_generator.num_classes}")

    if train_generator.samples == 0 or test_generator.samples == 0:
        raise ValueError("Dataset is empty. Run src/download_gtsrb.py first.")

    # -------------------------------------------------------------------------
    # Model architecture (deeper CNN for GTSRB 43-class)
    # -------------------------------------------------------------------------
    num_classes = train_generator.num_classes

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # LR scheduler: halve LR if val_loss doesn't improve for 3 epochs
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        )
    ]

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=callbacks
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Model saved   : {MODEL_PATH}")

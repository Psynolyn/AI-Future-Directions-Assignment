import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model


def preprocess(x, y, image_size):
    x = tf.image.resize(x, image_size)
    x = tf.cast(x, tf.float32) / 255.0
    return x, y


def main(args):
    image_size = (96, 96)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    num_classes = 10

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.map(lambda x, y: preprocess(x, y, image_size)).shuffle(5000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: preprocess(x, y, image_size)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.model_dir, 'best_model.h5'), save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    ]

    os.makedirs(args.model_dir, exist_ok=True)

    history = model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, callbacks=callbacks)

    model.save(os.path.join(args.model_dir, 'final_model.h5'))
    model.save(os.path.join(args.model_dir, 'saved_model'))

    loss, acc = model.evaluate(test_ds)
    print(f"Final test loss: {loss:.4f}, test accuracy: {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_dir', type=str, default='outputs/model')
    args = parser.parse_args()
    main(args)

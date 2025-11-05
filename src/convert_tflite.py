import argparse
import os
import numpy as np
import tensorflow as tf


def representative_data_gen(x_train, image_size, num_samples=100):
    for i in range(min(num_samples, x_train.shape[0])):
        img = x_train[i]
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        yield [img]


def convert(saved_model_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    image_size = (96, 96)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(os.path.join(out_dir, 'model_float32.tflite'), 'wb') as f:
        f.write(tflite_model)
    print('Saved model_float32.tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(out_dir, 'model_dynamic_quant.tflite'), 'wb') as f:
        f.write(tflite_model)
    print('Saved model_dynamic_quant.tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(x_train, image_size, num_samples=200)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    try:
        tflite_model = converter.convert()
        with open(os.path.join(out_dir, 'model_int8_fullquant.tflite'), 'wb') as f:
            f.write(tflite_model)
        print('Saved model_int8_fullquant.tflite')
    except Exception as e:
        print('Full integer quantization failed:', e)
    for fname in os.listdir(out_dir):
        fpath = os.path.join(out_dir, fname)
        print(fname, '-', os.path.getsize(fpath) / 1024.0, 'KB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs/tflite')
    args = parser.parse_args()
    convert(args.saved_model_dir, args.out_dir)

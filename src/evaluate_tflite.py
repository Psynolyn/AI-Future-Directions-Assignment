import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def load_test_data(image_size):
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.flatten()
    x_test = x_test.astype('float32') / 255.0
    x_resized = tf.image.resize(x_test, image_size).numpy()
    return x_resized, y_test


def evaluate_tflite(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']
    image_size = (input_shape[1], input_shape[2])
    x_test, y_test = load_test_data(image_size=image_size)
    correct = 0
    total = 0
    for i in range(len(x_test)):
        img = x_test[i]
        if dtype == np.uint8:
            inp = (img * 255).astype(np.uint8)
        else:
            inp = img.astype(np.float32)
        inp = np.expand_dims(inp, 0)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output[0])
        if pred == int(y_test[i]):
            correct += 1
        total += 1
        if i % 1000 == 0 and i > 0:
            print(f"Progress: {i}/{len(x_test)}")
    acc = correct / total
    print(f"TFLite model: {tflite_path} -- Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_model', type=str, required=True)
    args = parser.parse_args()
    evaluate_tflite(args.tflite_model)

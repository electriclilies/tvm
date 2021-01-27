import tvm
from tvm import relay
from tvm.relay.transform.quantize import Quantizer, Calibrater, TFDatasetManager, AverageMeanCalibrationCallback, Conv2DBiasAddPattern, Conv2DPattern, DensePattern, AddPattern, MultiplyPattern
import onnx
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
tf.enable_v2_behavior()

import numpy as np

batch_size = 5

# TFDS loading from https://www.tensorflow.org/datasets/keras_example
(ds_train, ds_test), ds_info = tfds.load('mnist',
                                         split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True,
                                         with_info=True)

# Import data
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

num_batches = 2000
mnist_train_manager = TFDatasetManager(ds_train, batch_size, 12000)
mnist_test_manager = TFDatasetManager(ds_test, batch_size, 2000)

# Import onnx model, quantize and calibrate
onnx_model = onnx.load('/home/lorthsmith/tvm/python/tvm/relay/new_quantize/mnist_model.onnx')
input_dict = {'flatten_input': [batch_size, 28, 28, 1]}
mod, params = relay.frontend.from_onnx(onnx_model, input_dict)
print(mod.astext())

cc = AverageMeanCalibrationCallback(mnist_train_manager)

quantizer = Quantizer(mod, params, [Conv2DBiasAddPattern(cc), Conv2DPattern(cc), DensePattern(cc), AddPattern(cc), MultiplyPattern(cc)])
calibrater = Calibrater(quantizer, target='llvm', ctx=tvm.cpu())
calibrated_mod = calibrater.calibrate()

with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(mod, target='llvm')
    q_lib = relay.build(calibrated_mod, target='llvm')
from tvm.contrib import graph_runtime
q_gmod = graph_runtime.GraphModule(q_lib["default"](tvm.cpu()))
gmod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
q_correct = 0
correct = 0
total = 0

while not mnist_test_manager.is_empty():
    images, labels = mnist_test_manager.get_next_batch()

    q_gmod.set_input(**{'flatten_input': images[0]})
    q_gmod.run()
    q_out = q_gmod.get_output(0).asnumpy()

    gmod.set_input(**{'flatten_input': images[0]})
    gmod.run()
    out = gmod.get_output(0).asnumpy()

    q_predicted_labels = np.argmax(q_out, axis=1)
    predicted_labels = np.argmax(out, axis=1)

    print("Int8 labels: ", q_predicted_labels)
    print("Float32 labels: ", predicted_labels)
    print("Actual labels: ", labels)

    q_correct += np.sum(q_predicted_labels == labels)
    correct += np.sum(predicted_labels == labels)
    total += batch_size

print("Int8 percent correct: ", (q_correct / total) * 100)
print("Float32 percent correct: ", (correct / total) * 100)
print("Difference: ", (((correct / total) * 100) - ((q_correct / total) * 100)))

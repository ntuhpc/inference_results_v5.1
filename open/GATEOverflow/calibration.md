# MLPerf Inference v5.1 - Calibration


## For Results using Nvidia Implementation

For the results taken using Nvidia implementation, we are following the same calibration procedure detailed by [Nvidia for their MLPerf Inference v5.0 submissions](https://github.com/mlcommons/inference_results_v5.0/blob/master/closed/NVIDIA/documentation/calibration.md)


## For Results using TFLite C++ Implementation

For the mobilenet and efficientnet submissions, we use quantized models from [TensorFlow Hub](https://tfhub.dev/). Details about the post-training quantization done for these models can be seen [here](
https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations)


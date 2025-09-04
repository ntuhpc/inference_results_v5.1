# TheStage AI - SDXL Submission

TheStage AI presents its submission for Stable Diffusion XL (SDXL), leveraging cutting-edge hardware and optimized software for high-performance ML inference. Our submission is designed to deliver fast and scalable inference using the latest advancements in GPU technology.

TheStage AI provides a powerful inference acceleration stack for AI models, allowing users to control models’ performance through a simple slider.

## System Configuration

The system for this submission is configured as follows:

- **Processor**: Intel(R) Xeon(R) Platinum 8468
- **Accelerator**: 8x NVIDIA H100-SXM-80GB GPUs
- **Software**: TheStage AI's custom solution

## Inference Submission

The model for this submission is created using **Qlip**, TheStage AI's full-stack AI framework for building, training, and deploying AI models. TheStage AI’s technology enables automatic application of quantization and pruning algorithms to optimize models for performance.

The model is accessible via [TheStage AI Elastic Models](https://github.com/TheStageAI/ElasticModels), a library of pre-compiled models with four performance tiers: XL, L, M, and S. This submission uses the smallest and fastest S model.

With Elastic Models, you can serve the desired model with a single line of code, either on your machine or in the cloud.

Detailed setup instructions for each workload are provided within the respective directories:

- **[code](code)**: Contains the source code and setup details for all workloads.
- **[measurements](measurements)**: Includes benchmarking commands and scenarios.
- **[calibration.md](documentation/calibration.md)**: Provides information on calibration and quantization techniques.

## Contact

For any queries, please reach out to hello@thestage.ai.


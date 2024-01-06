# neuralNetworks
This project comprises a series of experiments where build, analyze, and monitor neural networks. Moreover, we also explore an interpretability technique that I invented. That is, we build a neural network interpreter by fine-tuning Mistral7b &amp; Mixtral8x7b to interpret a target neural network in an automated way.



#### Building, Analyzing, Monitoring, Interpreting Neural Networks & Finetuning Large Language Models

#### Introduction
- **Project Overview**: This project comprises a series of experiments where build, analyze, and monitor neural networks. Moreover, we also explore an interpretability technique that I invented. That is, we build a neural network interpreter by fine-tuning Mistral7b & Mixtral8x7b to interpret a target neural network in an automated way.

#### Installation and Setup

To set up and run the code from this repository, you will need to ensure that your system meets the necessary hardware and software requirements. Here's a detailed guide:

##### Hardware Requirements:
- **GPU**: Most of the code can be executed on a consumer-grade GPU like the Geforce 4080 16GB. However, for fine-tuning large language models, especially models like Mixtral8x7b which require substantial GPU memory, a multiple GPU configuration is recommended. Alternatively, fine-tuning might be feasible on a single NVIDIA A100 GPU with some adjustments.
- **Multiple GPU Configuration**: For fine-tuning large language models, a setup with multiple GPUs is advised due to high memory requirements.

##### Software Requirements:
- **Python Libraries**: The project relies on several Python libraries. Below is a list of the primary libraries along with their installation commands. Ensure you have these installed in your Python environment:

  ```bash
  pip install jupyter notebook
  pip install torch torchvision
  pip install datasets
  pip install transformers
  pip install peft
  pip install accelerate
  pip install matplotlib numpy pandas
  pip install Pillow
  ```

- **CUDA**: I used CUDA 12.2 in combination with Python 3.10

##### Environment Setup:
- **Environment Variables**: Optionally, you can set the following environment variables for CUDA and Hugging Face token as depicted in the Jupyter Notebook:
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
  os.environ['HF_TOKEN'] = "your token"
  os.environ['TRANSFORMERS_CACHE'] = 'your Cache'
  ```
---


#### Training a Neural Network and Visualizing the Embedding Space
- [Link](https://github.com/Lukas-Santo-Puglisi/neuralNetworks/tree/main/testRun) to the code.  
- **Content**: In experiments 1 & 2 we train a large language model from scratch. The hyperparameters of this model originate from Experiment 3. Moreover, we visualize the embedding space after training showing that semantically similar words are close to each other in space. 

---

#### Finding good hyperparameters
-  [Link](https://github.com/Lukas-Santo-Puglisi/neuralNetworks/tree/main/hyperparameterTuningExperiment) to the code. 
- **Content**: In this experiment, we implement a random hyperparameter search.

---

#### Initializing a Neural Network
-  [Link](https://github.com/Lukas-Santo-Puglisi/neuralNetworks/tree/main/initialization) to the code. 
- **Content**: In experiment 5 we study the effects of Xavier initialization on the distribution of activations. Experiment 6 visualizes the effects of the input to the softmax to obtain a more uniform distribution before the training procedure starts. Experiment 9 visualizes variants of the  positional encoding scheme proposed in the initial transformer paper by Vaswani et al. 

---

#### Analyzing and Monitoring a Neural Network
-  [Link](https://github.com/Lukas-Santo-Puglisi/neuralNetworks/tree/main/gradientAnalytics) to the code. 
- **Content**: Experiment 4 visualizes the effects of different learning rates on neural networks' training stability and convergence speed. Experiment 7 visualizes the effects of layer and batch normalization on the ratio between the gradients and the parameter values, giving insight into the initial training process. Experiment 8 explores the gradient distribution before and after residual connections thus allowing us to train models with more than 50 layers.

---

#### Building a Neural Network Interpreter
-  [Link](https://github.com/Lukas-Santo-Puglisi/neuralNetworks/tree/main/trainingANeuralNetworkInterpreter) to the code. 
- **Content**: Experiments 10 and 11 train and analyze the target neural network. In experiment 12 we manually annotate 100 MNIST images with their respective shapes. In experiments 13 and 14, we construct a dataset and fine-tune Mistral7b and Mixtral8x7b to build neural network interpreters. Experiment 15 shortly teases how a powerful model like Chat-GPT 4 would act as an interpreter before any training. Experiment 16 displays the training and validation loss during fine-tuning. Experiments 17, 18, and 19 evaluate the trained neural interpreter's performance on various datasets and for various tasks.  

---
#### Acknowledgements
- **Contributors**: Thanks to Fabio Valdes and Jakob Johannes Metzger for their feedback.
- **Support**: Thanks to the Max-Delbrück Center for letting me run code on their cluster.


#### License
- **Project License**: MIT

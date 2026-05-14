# Distributed Edge AI Optimization

This project focuses on improving the efficiency of AI models for edge devices. Edge devices usually have limited memory, battery, and processing power, so running large AI models can be difficult. In this project, we train a baseline MNIST CNN model and then apply model optimization using quantization to reduce model size while keeping good accuracy.

---

## Project Goal

The main goal of this project is to compare a normal baseline AI model with an optimized quantized model.

We compare:

- Test accuracy
- Model size
- Number of parameters
- Suitability for edge devices

---

## Project Structure

```text
distributed-edge-ai-optimization/
│
├── baseline_mnist.py
├── optimized_quantized.py
├── federated_simulation.py
├── requirements.txt
├── README.md
│
├── notebooks/
│
├── data/              # ignored in GitHub
├── models/            # ignored in GitHub
├── results/           # ignored in GitHub
└── .gitignore
TraceCluster: A Lightweight and Adaptive Clustering-based Subgraph Attention Network for APT Detection in Provenance Graphs
Project Overview

This project implements the Cluster-GAT model based on Graph Neural Networks (GNNs) for APT (Advanced Persistent Threat) detection. By integrating Graph Attention Networks (GAT) and data partitioning, the model can efficiently handle large-scale Provenance Graph data and perform node classification. Specifically, the model addresses the class imbalance problem by using class weights to adjust the loss function during training, optimizing the model's training performance.

Features

Data Loading and Preprocessing: Automatically loads the training and validation datasets, constructs the graph data, and generates node features, label mappings, and edge type mappings.

Class Imbalance Handling: Computes and applies class weights to handle imbalanced datasets, enhancing the model's sensitivity to minority classes.

Graph Neural Network Model: Uses Graph Attention Network (GAT) for node classification in graph data, effectively learning complex relationships within the graph structure.

Training and Evaluation: Supports early stopping during the training process, trains and evaluates the model, and outputs loss and accuracy on the validation set to prevent overfitting.

Model Saving: Saves the best model weights after training for later use and deployment.

Requirements

Python >= 3.6

PyTorch >= 1.7

PyTorch Geometric >= 1.7

scikit-learn >= 0.24

Note

In this repository, we provide only key parts of the implementation, focusing on the core components such as the Cluster-GAT model, data processing, and training loop. The complete implementation is not included in this code release. The provided code can be used as a foundation for understanding the main logic and for extending or adapting it to different tasks or datasets.

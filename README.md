# Federated Learning Simulator with FedAvg

A Python-based simulation framework for decentralized machine learning using the standard Federated Averaging (FedAvg) algorithm. This repository demonstrates how to train a global deep learning model across multiple localized data silos (clients) without requiring raw data collection at a central server.

The project features an interactive Streamlit user interface that visualizes data distribution imbalances, training loss dynamics, and global accuracy convergence in real time.

## System Architecture

The simulation runs locally on a single machine but models a distributed server-client cluster:

1. **Global Broadcast:** The server initializes a central Convolutional Neural Network (SimpleCNN) and distributes its weights to all active clients.
2. **Local Training:** Each client trains its local copy of the model on its own isolated data subset using Stochastic Gradient Descent (SGD).
3. **Weight Upload:** Clients send their updated model weights back to the server. The raw data never leaves the client container.
4. **Weighted Aggregation:** The server aggregates the client weights by computing a weighted average proportional to each client's sample size, updating the global model for the next communication round.

## Core Features

* **Data Partitioning (Non-IID):** Simulates realistic, biased data environments by sorting the MNIST dataset by label before splitting it. Each client receives data dominated by specific digit classes.
* **Enhanced Model Stability:** The central architecture includes Batch Normalization to stabilize weight updates across highly divergent client states, along with Dropout regularization to mitigate local overfitting.
* **Hardware Acceleration:** Native PyTorch integration automatically detects and binds to local NVIDIA CUDA GPUs, shifting to CPU computation only as a fallback.
* **Proactive VRAM Management:** Incorporates systematic pointer deletion and cache flushing commands at the end of each local client cycle to guarantee stability on limited-memory devices during large-scale runs.
* **Dual-Stream Telemetry:** The interface uses non-blocking placeholders to stream dual line charts tracking test accuracy metrics and cross-entropy loss variations simultaneously.

## Repository Layout

```text
fedratd_ml/
├── data/               # Local repository for raw MNIST download files
├── .gitignore          # Rules for preventing local data and cache commits
├── README.md           # Core project documentation and technical overview
├── app.py              # Front-end dashboard configuration and execution engine
└── mg.py               # Deep learning infrastructure, training loops, and aggregation math

to be continued 

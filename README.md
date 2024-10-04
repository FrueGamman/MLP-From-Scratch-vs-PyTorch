# MLP-From-Scratch-vs-PyTorch

## Project Overview
This project involves implementing a Multilayer Perceptron (MLP) from scratch using Python and comparing it with an implementation using a high-level library like PyTorch. The comparison aims to highlight the differences in performance, training time, and implementation complexity.

## Steps to Complete the Project

### Step 1: Prepare the Dataset
1. Go to the Ecoli dataset using the link provided in the assignment.
2. Filter the dataset so that it only contains the classes 'cp' and 'im'.
3. This step ensures that the training data is correctly prepared for the neural network models.

### Step 2: Implement the MLP from Scratch
1. Implement a Multilayer Perceptron (MLP) using Python without high-level libraries.
2. Implement both forward propagation and backward propagation algorithms.
   - Forward propagation calculates the outputs from the network based on the input data.
   - Backward propagation updates the network's weights based on the error between the expected output and the actual output.
3. This implementation provides a deeper understanding of how MLPs work.

### Step 3: Implement the MLP Using a High-Level Library
1. Implement an MLP using a high-level library like PyTorch or TensorFlow.
2. These libraries provide pre-built functions and optimizations that simplify the process of building and training neural networks.

### Step 4: Training and Testing
1. Train both MLP implementations (from scratch and with a high-level library) using the prepared Ecoli dataset.
2. Split the dataset into training and test sets to evaluate the performance of the models.

### Step 5: Evaluation and Comparison
1. Evaluate and compare the two MLP implementations in terms of:
   - Performance: How accurately each model classifies the data in the test set.
   - Training time: How long it takes to train each model.
   - Implementation complexity: How complex the code is for each implementation.
2. This comparison helps understand the advantages and disadvantages of implementing MLPs from scratch versus using high-level libraries.

## Directory Structure
- `README.md`: Project overview and instructions.
- `requirements.txt`: Python dependencies required for the project (to be included).
- `src/`: Source code for MLP implementations.


---

Please review the draft and let me know if there are any sections that need more details or modifications.

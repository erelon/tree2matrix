# Project Title: Hierarchical Tree to Matrix Conversion for CNN Models (HERMANN)

This project introduces a novel model, HERMANN, that converts hierarchical tree structures with values into matrices suitable for Convolutional Neural Networks (CNNs). The core purpose of this project is to provide a new approach for handling hierarchical data in deep learning models, particularly CNNs.

## Overview

The project compares the performance of the HERMANN model in four different fields with well-known models. The comparison metrics include accuracy, Area Under the Curve (AUC), and F1 score. 

## Core Component

The core component of this project is located in the `main.py` file. This file contains the implementation of the HERMANN model and the necessary procedures to train and evaluate it.

## How to Use

To use the HERMANN model, follow the steps outlined in the `main.py` file. The file contains detailed comments and instructions on how to initialize, train, and evaluate the model. 

## Requirements

- Python 3.9.1
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/erelon/tree2matrix
cd tree2matrix
pip install -r requirements.txt
```

## Usage
### Tree2Matrix

The `Tree2Matrix` class is used to convert a tree structure into a matrix representation. Here's how to use it:

1. **Initialize the class**

```python
t2m = Tree2Matrix()
```

2. **Transform a tree into a matrix**

```python
tree = generate_random_tree(10)  # Replace with your tree
m, N = t2m.transform(tree, "val", "name")
```

In the `transform` method, the parameters are as follows:

- `tree`: This is the tree you want to transform. It should be a `networkx.DiGraph` object.
- `value_key`: This is the key in the node data dictionary that holds the value you want to use for the matrix.
- `name_key`: This is the key in the node data dictionary that holds the name of the node. This is optional.

The `transform` method returns two numpy arrays: `m` is the matrix representation of the tree, and `N` contains the names of the nodes.

### MatricesDendrogram

The `MatricesDendrogram` class is used to create a dendrogram from a list of matrices. Here's how to use it:

1. **Initialize the class**

```python
md = MatricesDendrogram()
```

2. **Transform a list of matrices into a dendrogram**

```python
matrices_result, new_order_names = md.transform(M, N)
```

In the `transform` method, the parameters are as follows:

- `M`: This is a list of matrices you want to transform. Each matrix should be a numpy array.
- `N`: This is a list of names corresponding to the matrices.

The `transform` method returns two numpy arrays: `matrices_result` is the dendrogram representation of the matrices, and `new_order_names` contains the names of the nodes in the new order.

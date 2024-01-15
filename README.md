# Multi-Branch Temporal Convolutional Network (MB-TCN) for Monaural Speech Enhancement

## Overview
This repository contains the implementation of a Multi-Branch Temporal Convolutional Network (MB-TCN) for monaural speech enhancement. The model architecture is based on the paper "Monaural Speech Enhancement Using a Multi-Branch Temporal Convolutional Network."

## Model Architecture

### TemporalConvNetBranch Class
- Represents a single branch of the temporal convolutional network.
- Consists of a convolutional unit with normalization and ReLU activation.

### MBTemporalConvNetBlock Class
- Comprises eight instances of TemporalConvNetBranch to create multiple branches.
- Concatenates the outputs of the branches and applies normalization, ReLU activation, and another convolutional layer.

### MBTemporalConvNet Class
- Utilizes multiple instances of MBTemporalConvNetBlock to construct the overall model.
- The number of blocks is determined by the parameter N.
- The model concludes with a dense layer with sigmoid activation.

## Configuration
The model's hyperparameters are defined in the `config.yaml` file:

- `dmodel`: Dimensionality of the model's output (256 in the provided example).
- `df`: Number of filters in the convolutional layers (64 in the provided example).
- `k`: Kernel size for the convolutional layers (30 in the provided example).
- `D`: A parameter influencing the dilation rate in the model (16 in the provided example).
- `N`: Number of MBTemporalConvNetBlock instances in the model (20 in the provided example).

## Usage
To use the model, instantiate an object of the MBTemporalConvNet class, providing the configuration parameters. Example usage is demonstrated in `test.py`.


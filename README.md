# Iris Neural Networks Classification

This repository contains the implementation of three types of neural networks — Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN)—for classifying the Iris dataset. The implementation is done using PyTorch and scikit-learn.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Iris-NeuralNetworks-Classification.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Iris-NeuralNetworks-Classification
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script:

   ```bash
   jupyter notebook iris-neural-networks-classification.ipynb
   ```

   or

   ```bash
   python iris-neural-networks-classification.py
   ```

## Description

The project focuses on implementing and comparing the performance of three neural network architectures—MLP, CNN, and RNN—on the Iris dataset. The Iris dataset is loaded from scikit-learn, and the features are visualized using scatter plots. The data is then preprocessed, split into training and testing sets, and standardized.

### Implemented Neural Networks:

1. **Multilayer Perceptron (MLP):**
   - Input Features: 4
   - Hidden Layers: 2 (10 neurons each)
   - Output Classes: 3

2. **Convolutional Neural Network (CNN):**
   - Input Channels: 1
   - Convolutional Layer: 1D with 16 output channels and a filter size of 3
   - Max Pooling Layer: 1D with a window size of 2
   - Fully Connected Layer: 3 output classes

3. **Recurrent Neural Network (RNN):**
   - LSTM Layer: 1 with 8 hidden units
   - Fully Connected Layer: 3 output classes

## Results and Visualization

The project includes training loops for each neural network, visualizing the loss, accuracy, precision, and recall over epochs. Consecutive loss increases are monitored, and early stopping is implemented to prevent overfitting.

To visualize the results, run the provided Jupyter Notebook or Python script.

## Sample Classification

The implementation includes a sample classification test with new iris samples. The MLP, CNN, and RNN models predict the classes of these samples and print the results.

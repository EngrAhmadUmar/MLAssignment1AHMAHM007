# MLAssignment1AHMAHM007

Handwritten Digit Classification using Feedforward Neural Networks


This project implements a feedforward neural network to classify handwritten digits from the MNIST dataset. The neural network is implemented in Python using PyTorch and Torchvision libraries. The project consists of the following files:
classifier.py: This file contains the main script for training the neural network, evaluating its performance, and interactive classification of handwritten digit images.
mnist_model.pth: This file contains the trained model weights saved after training the neural network. It is loaded by the classifier.py script for inference.
log.txt: This text file contains the training loss after each epoch and the final accuracy on the test set obtained during a previous training run of the model.


Usage
To train the neural network and perform interactive classification of handwritten digit images, follow these steps:
Make sure you have Python installed along with the necessary libraries (PyTorch, Torchvision, scikit-learn).
Run the classifier.py script using the following command:
python classifier.py
This will train the neural network on the MNIST dataset, save the trained model weights to mnist_model.pth, and allow you to enter file paths to handwritten digit images for classification.
Follow the prompts to enter file paths to images, and the script will output the predicted digit for each image.


Files
classifier.py: Main script for training the neural network, evaluating its performance, and interactive classification.
mnist_model.pth: Trained model weights saved after training.

Note
The log.txt file contains output from a previous training run of the model. This information is provided for reference and validation of the model's performance.

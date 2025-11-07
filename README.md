# Neural Network CLI
A simple neural network built with reLU in hidden layer and softmax cross entropy function is used to train multi-class classification models. Also, an Adam Optimizer is used to train the models efficiently.

It is trained until it reaches a loss of less than 0.005, where we assume that a good predicition is made and program is terminated then outputs the results.

The program also keeps track of the loss and create a csv file to track them. The loss function in relation to the epoch is then demonstrated by plotting a graph using MATLAB.

# HOW IT WORKS
- The computer generates a random integer between the values (1-16) and converts it into a 4-bit binary value.
- There are three phases: Prediction , Training and Optimization

# 1. Prediction phase (forward pass):
  - The generated binary value is passed into the input layer. At each neuron, calculation is made (z = input * weights + bias) which is then passed through an activation function.
- The output (z) is passed to reLU where another calculation is made, f(x) = max(0,z).
- This output is finally passed onto the softmax output layer, where the raw score is converted into a probability distribution.

# 2 Learning phase (backward pass):
- Compare the prediction to the target. Using cross-entropy loss, a single number that represents how wrong the prediction was gets calculated.
- With the calculated loss, backpropagation occurs. It calculates gradient of loss with respect to every single weight and bias. It goes through all the layers backwards till it reaches the first hidden layer. It gives a gradient for every weight and bias.

# 3 Adam Optimizer:
- Takes in all the gradients and updates the weights and biases accordingly to generate a better network.


This entire process is repeated for each epoch until a target loss of less than 0.005 is reached.

# RESULT:
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/f57abc80-64e0-47aa-8d75-16f3229f9dbe" />
<img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/3ff78170-3ef6-4928-a70e-7dec801e70cc" />

PS: The computer I used to compile this code had an older version of windows therefore the ANSII colors I used was not showing.


  

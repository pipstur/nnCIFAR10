# Neural Network Architectures

Note: Seems the formatting in the notebook is kinda off, but I think it's a github problem currently

## SimpleCNN

The `SimpleCNN` class is a basic Convolutional Neural Network (CNN) for image classification. It consists of three main types of layers:

1. **Convolutional Layers (`nn.Conv2d`):**
   - `self.conv1`: Takes an input with 3 channels (for RGB images) and applies 64 filters with a kernel size of 3x3 and padding of 1.
   - `self.conv2`: Applies 128 filters to the output of the first convolutional layer with the same kernel size and padding.
   
2. **Activation Function (`nn.ReLU`):**
   - `self.relu1`, `self.relu2`: Applies the Rectified Linear Unit (ReLU) activation function after each convolutional layer.

3. **Pooling Layers (`nn.MaxPool2d`):**
   - `self.pool1`, `self.pool2`: Apply max-pooling with a 2x2 kernel and a stride of 2.

4. **Fully Connected Layers (`nn.Linear`):**
   - `self.fc1`: Takes the flattened output of the second pooling layer and applies a fully connected layer with 512 neurons.
   - `self.fc2`: Applies another fully connected layer to produce the final output with the specified number of classes.

5. **Forward Method (`forward`):**
   - Defines the forward pass of the network, chaining the convolutional, activation, and pooling layers followed by the fully connected layers.

## VGGNet

The `VGGNet` class is inspired by the VGG architecture, known for its simplicity and effectiveness. It is deeper than `SimpleCNN` and features more convolutional layers:

1. **Convolutional Layers (`nn.Conv2d`), Activation (`nn.ReLU`), and Pooling (`nn.MaxPool2d`):**
   - Similar to `SimpleCNN` but with additional convolutional layers, providing increased depth.

2. **Fully Connected Layers (`nn.Linear`):**
   - The final fully connected layers operate on the flattened output of the last pooling layer.

3. **Forward Method (`forward`):**
   - Defines the forward pass through the convolutional, activation, and pooling layers, followed by the fully connected layers.

## ResidualBlock

The `ResidualBlock` class defines a basic residual block used in the ResNet architecture. A residual block contains two convolutional layers with a shortcut connection, allowing the network to learn residual functions.

1. **Convolutional Layers (`nn.Conv2d`), Batch Normalization (`nn.BatchNorm2d`), Activation (`nn.ReLU`):**
   - `self.conv1` and `self.conv2` form the convolutional layers with batch normalization and ReLU activation.
   
2. **Shortcut Connection:**
   - The `self.shortcut` module provides a shortcut connection, allowing the input to bypass the residual block if necessary.

3. **Forward Method (`forward`):**
   - Defines the forward pass through the residual block, combining the output of convolutional layers and the shortcut connection.

## ResNet

The `ResNet` class assembles multiple `ResidualBlock` instances to create a Residual Neural Network (ResNet). It has a specific layer structure with varying block counts:

1. **Convolutional Layer (`nn.Conv2d`), Batch Normalization (`nn.BatchNorm2d`), Activation (`nn.ReLU`):**
   - The initial layers (`self.conv1`, `self.bn1`, `self.relu1`) process the input image.
   
2. **Residual Blocks (`ResidualBlock`):**
   - Four layers (`self.layer1` to `self.layer4`) consist of multiple residual blocks with increasing channels and reduced spatial dimensions.

3. **Global Average Pooling (`nn.AdaptiveAvgPool2d`):**
   - Performs global average pooling to reduce spatial dimensions to 1x1.

4. **Fully Connected Layer (`nn.Linear`):**
   - The flattened output is passed through a fully connected layer to produce the final class predictions.

5. **Forward Method (`forward`):**
   - Defines the forward pass through the convolutional layers, residual blocks, and final fully connected layer.

These architectures demonstrate the evolution from a simple CNN (`SimpleCNN`) to deeper and more sophisticated architectures like VGGNet and ResNet, which incorporate residual connections for improved training and convergence.

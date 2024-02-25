# Computer Vision

These notes about the Computer Vision are suggested by the [Stanford Course CS231N](https://www.youtube.com/playlist?list=PLf7L7Kg8_FNxHATtLwDceyh72QQL9pvpQ).

## Lecture 1 - Introduction to Convolutional Neural Networks for Visual Recognition
In this Lecture, Professor Fei-Fei Li presented the history of computer vision practices. One of the key focuses in computer vision is understanding how to translate images into mathematical representations. One approach involves conceptualizing real-world objects as assemblies of shapes, such as cylinders, or as assemblies of edges and points.

Moreover, a significant portion of the problems addressed in this field revolves around *classification tasks*. If colors play a crucial role in conveying important features within an image, the RGB (Red, Green, Blue) color encoding scheme can be employed. This encoding facilitates the representation of an image as a three-dimensional matrix, where the dimensions correspond to height, width, and RGB values.

To further elaborate on this introductory overview, it's essential to recognize that computer vision extends beyond classification problems. Object detection, segmentation, and image captioning are additional areas within the discipline that contribute to a comprehensive understanding of visual data. Additionally, advancements in deep learning techniques, particularly **convolutional neural networks** (CNNs), have significantly enhanced the capabilities of computer vision systems in recent years. These networks excel at automatically learning hierarchical features from images, making them highly effective for various tasks, including image recognition and object detection.

## Lecture 2 - Image Classification

Recognizing objects like a cat in an image poses a formidable challenge, often necessitating a data-driven approach comprising both a training phase and a prediction phase. Deep learning, and specifically convolutional neural networks (CNNs), emerge as a powerful solution.

In the initial stages of addressing image classification problems, one may consider the **K-Nearest Neighbor** (KNN) algorithm. This approach involves placing each training point in a multidimensional space. Subsequently, the prediction label for a test sample is determined by the labels of its k-nearest training samples. However, this algorithm has certain limitations.

Firstly, it exhibits *O(1)* complexity during training, making it efficient in this phase. Nevertheless, during test predictions, the algorithm's complexity is *O(N)*, as it involves measuring the distance of the test sample to each training sample. This can lead to computationally expensive operations, particularly with large datasets.

Additionally, the KNN algorithm may encounter challenges when dealing with outliers, as it relies heavily on the distances between points. Outliers can significantly influence the determination of nearest neighbors, potentially leading to inaccurate predictions.

Furthermore, the presence of regions with a lack of majority label poses a substantial challenge. This situation often arises in areas where no single label class predominates, commonly referred to as the "white region." The success of the KNN algorithm is also contingent on the choice of a suitable distance metric, as different metrics may yield different results. Therefore, careful consideration of the distance metric is essential for optimal performance in image classification tasks.

As the field progresses, the limitations of traditional algorithms like KNN have led to the widespread adoption of deep learning techniques, particularly CNNs, which excel at automatically learning hierarchical features from images, thereby addressing many of the challenges associated with image recognition and classification.

As with any other algorithm, we need to set the hyperparameters, such as K, the number of considered neighbors (it is a bad idea to set this equal to 1, as it could lead to overfitting). 

We also need to split our dataset, preferably into three sets: training, validation, and test. The presence of validation can give us a measure of how well the algorithm performs on new data points, points not used for training. Consideration can be given to using K-fold cross-validation as well.

However, KNN is not commonly used on images. The distance between pixels is not very informative, and it tends to be slow. Another problem is the **curse of dimensionality**.

What if we tried to use linear classification? It's the simplest way to build a parametric model. The parameters are the input $X$ and $W$, the matrix of weights for the prediction function. The output consists of 10 numbers that encode the class labels.

$f(x, W) = Wx + b \in \mathbb{R}^{m}$

where $m$ is the cardinality of the output set. To execute this operation, we need to stretch the pixel matrix into columns, achieved by performing the **flatten** operation.

However, a linear classifier works only if the dataset is linearly separable. This is why we introduce neural networks.

## Lecture 3 - Loss Functions and Optimization

Before delving deep into deep learning, it's important to understand how a data-driven training process works. At each training iteration, we aim for a new and improved set of parameters for our classification function. But how and why do we modify these weights? Thanks to the **optimization of the loss function**, a measure of how well our algorithm performs.

$L = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i)$

where $x_i$ is the i-th image, and $y_i$ is its integer label in the ground truth.

The initial loss function under discussion is the **multi-class Support Vector Machine** (SVM) loss function.

In machine learning, particularly in the context of classification tasks, the multi-class SVM loss function is a crucial component. This loss function is designed to optimize the performance of a model when faced with multiple classes or categories.

The multi-class SVM loss function operates by assigning a score to each class for a given input and penalizing the model based on the difference between the predicted scores and the ground truth scores. The objective is to encourage correct class predictions and simultaneously penalize deviations from the correct classification.

Mathematically, the loss function is expressed as the sum of the margins of the correct class and the maximum margin among all incorrect classes. The formulation involves minimizing this sum, effectively pushing the correct class scores higher than the incorrect ones by a certain margin.

This loss function is instrumental in training models for tasks such as image classification, where an input can belong to one of several classes. Understanding and effectively implementing the multi-class SVM loss function contribute significantly to enhancing the performance of machine learning models.

$L_i = \sum_{j \neq y_i}{\max{0, s_j - s_{y_i} + 1}}$

and its shape looks like the *Hinge loss*:

![Hinge Loss](https://media.geeksforgeeks.org/wp-content/uploads/20231109124420/1-(1).png)

After calculating the SVM loss for each data point in the entire training dataset, we obtain the overall loss, which is the average of the individually calculated SVM losses.

$L = \frac{1}{N} \sum_{i=1}^{N}{L_i}$ 

Continuing further, we can introduce regularization, which is a technique used to manage the incremental complexity of the model being employed. There are several types of regularization terms:

- **L1 regularization**
- **L2 regularization**
- **Elastic Net (L1 + L2)**
- **Max Norm Regularization**

These terms measure the complexity of the model and are added to the loss function that needs to be minimized.

Other techniques that share the same objective as regularization are dropout and batch normalization.

Certainly, there are various loss functions, often dependent on the algorithm being implemented. In the context of a classification problem, one can also implement a Softmax Classifier, an algorithm that produces a probability vector as output.


$L_i = -log P(Y = y_i)(X = x_i) = -log \frac{e^{s_{y_i}}}{\sum_{j}{e^{s_j}}}$

where

$s = f(x_i, W)$

But how can we find the W, the set of weights, which minimizes the loss function? Thanks to **optimization**. Optimization is like finding the lowest point in a mountain valley. Now, with the loss function we saw, it may seem simple, but with neural networks, especially with deep neural networks, the loss function becomes more complex than these.

A first idea solution is *random search*, attempting to guess the set of weights.

Another bad idea is *complete enumeration*, trying every possible set of weights, but it has a problem with computing time.

Another strategy can be *following the slope*. Thanks to *gradient computation*, the vector of the partial derivatives of a function (the loss function, in our case), the direction of steepest descent is the negative gradient.

**Gradient descent** is a very powerful method to update the weights of the neural network. We first need to initialize W and then update W in each iteration to reach the minimum area of the Loss Function.

\[W = W - \text{{step-size}} \times \text{{gradient}}_W\]

and the step size is commonly known as **learning rate** $\eta$.

A different version of this method is **Stochastic Gradient Descent** (SGD).

## Lecture 4 - Introduction to Neaural Networks
How we saw in the previous Lecture, the most useful method used to compute the optimal set of weights W is the gradient descent.
Also, we said that it is very difficult to compute when the loss function, which is the function to be minimized, become more complex, as in the neural networks.
To understand the derivative of a complex function, we can think about the computational graphs. In fact, a function can be represented by a graph where each operation is a node .
This is a first approach to the **backpropagation**, a recursive application of the chain derivative rule.
Thanks to this we will able to identify the exact weight responsbile for a bad prediction, and then modify it.

## Lecture 5 - Convolutional Neural Networks

### Convolutional Neural Networks (CNNs) Overview

In the context of Convolutional Neural Networks (CNNs), the primary objective is to process images efficiently for subsequent classification or prediction tasks. Let's delve into the functional perspective of CNNs and their key components.

Here is the functional workflow:
1. **Image to Multi-dimensional Array:**
   - Convert the input image into a multi-dimensional array.
   - For instance, consider a 32x32x3 image, where each pixel is represented by an RGB 3-dimensional vector.

2. **Flatten Array as Fully Connected Layer Input:**
   - Utilize this multi-dimensional array as the input for a fully connected layer with adjustable weights.
   - The weights in this layer are optimized during the training process.

3. **Activation Function for Prediction:**
   - Apply an activation function to the output of the fully connected layer to obtain the final prediction.

![Functional Workflow](https://miro.medium.com/v2/resize:fit:1200/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)

The **Convolutional Layer** plays a crucial role in the CNN by reducing the spatial structure of the input image while preserving essential features. This layer involves the use of Convolutional Filters, which slide over the image and compute dot products to generate activation maps.
This layer has two goals:
   - Employ a Convolutional Filter (e.g., 5x5x3) that traverses the image spatially, performing dot product computations.
   - This operation results in an activation map, such as a 28x28x1 map in the provided example.

![Convolutional Filter Operation](https://miro.medium.com/v2/resize:fit:1400/1*EQe39aT2EIrjGDG3v5Burg.png)

But in a **Convolutional Neural Networks**, we have to use more than one filter in a Convolutional Layer:
   - Utilize multiple filters with distinct purposes, such as detecting edges, curves, or analyzing depth.
   - Each filter contributes to extracting specific features from the input image.
A filter can be set in very different ways: we can set parameters such as *stride* and *padding*. *Stride* refers to the step the filter takes at each iteration while sliding on the Activation Map in input; *Padding* is the border of the Convolutional Filter (usually, we work with zero-padding) and it is usally used when the dimension of the filter does not fit with the dimension o the input matrix. And we use the *zero-padding* to mantain the same information but in a different wat (or size)

![Multiple Filters](https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png?w=736)

To use more filter has the following consequences:
   - Stack Convolutional Layers to create a hierarchy of information extraction.
   - Lower layers capture low-level features, while higher layers focus on high-level features.

After the implementation of each filter, we can pass the output array to the **Pooling Layer**, which makes the representations smaller (*downsample*) and more manageable. The most commonly used *Pooling layer* is the **Max-Pooling**. To implement this layer, we need to define the size of the filter and its stride. *Max Pooling* is not the only option; there are several pooling filters we can use.

![Max Pooling](https://th.bing.com/th/id/R.4738d8f79d2d381a7d586526b229409e?rik=OJCrwrv5eizlaQ&pid=ImgRaw&r=0)

After a sequence of several Convolutional Layers, we are ready to pass the information to the last part of our CNN, the **fully connected layer**, which has the purpose of making the effective prediction or classification for the input image.

## Lecture 6 - Training Neural Networks I

### Optimization Algorithms

#### Gradient Descent
**Gradient Descent** is a foundational optimization algorithm used to minimize the cost function in machine learning models. It works by iteratively moving towards the minimum of the cost function by updating the parameters in the opposite direction of the gradient of the cost function with respect to the parameters.

- **Process**: Calculate the gradient (partial derivatives) of the cost function for the whole training dataset concerning each parameter in the model, and update the parameters simultaneously.
- **Challenge**: Computationally expensive and slow for large datasets, as it requires calculating the gradients for the whole dataset to make a single update.

#### Stochastic Gradient Descent (SGD)
**SGD** is a variation of gradient descent that updates the model's parameters using only a single sample or a small subset of the training data at a time. This randomness can help the model to escape local minima.

- **Process**: Instead of calculating the gradients for the whole dataset, SGD randomly selects one data point (or a small batch) to calculate the gradient and update the parameters.
- **Benefits**: Significantly faster computation per iteration and the ability to escape local minima.
- **Challenges**: Higher variance in the parameter updates, which can cause the cost function to fluctuate heavily.

#### Mini-batch Gradient Descent
**Mini-batch Gradient Descent** strikes a balance between the robustness of full-batch gradient descent and the efficiency of SGD. It updates the model's parameters by calculating the gradient of the cost function using a subset of the training data.

- **Batch Size**: Typically ranges from 10 to a few hundred samples. The choice of batch size can significantly affect the convergence speed and training stability.
- **Benefits**: Reduces the variance of the parameter updates compared to SGD, leading to more stable convergence. More efficient than full-batch gradient descent on large datasets.
- **Challenges**: Requires tuning of the batch size hyperparameter to find the optimal balance between efficiency and stability.

### Hyperparameter Optimization Techniques

*Hyperparameter optimization* is a crucial step in designing and training neural network models. It involves finding the most effective combination of hyperparameters that yields the best performance. Here's a detailed look at the main techniques for hyperparameter optimization.

#### 1. Grid Search
**Grid Search** is the most straightforward method for hyperparameter optimization. It systematically works through multiple combinations of hyperparameter values, evaluating each combination to find the best result.

- **Process**: Define a grid of hyperparameter values and exhaustively train models with every combination of these values.
- **Pros**: Simple to implement and guaranteed to find the best combination within the grid.
- **Cons**: Computationally expensive and inefficient as the number of hyperparameters increases (curse of dimensionality).

#### 2. Random Search
**Random Search** selects random combinations of hyperparameters to train the model, rather than exhaustively trying all combinations.

- **Process**: Define a search space as the range of values for each hyperparameter and randomly select combinations to train the model.
- **Pros**: More efficient than Grid Search, especially when some hyperparameters do not influence the outcome.
- **Cons**: Still requires a large number of iterations to find optimal hyperparameters and may miss the best combination.

#### 3. Bayesian Optimization
**Bayesian Optimization** uses a probabilistic model to guide the search for the best hyperparameter combination. It is more efficient than Grid and Random search, especially for high-dimensional spaces.

- **Process**: Build a probabilistic model of the function mapping hyperparameters to the objective evaluated on the validation set. Use this model to select the most promising hyperparameters to evaluate in the true objective function.
- **Pros**: Efficiently finds the best hyperparameters by focusing on areas predicted to offer improvement.
- **Cons**: Complexity in implementation and model selection.

## Lecture 7 - Training Neural Networks II
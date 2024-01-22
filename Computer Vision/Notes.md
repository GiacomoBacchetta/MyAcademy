# Computer Vision

These notes about the Computer Vision are suggested by the [Stanford Course CS231N](https://www.youtube.com/playlist?list=PLf7L7Kg8_FNxHATtLwDceyh72QQL9pvpQ).

## Lesson 1
In this lesson, Professor Fei-Fei Li presented the history of computer vision practices. One of the key focuses in computer vision is understanding how to translate images into mathematical representations. One approach involves conceptualizing real-world objects as assemblies of shapes, such as cylinders, or as assemblies of edges and points.

Moreover, a significant portion of the problems addressed in this field revolves around *classification tasks*. If colors play a crucial role in conveying important features within an image, the RGB (Red, Green, Blue) color encoding scheme can be employed. This encoding facilitates the representation of an image as a three-dimensional matrix, where the dimensions correspond to height, width, and RGB values.

To further elaborate on this introductory overview, it's essential to recognize that computer vision extends beyond classification problems. Object detection, segmentation, and image captioning are additional areas within the discipline that contribute to a comprehensive understanding of visual data. Additionally, advancements in deep learning techniques, particularly **convolutional neural networks** (CNNs), have significantly enhanced the capabilities of computer vision systems in recent years. These networks excel at automatically learning hierarchical features from images, making them highly effective for various tasks, including image recognition and object detection.

## Lesson 2

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

## Lesson 3
Before delving deep into deep learning, it's important to understand how a data-driven training process works. At each training iteration, we aim for a new and improved set of parameters for our classification function. But how and why do we modify these weights? Thanks to the **optimization of the loss function**, a measure of how well our algorithm performs.

$ L = \frac{1}{N} \sum_i{L_i(f(x_i, W), y_i)} $

where $x_i$ is the i-th image, and $y_i$ is its integer label in the ground truth.

The initial loss function under discussion is the **multi-class Support Vector Machine** (SVM) loss function.

In machine learning, particularly in the context of classification tasks, the multi-class SVM loss function is a crucial component. This loss function is designed to optimize the performance of a model when faced with multiple classes or categories.

The multi-class SVM loss function operates by assigning a score to each class for a given input and penalizing the model based on the difference between the predicted scores and the ground truth scores. The objective is to encourage correct class predictions and simultaneously penalize deviations from the correct classification.

Mathematically, the loss function is expressed as the sum of the margins of the correct class and the maximum margin among all incorrect classes. The formulation involves minimizing this sum, effectively pushing the correct class scores higher than the incorrect ones by a certain margin.

This loss function is instrumental in training models for tasks such as image classification, where an input can belong to one of several classes. Understanding and effectively implementing the multi-class SVM loss function contribute significantly to enhancing the performance of machine learning models.

$L_i = \sum_{j \neq y_i}{\max{0, s_j - s_{y_i} + 1}}$

and its shape looks like the *Hinge loss*:

![Hinge Loss](data:image/png;base64,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)

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
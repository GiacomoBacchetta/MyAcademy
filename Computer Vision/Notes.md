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

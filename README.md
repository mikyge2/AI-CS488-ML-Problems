# Machine Learning on Iris Dataset

This repository contains solutions to a series of machine learning problems using the Iris dataset. Each problem focuses on different machine learning algorithms and tasks, including classification and clustering, along with performance evaluation and visualization. Below is a summary of the problems and their corresponding tasks.

## Problem #1: Decision Tree Depth Optimization (2.5 pts)

### Task:
- Train a decision tree classifier on the Iris dataset.
- Optimize its performance by adjusting the maximum depth of the tree.

### Instructions:
1. Split the dataset into training and testing sets.
2. Train a decision tree with various maximum depths.
3. Evaluate the performance of each model using accuracy, precision, recall, and F1-score.
4. Visualize the decision boundaries of the best-performing model.
5. Plot performance metrics against tree depth to analyze the impact.

### Objective:
Understand how tree depth affects overfitting and underfitting in decision trees.

## Problem #2: K-Nearest Neighbors Hyperparameter Tuning (2.5 pts)

### Task:
- Implement a k-nearest neighbors (KNN) classifier on the Iris dataset.
- Optimize the number of neighbors (k) and distance metrics (e.g., Euclidean, Manhattan).

### Instructions:
1. Split the dataset into training and testing sets.
2. Train multiple KNN classifiers with different values of k and distance metrics.
3. Evaluate the models using accuracy, precision, recall, and F1-score.
4. Visualize the decision boundaries for different k values.
5. Plot performance metrics against different k values and distance metrics.

### Objective:
Explore how the choice of k and distance metric impacts the performance of the KNN classifier.

## Problem #3: Perceptron Learning Algorithm (2 pts)

### Task:
- Train a perceptron classifier on the Iris dataset and analyze its performance.

### Instructions:
1. Split the dataset into training and testing sets.
2. Train a perceptron classifier on the training set.
3. Evaluate the model using accuracy, precision, recall, and F1-score.
4. Visualize the decision boundaries of the perceptron.
5. Plot the convergence of the perceptron learning algorithm over iterations.

### Objective:
Gain an understanding of how the perceptron learning algorithm works and its limitations.

## Problem #4: Comparing Decision Tree, KNN, and Perceptron (3 pts)

### Task:
- Compare the performance of decision tree, KNN, and perceptron classifiers on the Iris dataset.

### Instructions:
1. Split the dataset into training and testing sets.
2. Train a decision tree, KNN, and perceptron classifier on the training set.
3. Evaluate each model using accuracy, precision, recall, and F1-score.
4. Visualize the decision boundaries for all three classifiers.
5. Plot the performance metrics for each classifier.

### Objective:
Understand the strengths and weaknesses of each classifier and how they perform on the same dataset.

## Problem #5: K-Means Clustering and Visualization (2 pts)

### Task:
- Apply k-means clustering to the Iris dataset and visualize the results.

### Instructions:
1. Normalize the Iris dataset.
2. Apply k-means clustering with k=3.
3. Visualize the cluster assignments and the centroids in 2D and 3D plots.
4. Compare the cluster assignments with the actual class labels.
5. Plot the silhouette scores to evaluate the quality of the clusters.

### Objective:
Understand how k-means clustering works and how to evaluate clustering performance.

## Problem #6: K-Medoids Clustering and Comparison with K-Means (3 pts)

### Task:
- Apply k-medoids clustering to the Iris dataset and compare it with k-means clustering.

### Instructions:
1. Normalize the Iris dataset.
2. Apply k-medoids clustering with k=3.
3. Visualize the cluster assignments and the medoids in 2D and 3D plots.
4. Compare the cluster assignments with the actual class labels.
5. Compare the performance of k-means and k-medoids using silhouette scores and other clustering metrics.

### Objective:
Understand the differences between k-means and k-medoids clustering and how they perform on the same dataset.

---

### Dataset
- **Iris Dataset**: This dataset consists of 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica).

### Requirements
- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the scripts for each problem to generate the results and visualizations.

```bash
git clone <repository-url>
cd <repository-directory>


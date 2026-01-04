# Phase 4: Classical ML - Exercises & Self-Assessment

> Deeply understand ML algorithms through implementation and application

## Strategy

Two-pass approach:
1. **First pass**: Use scikit-learn to build intuition
2. **Second pass**: Implement from scratch to understand theory

---

## 4.1 Regression - Exercises

### Must-Implement Algorithms ✓

**Linear Regression**
- [ ] Implement using normal equation
- [ ] Implement using gradient descent
- [ ] Add L1, L2, and elastic net regularization
- [ ] Visualize regularization paths

**Deliverable**: Compare your implementation vs sklearn on 3 datasets

### Concept Checks

**CC1.1**: Derive the normal equation from scratch (show all steps)

**CC1.2**: When is the normal equation better than gradient descent? When is it worse?

**CC1.3**: Explain bias-variance tradeoff using a polynomial regression example

### Checkpoint 🚦
- [ ] Can derive and implement linear regression?
- [ ] Understand regularization deeply?
- [ ] Can explain overfitting/underfitting with examples?

---

## 4.2 Classification - Exercises

### Must-Implement Algorithms ✓

**Logistic Regression**
- [ ] Derive cross-entropy loss from MLE (Phase 1 connection!)
- [ ] Implement binary logistic regression
- [ ] Extend to multi-class (one-vs-rest and softmax)
- [ ] Regularization (L1 for feature selection, L2 for generalization)

**Decision Trees**
- [ ] Implement ID3 or CART algorithm
- [ ] Information gain vs Gini impurity
- [ ] Pruning strategies
- [ ] Visualize learned trees

**SVM (using library for QP solver)**
- [ ] Understand primal and dual formulations
- [ ] Implement linear SVM
- [ ] Kernel trick (RBF, polynomial)
- [ ] Compare kernels empirically

### Coding Exercises 💻

**CE2.1**: Implement from scratch
```python
class LogisticRegression:
    def fit(self, X, y):
        # YOUR CODE HERE
        # Use gradient descent or BFGS
        pass

    def predict_proba(self, X):
        # YOUR CODE HERE
        pass

    def predict(self, X):
        # YOUR CODE HERE
        pass

class DecisionTree:
    def fit(self, X, y):
        # YOUR CODE HERE
        # Recursive tree building
        pass

    def predict(self, X):
        # YOUR CODE HERE
        pass

    def visualize(self):
        # YOUR CODE HERE
        pass
```

### Applied Problems 🎯

**AP2.1**: Binary classification shootout
- Dataset: Wisconsin Breast Cancer
- Implement: Logistic Regression, Decision Tree, SVM
- Compare: Accuracy, precision, recall, F1, ROC-AUC
- Analyze: Which works best and why?

**AP2.2**: Multi-class classification
- Dataset: Iris or MNIST
- Implement one-vs-rest and softmax
- Compare results
- Visualize decision boundaries (2D projection)

### Checkpoint 🚦
- [ ] Implemented logistic regression from scratch?
- [ ] Understand decision trees deeply?
- [ ] Can explain SVM kernel trick?
- [ ] Know when to use which algorithm?

---

## 4.3 Ensemble Methods - Exercises

### Must-Implement Algorithms ✓

**Bagging & Random Forests**
- [ ] Implement bootstrap sampling
- [ ] Build random forest from your decision trees
- [ ] Out-of-bag error estimation
- [ ] Feature importance

**Boosting**
- [ ] Implement AdaBoost
- [ ] Understand gradient boosting intuitively
- [ ] Use XGBoost/LightGBM on real problems

### Coding Exercises 💻

**CE3.1**: Build ensemble from scratch
```python
class RandomForest:
    def __init__(self, n_trees=100):
        self.trees = []
        self.n_trees = n_trees

    def fit(self, X, y):
        # YOUR CODE HERE
        # Bootstrap + random feature subsets
        pass

    def predict(self, X):
        # YOUR CODE HERE
        # Aggregate predictions
        pass
```

### Applied Problems 🎯

**AP3.1**: Kaggle competition
- Pick a tabular data competition
- Use ensemble methods
- Feature engineering
- Hyperparameter tuning
- Aim for top 25%

### Checkpoint 🚦
- [ ] Understand bagging vs boosting?
- [ ] Implemented random forest from scratch?
- [ ] Can use XGBoost effectively?

---

## 4.4 Unsupervised Learning - Exercises

### Must-Implement Algorithms ✓

**K-Means**
- [ ] Lloyd's algorithm
- [ ] K-means++ initialization
- [ ] Elbow method for choosing k
- [ ] Silhouette analysis

**PCA**
- [ ] Already done in Phase 2, but apply to real datasets
- [ ] Visualize high-dimensional data

### Coding Exercises 💻

**CE4.1**: Clustering from scratch
```python
class KMeans:
    def fit(self, X, k):
        # YOUR CODE HERE
        # Initialize centroids (k-means++)
        # Iterate: assign + update
        pass

    def predict(self, X):
        # YOUR CODE HERE
        pass

    def plot_clusters(self, X):
        # YOUR CODE HERE
        pass
```

### Applied Problems 🎯

**AP4.1**: Customer segmentation
- Load customer data
- Feature engineering
- Try k-means with different k
- Interpret clusters
- Visualize with PCA/t-SNE

**AP4.2**: Image compression with k-means
- Load color image
- Cluster pixel colors
- Reconstruct with k colors
- Compare k=4, 8, 16, 64

### Checkpoint 🚦
- [ ] Implemented k-means from scratch?
- [ ] Can choose k intelligently?
- [ ] Applied clustering to real problems?

---

## 4.5 Model Evaluation - Exercises

### Must-Master Concepts ✓

**Cross-Validation**
- [ ] Implement k-fold CV from scratch
- [ ] Understand when to use stratified CV
- [ ] Time series CV
- [ ] Nested CV for hyperparameter tuning

**Metrics**
- [ ] Implement all metrics from scratch:
  - Regression: MSE, RMSE, MAE, R², MAPE
  - Classification: accuracy, precision, recall, F1, ROC-AUC
- [ ] Understand when to use which metric

### Coding Exercises 💻

**CE5.1**: Implement CV framework
```python
def cross_validate(model, X, y, cv=5, metric='accuracy'):
    """
    Implement k-fold cross-validation from scratch
    """
    # YOUR CODE HERE
    pass

def plot_learning_curve(model, X, y):
    """
    Plot learning curves (train/val performance vs training size)
    Diagnose overfitting/underfitting
    """
    # YOUR CODE HERE
    pass
```

### Checkpoint 🚦
- [ ] Understand all evaluation metrics?
- [ ] Can implement CV from scratch?
- [ ] Know how to diagnose overfitting?

---

## Final Phase 4 Checkpoint 🎯

### Comprehensive Project: ML Algorithm Library

Build "mini-sklearn" from scratch (numpy only):

**Must include:**
1. **Algorithms**:
   - Linear/Logistic Regression (with regularization)
   - Decision Tree
   - Random Forest
   - K-Means
   - PCA

2. **API Design**:
   - Consistent `.fit()` / `.predict()` interface
   - Support for `fit_transform()` where appropriate

3. **Utilities**:
   - Train/test split
   - Cross-validation
   - Metrics (regression + classification)
   - Preprocessing (scaling, encoding)

4. **Software Engineering**:
   - Clean, modular code
   - Comprehensive unit tests (pytest)
   - Documentation (docstrings + README)
   - Type hints
   - Examples/tutorials

5. **Validation**:
   - Compare with sklearn on multiple datasets
   - Results should match (within numerical precision)

**Time estimate**: 4-6 weeks

### Alternative Project: Kaggle Competition Deep Dive

Complete a Kaggle competition with:
- Top 25% finish (or better)
- Comprehensive writeup
- Clean, reproducible code
- Novel feature engineering or ensembling

### Knowledge Self-Assessment

You're ready for Phase 5 if:

**Implementation:**
- [ ] Implemented 5+ ML algorithms from scratch
- [ ] All implementations match sklearn results
- [ ] Clean, tested, documented code

**Understanding:**
- [ ] Can derive algorithms mathematically
- [ ] Know when to use which algorithm
- [ ] Understand bias-variance tradeoff deeply
- [ ] Can debug poor model performance

**Application:**
- [ ] Completed 3+ end-to-end ML projects
- [ ] Proper evaluation methodology
- [ ] Feature engineering skills
- [ ] Hyperparameter tuning experience

**Red Flags (need more practice):**
- ❌ Can't explain why random forest reduces variance
- ❌ Don't know when to use precision vs recall
- ❌ Can't implement k-means from memory
- ❌ Don't understand overfitting vs underfitting

### Quick Self-Test (45 minutes)

1. Implement logistic regression from scratch (no libraries)
2. Derive the decision tree split criterion
3. Explain bagging vs boosting with examples
4. When would you use k-means vs hierarchical clustering?
5. Design a CV strategy for time series forecasting

All correct + comprehensive project done? → Phase 5! 🚀

---

**Remember**: The goal is deep understanding through implementation. Don't rush - master these fundamentals before moving to deep learning.

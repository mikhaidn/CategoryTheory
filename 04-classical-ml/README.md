# Phase 4: Classical Machine Learning

> Understand how ML algorithms actually work from first principles

## Prerequisites
- Phases 1-3 complete (statistics, math, Python tools)
- Comfortable with calculus, linear algebra, optimization

## Strategy

**Two-pass approach**:
1. **First pass**: Use scikit-learn to build intuition and see results
2. **Second pass**: Implement from scratch to understand theory deeply

This leverages your SWE skills - understand through implementation.

## Learning Path

### 4.1 Supervised Learning: Regression

#### Linear Regression
- [ ] Theory: ordinary least squares, MLE derivation
- [ ] From scratch: implement using normal equation and gradient descent
- [ ] Extensions: ridge, lasso, elastic net (regularization)
- [ ] Understand bias-variance tradeoff empirically

#### Polynomial & Non-linear Regression
- [ ] Feature engineering for non-linearity
- [ ] Overfitting visualization
- [ ] Regularization's role

#### Advanced Regression
- [ ] Generalized linear models (GLMs)
- [ ] Logistic regression (bridge to classification)

**Resources**:
- Book: "Elements of Statistical Learning" (chapters 2-3)
- Course: Stanford CS229 lecture notes

**Projects**:
- [ ] Implement linear regression from scratch with gradient descent
- [ ] Build a regularization comparison tool
- [ ] Predict housing prices (Kaggle dataset)

### 4.2 Supervised Learning: Classification

#### Logistic Regression
- [ ] Derive from MLE (connects to Phase 1)
- [ ] Cross-entropy loss derivation
- [ ] Multi-class: softmax regression
- [ ] Implement from scratch

#### Support Vector Machines
- [ ] Geometric intuition (maximum margin)
- [ ] Lagrange multipliers and duality (Phase 2 optimization)
- [ ] Kernel trick (maps to infinite dimensions!)
- [ ] Implementation using optimization libraries

#### Naive Bayes
- [ ] Bayes theorem application (Phase 1!)
- [ ] Conditional independence assumption
- [ ] Text classification example
- [ ] When it works and when it fails

#### Decision Trees
- [ ] Entropy and information gain (Phase 2!)
- [ ] CART algorithm
- [ ] Pruning strategies
- [ ] Implement from scratch

**Resources**:
- Book: "Pattern Recognition and Machine Learning" by Bishop
- Interactive: Seeing Theory (Bayesian inference)

**Projects**:
- [ ] Implement logistic regression from scratch
- [ ] Build a decision tree classifier from scratch
- [ ] Compare classifiers on various datasets
- [ ] Implement SVM using cvxopt (convex optimization)

### 4.3 Ensemble Methods
**Combining weak learners into strong ones**

#### Bagging
- [ ] Bootstrap aggregating
- [ ] Random forests (bagging + decision trees)
- [ ] Variance reduction
- [ ] Out-of-bag error

#### Boosting
- [ ] AdaBoost algorithm
- [ ] Gradient boosting intuition
- [ ] XGBoost, LightGBM, CatBoost (practical tools)
- [ ] Why boosting works (theory)

#### Stacking
- [ ] Meta-learners
- [ ] Cross-validation strategies

**Resources**:
- Blog: StatQuest videos on ensemble methods
- XGBoost documentation

**Projects**:
- [ ] Implement bagging from scratch
- [ ] Build a random forest from your decision tree implementation
- [ ] Kaggle competition with XGBoost
- [ ] Compare ensemble methods empirically

### 4.4 Unsupervised Learning

#### Clustering
- [ ] **K-means**
  - Lloyd's algorithm
  - Convergence guarantees
  - Choosing k (elbow method, silhouette)
  - Implement from scratch
- [ ] **Hierarchical clustering**
  - Agglomerative vs divisive
  - Dendrogram interpretation
  - Linkage criteria
- [ ] **DBSCAN**
  - Density-based clustering
  - Handling arbitrary shapes
  - No need to specify k
- [ ] **Gaussian Mixture Models**
  - EM algorithm (Bayesian connection!)
  - Soft clustering
  - Model selection with BIC/AIC

#### Dimensionality Reduction
- [ ] **PCA** (Principal Component Analysis)
  - Implement using eigendecomposition (Phase 2!)
  - Implement using SVD
  - Variance explained
  - Visualization in 2D/3D
- [ ] **t-SNE**
  - Non-linear dimensionality reduction
  - Visualization tool
  - Understanding perplexity
- [ ] **UMAP**
  - Modern alternative to t-SNE
  - Preserves more global structure

**Resources**:
- Book: "Elements of Statistical Learning" (chapter 14)
- Distill.pub: How to Use t-SNE Effectively

**Projects**:
- [ ] Implement k-means from scratch
- [ ] Implement PCA from scratch (both methods)
- [ ] Customer segmentation project
- [ ] Visualize high-dimensional data with PCA, t-SNE, UMAP

### 4.5 Model Evaluation & Selection

#### Cross-Validation
- [ ] k-fold cross-validation
- [ ] Stratified k-fold
- [ ] Leave-one-out
- [ ] Time series cross-validation
- [ ] Nested cross-validation for hyperparameter tuning

#### Metrics
- [ ] **Regression**: MSE, RMSE, MAE, R²
- [ ] **Classification**: Accuracy, precision, recall, F1, ROC-AUC
- [ ] Confusion matrix interpretation
- [ ] When to use which metric

#### Hyperparameter Tuning
- [ ] Grid search
- [ ] Random search
- [ ] Bayesian optimization (Phase 1 connection!)
- [ ] Optuna, Hyperopt tools

#### Avoiding Pitfalls
- [ ] Data leakage
- [ ] Target leakage
- [ ] Training on test set (surprisingly common!)
- [ ] Improper cross-validation

**Resources**:
- Book: "Applied Predictive Modeling" by Kuhn & Johnson
- Scikit-learn model evaluation guide

**Projects**:
- [ ] Build a model evaluation framework
- [ ] Implement cross-validation from scratch
- [ ] Hyperparameter tuning comparison

### 4.6 Feature Engineering
**Often more impactful than algorithm choice**

- [ ] Numerical features (scaling, binning, transformations)
- [ ] Categorical features (encoding strategies)
- [ ] Text features (TF-IDF, word embeddings)
- [ ] Time features (cyclical encoding, lags, windows)
- [ ] Feature interactions
- [ ] Feature selection (filter, wrapper, embedded methods)
- [ ] Automated feature engineering (Featuretools)

**Resources**:
- Book: "Feature Engineering for Machine Learning" by Zheng & Casari
- Kaggle: Feature engineering tutorials

## Key Projects

### Project 1: ML Algorithm Library
Build a scikit-learn-like library from scratch (numpy only):
- Linear regression, logistic regression
- Decision trees
- K-means, PCA
- All with fit/predict API
- Unit tests
- Documentation

**Skills**: Deep algorithm understanding, software engineering

### Project 2: End-to-End ML Competition
Pick a Kaggle competition (tabular data):
- Full EDA
- Feature engineering
- Model selection
- Ensemble methods
- Proper cross-validation
- Comprehensive report

**Skills**: Applying all Phase 4 knowledge

### Project 3: Algorithm Comparison Framework
Build a tool that:
- Takes a dataset
- Tries multiple algorithms
- Performs hyperparameter tuning
- Compares with statistical rigor
- Generates visualizations and report
- All automated with good software design

**Skills**: ML + SWE skills combined

## Self-Assessment

Before moving to deep learning:
- [ ] Can derive and implement linear/logistic regression from scratch
- [ ] Understand when to use which algorithm
- [ ] Can explain bias-variance tradeoff with examples
- [ ] Know how to properly evaluate models
- [ ] Comfortable with feature engineering
- [ ] Understand ensemble methods deeply
- [ ] Can implement k-means and PCA from scratch
- [ ] Know common pitfalls and how to avoid them

## Estimated Timeline

- **Regression**: 2 weeks
- **Classification**: 3 weeks
- **Ensemble methods**: 2 weeks
- **Unsupervised learning**: 3 weeks
- **Evaluation & selection**: 2 weeks
- **Feature engineering**: 2 weeks
- **Projects**: 4-5 weeks

**Total**: 18-21 weeks

## Next Steps

Proceed to [Phase 5: Deep Learning](../05-deep-learning/README.md) to understand neural networks.

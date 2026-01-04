# Phase 3: Python ML Tools - Exercises & Self-Assessment

> Master NumPy, SciPy, Pandas, and scikit-learn through hands-on practice

---

## 3.1 NumPy Mastery - Exercises

### Concept Checks ✓

**CC1.1**: Explain broadcasting. Give 3 examples where it's useful in ML.

**CC1.2**: What's the difference between a view and a copy in NumPy? When does each happen?

**CC1.3**: Explain memory layout (C vs Fortran order). When does it matter?

### Coding Exercises 💻

**CE1.1**: Vectorization challenges (no loops!)
```python
# 1. Compute pairwise Euclidean distances between all rows of X
def pairwise_distances(X):
    # YOUR CODE HERE (one line with broadcasting)
    pass

# 2. Implement softmax without loops
def softmax(X):
    # YOUR CODE HERE (handle numerical stability!)
    pass

# 3. Batch matrix multiplication
def batch_matmul(A, B):
    # A: (batch, n, m), B: (batch, m, k)
    # Return: (batch, n, k)
    # YOUR CODE HERE
    pass
```

**CE1.2**: 100 NumPy exercises
Complete at least 50 from: https://github.com/rougier/numpy-100

**CE1.3**: Implement neural network forward pass (numpy only)
```python
def forward_pass(X, W1, b1, W2, b2):
    """
    Two-layer network with ReLU activation
    Use only vectorized numpy operations
    """
    # YOUR CODE HERE
    pass
```

### Applied Problems 🎯

**AP1.1**: Optimize a slow function
Given a slow implementation with loops, rewrite using vectorized numpy:
- 10x+ speedup required
- Benchmark with `%timeit`

**AP1.2**: Implement k-NN from scratch (vectorized)
- Compute all pairwise distances efficiently
- Find k nearest neighbors
- Make predictions
- Test on Iris dataset

### Checkpoint 🚦
- [ ] Can write vectorized code without loops?
- [ ] Understand broadcasting deeply?
- [ ] Can profile and optimize numpy code?

---

## 3.2 SciPy - Exercises

### Coding Exercises 💻

**CE2.1**: Use scipy.optimize
```python
# Fit a complex function to data
# Compare different optimization methods
# Visualize results
```

**CE2.2**: Statistical tests
```python
# Implement 5 statistical tests using scipy.stats
# t-test, chi-square, ANOVA, etc.
# Compare with your Phase 1 implementations
```

**CE2.3**: Solve a system of ODEs
```python
# SIR model (epidemiology) or
# Lotka-Volterra (predator-prey)
# Plot trajectories
```

### Checkpoint 🚦
- [ ] Can use scipy.optimize for complex problems?
- [ ] Comfortable with scipy.stats?
- [ ] Can solve numerical problems (ODEs, integration)?

---

## 3.3 Pandas - Exercises

### Coding Exercises 💻

**CE3.1**: Data cleaning challenge
```python
# Given a messy dataset:
# - Handle missing values (multiple strategies)
# - Fix data types
# - Remove duplicates
# - Detect outliers
# - Create clean dataset
```

**CE3.2**: GroupBy mastery
```python
# Perform complex aggregations
# Custom aggregation functions
# Multi-level grouping
# Transform vs aggregate vs filter
```

**CE3.3**: Time series analysis
```python
# Load stock price data
# Resample to different frequencies
# Compute rolling statistics
# Handle date/time properly
```

### Applied Problems 🎯

**AP3.1**: Complete EDA on Kaggle dataset
- Load data
- Summary statistics
- Visualizations
- Feature engineering
- Document insights

### Checkpoint 🚦
- [ ] Proficient with pandas operations?
- [ ] Can perform complex groupby operations?
- [ ] Comfortable with time series?

---

## 3.4 Scikit-learn - Exercises

### Coding Exercises 💻

**CE4.1**: Build a complete ML pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Create pipeline with:
# - Preprocessing
# - Feature engineering
# - Model
# - Cross-validation
# - Hyperparameter tuning
```

**CE4.2**: Implement a custom transformer
```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # YOUR CODE HERE
        return self

    def transform(self, X):
        # YOUR CODE HERE
        return X_transformed
```

**CE4.3**: Cross-validation strategies
```python
# Compare different CV strategies:
# - KFold
# - StratifiedKFold
# - TimeSeriesSplit
# - Leave-One-Out
# When to use each?
```

### Applied Problems 🎯

**AP4.1**: End-to-end ML project
- Pick a Kaggle dataset
- Full pipeline (preprocessing, model, evaluation)
- Compare 5+ algorithms
- Hyperparameter tuning
- Final model evaluation
- Clean, documented code

### Checkpoint 🚦
- [ ] Fluent with scikit-learn API?
- [ ] Can build complete pipelines?
- [ ] Understand cross-validation deeply?

---

## Final Phase 3 Checkpoint 🎯

### Comprehensive Project: End-to-End ML Pipeline

Build a production-quality ML system:

**Requirements:**
1. Data loading and validation (pandas)
2. EDA with visualizations (matplotlib/seaborn)
3. Feature engineering pipeline (scikit-learn)
4. Model comparison framework (5+ algorithms)
5. Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
6. Cross-validation
7. Final evaluation with multiple metrics
8. Comprehensive notebook + clean .py modules
9. Unit tests for data processing
10. Documentation

**Dataset**: Pick from Kaggle (tabular data, classification or regression)

**Success criteria:**
- Top 25% performance on leaderboard (if competition is active)
- Clean, modular code
- Proper train/validation/test split
- No data leakage
- Well-documented

### Knowledge Self-Assessment

You're ready for Phase 4 if:
- [ ] Write vectorized numpy code fluently
- [ ] Use pandas for complex data manipulation
- [ ] Build scikit-learn pipelines with custom components
- [ ] Completed at least 2 end-to-end ML projects
- [ ] Code is clean and well-tested

### Quick Self-Test (30 minutes)

1. Implement softmax in one vectorized line
2. GroupBy + custom aggregation in pandas
3. Build a scikit-learn pipeline from memory
4. Implement a custom sklearn transformer
5. Set up proper cross-validation for time series

All correct? → Move to Phase 4! 🚀

---

**Note**: The comprehensive project is the main checkpoint. If you can build a high-quality end-to-end system, you've mastered the tools.

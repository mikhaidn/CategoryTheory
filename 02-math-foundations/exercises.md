# Phase 2: Math Foundations - Exercises & Self-Assessment

> Verify your understanding of linear algebra, optimization, and information theory

## How to Use This

Complete at least 80% of exercises in each section before moving on. Focus on implementation and derivation - this phase is about deeply understanding the math that powers ML.

---

## 2.1 Linear Algebra - Exercises

### Concept Checks ✓

**CC1.1**: What's the geometric interpretation of eigenvalues and eigenvectors? Draw it.

**CC1.2**: Explain the relationship between eigendecomposition and SVD. When can you use each?

**CC1.3**: What does it mean for a matrix to be positive definite? Why do covariance matrices have this property?

**CC1.4**: Explain the chain rule for matrix derivatives. How does it apply to backpropagation?

**CC1.5**: What's the difference between the L1 and L2 norm? How do they affect optimization?

### Coding Exercises 💻

**CE1.1**: Implement eigendecomposition-based PCA
```python
def pca_eigen(X, n_components):
    """
    Implement PCA using eigendecomposition

    Steps:
    1. Center the data
    2. Compute covariance matrix
    3. Find eigenvectors/eigenvalues
    4. Project onto top k eigenvectors

    Returns:
        X_transformed, explained_variance_ratio
    """
    # YOUR CODE HERE
    pass
```

**CE1.2**: Implement SVD-based PCA
```python
def pca_svd(X, n_components):
    """
    Implement PCA using SVD
    Compare results with eigendecomposition method
    Which is more numerically stable?
    """
    # YOUR CODE HERE
    pass
```

**CE1.3**: Matrix calculus practice
```python
def gradient_linear_regression(X, y, w):
    """
    Compute gradient of MSE loss for linear regression
    Using matrix calculus

    Loss = (1/2n) * ||Xw - y||^2

    Derive ∇_w Loss on paper first!

    Returns:
        gradient vector
    """
    # YOUR CODE HERE
    pass
```

**CE1.4**: Implement power iteration
```python
def power_iteration(A, num_iterations=100):
    """
    Find largest eigenvalue and eigenvector using power iteration

    This is how PCA is computed efficiently for large matrices
    """
    # YOUR CODE HERE
    pass
```

### Applied Problems 🎯

**AP1.1**: Image Compression with SVD
- Load a grayscale image (e.g., from matplotlib.image)
- Perform SVD
- Reconstruct using k=5, 10, 50, 100 singular values
- Plot original vs reconstructions
- What's the compression ratio?

**AP1.2**: Matrix Calculus Derivations
Derive the following gradients:
1. ∇_w (w^T x) = ?
2. ∇_w (w^T A w) = ?
3. ∇_W (||Xw - y||^2) = ?
4. ∇_W (trace(W^T A W)) = ?

**AP1.3**: Implement collaborative filtering
Use SVD for matrix factorization on MovieLens dataset:
- User-item ratings matrix (sparse)
- Factorize into user and item embeddings
- Predict missing ratings
- Evaluate RMSE

### Checkpoint 🚦

**Can you:**
- [ ] Implement PCA from scratch (both methods)?
- [ ] Derive gradients using matrix calculus?
- [ ] Explain eigendecomposition vs SVD?
- [ ] Apply SVD to real problems?

---

## 2.2 Optimization - Exercises

### Concept Checks ✓

**CC2.1**: What's a convex function? Why do we care about convexity in ML?

**CC2.2**: Explain the difference between local and global minima. Which can gradient descent find?

**CC2.3**: What are the KKT conditions? How do they relate to SVMs?

**CC2.4**: Why does momentum help gradient descent? Explain intuitively and mathematically.

**CC2.5**: Explain the bias-variance tradeoff in terms of optimization.

### Coding Exercises 💻

**CE2.1**: Implement gradient descent variants
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iter=1000):
    """Vanilla gradient descent"""
    # YOUR CODE HERE
    pass

def sgd_with_momentum(f, grad_f, x0, learning_rate=0.01, momentum=0.9, max_iter=1000):
    """SGD with momentum"""
    # YOUR CODE HERE
    pass

def adam(f, grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, max_iter=1000):
    """Adam optimizer"""
    # YOUR CODE HERE
    pass
```

**CE2.2**: Visualize optimization trajectories
```python
def visualize_optimizer(optimizer, function, start_point):
    """
    Create a 2D visualization of optimizer path

    - Plot the loss landscape (contour plot)
    - Overlay optimizer trajectory
    - Compare different optimizers
    """
    # YOUR CODE HERE
    pass

# Test on: Rosenbrock function, Beale function, sphere function
```

**CE2.3**: Implement line search
```python
def backtracking_line_search(f, grad_f, x, p, alpha=1.0, rho=0.5, c=1e-4):
    """
    Backtracking line search for step size selection

    Used in gradient descent for adaptive learning rates
    """
    # YOUR CODE HERE
    pass
```

**CE2.4**: Constrained optimization with Lagrange multipliers
```python
def lagrange_multipliers_solve(objective, constraints):
    """
    Solve a simple constrained optimization problem

    Example: minimize x^2 + y^2 subject to x + y = 1
    """
    # YOUR CODE HERE
    # Set up and solve the KKT conditions
    pass
```

### Applied Problems 🎯

**AP2.1**: Linear Regression with Different Optimizers
- Generate synthetic data: y = 3x + 2 + noise
- Implement linear regression using:
  - Closed-form solution (normal equation)
  - Gradient descent
  - SGD
  - SGD with momentum
  - Adam
- Compare convergence speed and final results
- Visualize loss curves

**AP2.2**: Logistic Regression from Scratch
- Implement logistic regression using gradient descent
- Use cross-entropy loss
- Derive the gradient analytically
- Test on a binary classification dataset
- Plot decision boundary

**AP2.3**: Optimizer Comparison Study
Test all optimizers on various functions:
- Convex: quadratic, bowl-shaped
- Non-convex: Rosenbrock, Beale
- Measure: convergence speed, final loss, stability
- Create visualizations and report

### Checkpoint 🚦

**Can you:**
- [ ] Implement common optimizers from scratch?
- [ ] Explain and recognize convex vs non-convex problems?
- [ ] Apply Lagrange multipliers to constrained optimization?
- [ ] Debug optimization issues (learning rate, convergence)?

---

## 2.3 Information Theory - Exercises

### Concept Checks ✓

**CC3.1**: What is entropy? What does high entropy mean?

**CC3.2**: Explain cross-entropy. Why is it used as a loss function?

**CC3.3**: What is KL divergence? Is it symmetric?

**CC3.4**: How does mutual information relate to correlation?

### Coding Exercises 💻

**CE3.1**: Implement information theory metrics
```python
def entropy(p):
    """Compute entropy H(p) of a discrete distribution"""
    # YOUR CODE HERE
    pass

def cross_entropy(p, q):
    """Compute cross-entropy H(p, q)"""
    # YOUR CODE HERE
    pass

def kl_divergence(p, q):
    """Compute KL(p || q)"""
    # YOUR CODE HERE
    pass

def mutual_information(p_xy):
    """Compute mutual information I(X; Y)"""
    # YOUR CODE HERE
    pass
```

**CE3.2**: Derive cross-entropy loss
```python
# Show mathematically that minimizing cross-entropy loss
# is equivalent to maximizing likelihood for classification

# Implement and verify:
def cross_entropy_loss(y_true, y_pred):
    """Binary cross-entropy loss"""
    # YOUR CODE HERE
    pass

def nll_loss(y_true, y_pred):
    """Negative log-likelihood"""
    # YOUR CODE HERE
    pass

# Verify they're equivalent (up to constants)
```

### Applied Problems 🎯

**AP3.1**: Information Theory of Language
- Take a text corpus
- Compute character-level entropy
- Compute word-level entropy
- What does this tell you about compressibility?

**AP3.2**: Feature Selection with Mutual Information
- Generate data with relevant and irrelevant features
- Compute mutual information between each feature and target
- Select top-k features
- Compare with correlation-based selection

### Checkpoint 🚦

**Can you:**
- [ ] Compute and interpret entropy, cross-entropy, KL divergence?
- [ ] Explain why cross-entropy is the right loss for classification?
- [ ] Connect information theory to ML concepts?

---

## Final Phase 2 Checkpoint 🎯

### Comprehensive Project: Optimizer Comparison Framework

Build a tool that:

1. **Implements multiple optimizers** (GD, momentum, Adam, etc.)
2. **Tests on various functions** (convex and non-convex)
3. **Visualizes trajectories** (2D/3D plots, loss curves)
4. **Adaptive learning rates** (line search, decay schedules)
5. **Clean software design** (extensible, well-tested)

**Plus**: Implement a simple ML algorithm (logistic regression or neural network) using your optimizer framework.

### Knowledge Self-Assessment

You're ready for Phase 3 if:

**Linear Algebra:**
- [ ] Can implement PCA from scratch (both ways)
- [ ] Understand matrix calculus and can derive gradients
- [ ] Know when to use eigendecomposition vs SVD
- [ ] Comfortable with matrix operations in numpy

**Optimization:**
- [ ] Implemented common optimizers from scratch
- [ ] Understand convexity and its implications
- [ ] Can debug convergence issues
- [ ] Know how to select learning rates

**Information Theory:**
- [ ] Understand entropy, cross-entropy, KL divergence
- [ ] Can derive loss functions from first principles
- [ ] See connections to ML

**Critical Understanding:**
- [ ] Can derive backprop for a simple network using matrix calculus
- [ ] Understand why Adam works better than SGD (usually)
- [ ] Can explain regularization in terms of optimization landscape

### Quick Self-Test (30 minutes)

1. Implement PCA using SVD in 15 lines of numpy
2. Derive the gradient of logistic loss
3. Implement Adam optimizer from memory
4. Explain why cross-entropy loss for classification
5. What's the relationship between SVD and eigendecomposition?

All correct? → Move to Phase 3! 🚀

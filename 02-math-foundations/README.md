# Phase 2: Mathematical Foundations for ML

> Strengthen and refresh mathematical tools needed for machine learning

## Prerequisites
- Comfortable with calculus (✅ you have this)
- Undergraduate linear algebra (✅ you have this)
- Phase 1 statistics complete or in progress

## Learning Path

### 2.1 Linear Algebra Refresher & Extension
**Building on undergrad linear algebra**

#### Core Refresher
- [ ] Vector spaces, subspaces, basis, dimension
- [ ] Linear transformations and matrices
- [ ] Matrix operations and properties
- [ ] Determinants and trace

#### Deep Dive for ML
- [ ] **Eigenvalues and eigenvectors** (critical for PCA, spectral methods)
  - Geometric interpretation
  - Diagonalization
  - Spectral theorem
- [ ] **Singular Value Decomposition (SVD)** (foundation of many ML algorithms)
  - Relation to eigendecomposition
  - Low-rank approximations
  - Applications: PCA, recommender systems
- [ ] **Matrix calculus** (needed for deriving gradient descent)
  - Gradients of scalar functions
  - Jacobians and Hessians
  - Chain rule for matrices
  - Common derivatives (quadratic forms, etc.)
- [ ] Norms and distance metrics
- [ ] Positive definite matrices (covariance matrices)
- [ ] Matrix factorizations (LU, QR, Cholesky)

**Key Insights for ML**:
- PCA = finding eigenvectors of covariance matrix
- SVD is the basis of collaborative filtering and topic modeling
- Matrix calculus is how backpropagation actually works
- Understanding matrix operations speeds up implementation

**Resources**:
- Book: "Introduction to Linear Algebra" by Gilbert Strang (refresher + depth)
- Video: 3Blue1Brown Essence of Linear Algebra series (geometric intuition)
- Interactive: Eigenvector visualization tools
- Book: "Matrix Calculus for Deep Learning" by Parr & Howard

**Exercises**:
- [ ] Implement PCA from scratch using eigendecomposition
- [ ] Implement PCA using SVD and compare
- [ ] Derive gradient descent update for linear regression using matrix calculus
- [ ] Implement basic matrix operations in numpy efficiently

### 2.2 Optimization Theory
**New material - critical for training ML models**

#### Fundamentals
- [ ] Convex sets and convex functions
- [ ] Local vs global minima
- [ ] Gradient, directional derivative
- [ ] Necessary and sufficient conditions for optimality (KKT conditions)

#### Optimization Algorithms
- [ ] **Gradient Descent** (the workhorse of ML)
  - Derivation and convergence analysis
  - Step size selection
  - Stochastic gradient descent (SGD)
  - Mini-batch gradient descent
- [ ] **Advanced Optimizers** (what modern frameworks use)
  - Momentum
  - AdaGrad, RMSprop
  - Adam and variants
- [ ] **Constrained Optimization**
  - Lagrange multipliers (foundation of SVMs)
  - Duality
- [ ] **Second-order Methods**
  - Newton's method
  - Quasi-Newton (BFGS)

**Key Insights for ML**:
- All neural network training is optimization
- Understanding convergence helps debug training
- Convexity determines whether we can find global optima
- SVMs are derived from constrained optimization

**Resources**:
- Book: "Convex Optimization" by Boyd & Vandenberghe (chapters 1-5, free PDF)
- Course: Stanford CS229 optimization notes
- Interactive: Distill.pub optimizer visualizations

**Exercises**:
- [ ] Implement gradient descent from scratch
- [ ] Implement SGD, momentum, Adam
- [ ] Visualize optimizer behavior on simple 2D functions
- [ ] Derive SVM dual problem using Lagrange multipliers

### 2.3 Information Theory Basics
**Connects probability, statistics, and ML**

- [ ] Entropy and cross-entropy
- [ ] KL divergence
- [ ] Mutual information
- [ ] Information theoretic perspective on ML

**Key Insights for ML**:
- Cross-entropy loss = KL divergence + constant
- Mutual information used in feature selection
- VAEs optimize evidence lower bound (ELBO), derived from KL divergence
- Understanding why certain loss functions work

**Resources**:
- Book: "Elements of Information Theory" by Cover & Thomas (chapter 2)
- Blog: Christopher Olah on information theory and ML

**Exercises**:
- [ ] Implement entropy calculation for discrete distributions
- [ ] Derive cross-entropy loss from MLE + KL divergence
- [ ] Visualize KL divergence between distributions

### 2.4 Probability Distributions (Advanced)
**Building on Phase 1**

- [ ] Exponential family (unifying framework)
- [ ] Gaussian processes intuition
- [ ] Mixture models
- [ ] Graphical models basics (directed/undirected)

**Resources**:
- Book: "Pattern Recognition and Machine Learning" by Bishop (chapter 2)

## Projects

### Project 1: Optimizer Comparison Framework
Build a framework that:
- Implements multiple optimizers (SGD, momentum, Adam, etc.)
- Visualizes optimization trajectories on various functions
- Compares convergence rates
- Includes adaptive learning rate schedules

### Project 2: Matrix Factorization Recommender System
- Implement collaborative filtering using SVD
- Compare with gradient descent-based matrix factorization
- Evaluate on MovieLens dataset
- Visualize latent factors

### Project 3: From-Scratch ML with Math
Implement these algorithms using only numpy, deriving all math:
- Linear regression (with matrix calculus derivation)
- Logistic regression (with derivation)
- Principal Component Analysis (both eigen and SVD methods)
- K-means (with convergence proof understanding)

## Self-Assessment

Before moving to Phase 3, you should be able to:
- [ ] Explain eigendecomposition vs SVD and when to use each
- [ ] Derive backpropagation for a simple neural network using matrix calculus
- [ ] Implement gradient descent variants from scratch
- [ ] Explain why cross-entropy is the right loss for classification
- [ ] Recognize convex vs non-convex optimization problems
- [ ] Use matrix operations efficiently in numpy

## Estimated Timeline

- **Linear algebra refresher + extensions**: 2-3 weeks
- **Optimization theory**: 3-4 weeks (new material, critical for ML)
- **Information theory**: 1-2 weeks
- **Projects**: 2-3 weeks

**Total**: 8-12 weeks

## Next Steps

Proceed to [Phase 3: Python ML Tools](../03-python-tools/README.md) to get hands-on with the ecosystem.

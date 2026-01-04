# Phase 3: Python ML Tools

> Refresh numpy/scipy skills and master the Python ML ecosystem

## Prerequisites
- Professional programming experience (✅ you have this)
- Phase 1 & 2 complete or in progress
- Python basics (assumed)

## Learning Path

### 3.1 NumPy Deep Dive
**Refresh and go deeper than your 400-level numerical methods course**

#### Core NumPy
- [ ] Array fundamentals (ndarray, dtype, shape, strides)
- [ ] **Broadcasting** (critical for efficient ML code)
- [ ] Indexing, slicing, fancy indexing
- [ ] Universal functions (ufuncs)
- [ ] Array manipulation (reshape, transpose, concatenate)
- [ ] Linear algebra with `numpy.linalg`
- [ ] Random number generation (`numpy.random`)

#### Performance & Best Practices
- [ ] Vectorization techniques (avoid loops)
- [ ] Memory layout (C vs Fortran order)
- [ ] In-place operations
- [ ] Memory-mapped arrays for large datasets
- [ ] Profiling numpy code

**Resources**:
- Book: "From Python to Numpy" by Nicolas Rougier (free online)
- Tutorial: NumPy official tutorial (updated for modern numpy)
- Practice: 100 NumPy exercises (github.com/rougier/numpy-100)

**Exercises**:
- [ ] Reimplement all Phase 2 projects with optimized numpy
- [ ] Benchmark vectorized vs loop implementations
- [ ] Implement matrix multiplication without loops
- [ ] Build a mini neural network using only numpy

### 3.2 SciPy for Scientific Computing
**Building on your numerical methods background**

- [ ] **Optimization** (`scipy.optimize`)
  - Finding minima/maxima
  - Curve fitting
  - Root finding
- [ ] **Linear algebra** (`scipy.linalg`) - extensions beyond numpy
  - Sparse matrices
  - Specialized decompositions
- [ ] **Statistics** (`scipy.stats`)
  - Distribution functions
  - Statistical tests
  - Integration with Phase 1 learning
- [ ] **Integration and ODEs** (`scipy.integrate`)
  - Numerical integration
  - Solving differential equations (refresh those rusty DE skills!)
- [ ] **Signal processing** (`scipy.signal`) - useful for time series
- [ ] **Sparse matrices** (`scipy.sparse`) - essential for large-scale ML

**Resources**:
- Docs: SciPy official documentation and tutorials
- Book: "Elegant SciPy" by Juan Nunez-Iglesias et al.

**Exercises**:
- [ ] Use `scipy.optimize` to fit a complex function to data
- [ ] Implement a statistical test from Phase 1 using scipy.stats
- [ ] Solve a system of ODEs (refresh those DE skills!)
- [ ] Compare dense vs sparse matrix operations

### 3.3 Pandas for Data Manipulation
**Essential for real-world ML workflows**

#### Core Pandas
- [ ] Series and DataFrame fundamentals
- [ ] Indexing and selection (loc, iloc, boolean indexing)
- [ ] Data cleaning (handling missing values, duplicates)
- [ ] GroupBy operations (split-apply-combine)
- [ ] Merge, join, concatenate operations
- [ ] Time series functionality
- [ ] Categorical data

#### ML-Specific Workflows
- [ ] Feature engineering with pandas
- [ ] Train/test splits
- [ ] Exploratory data analysis (EDA) patterns
- [ ] Integration with scikit-learn

**Resources**:
- Book: "Python for Data Analysis" by Wes McKinney (pandas creator)
- Tutorial: Kaggle's pandas course

**Exercises**:
- [ ] Complete a full EDA on Kaggle dataset
- [ ] Build a feature engineering pipeline
- [ ] Implement custom aggregation functions

### 3.4 Visualization: Matplotlib & Seaborn
**Communicate your findings**

#### Matplotlib
- [ ] Figure and axes architecture
- [ ] Common plot types (line, scatter, bar, histogram)
- [ ] Customization (labels, legends, styles)
- [ ] Subplots and layouts
- [ ] Saving high-quality figures

#### Seaborn
- [ ] Statistical visualizations
- [ ] Distribution plots
- [ ] Categorical plots
- [ ] Heatmaps and cluster maps
- [ ] Pair plots and joint plots

**Exercises**:
- [ ] Recreate famous statistical visualizations
- [ ] Build a visualization library for common ML tasks
- [ ] Create publication-quality figures

### 3.5 Scikit-learn Ecosystem
**The practical ML workhorse**

#### Core Concepts
- [ ] Estimator API (fit, predict, transform)
- [ ] Pipelines (chaining transformations)
- [ ] Cross-validation utilities
- [ ] Metrics and evaluation
- [ ] Feature preprocessing and scaling

#### Algorithms (high-level first, then dive deep)
- [ ] Linear models
- [ ] Tree-based models
- [ ] Ensemble methods
- [ ] Clustering
- [ ] Dimensionality reduction

**Strategy**: Use scikit-learn to get intuition, then implement from scratch to understand theory (Phase 4)

**Resources**:
- Docs: Scikit-learn official documentation (excellent!)
- Tutorial: Scikit-learn official tutorials
- Book: "Hands-On Machine Learning" by Aurélien Géron (chapters 2-4)

**Exercises**:
- [ ] Build an end-to-end ML pipeline with Pipeline API
- [ ] Implement custom transformers
- [ ] Compare cross-validation strategies

### 3.6 Jupyter Ecosystem
**Interactive development and documentation**

- [ ] Jupyter notebooks best practices
- [ ] JupyterLab advanced features
- [ ] Reproducible notebooks
- [ ] Converting notebooks to scripts/reports
- [ ] Jupyter extensions for ML

## Projects

### Project 1: NumPy Neural Network Library
Build a mini deep learning library using only numpy:
- Forward pass with various activation functions
- Backpropagation (applying your matrix calculus from Phase 2)
- Training loop with different optimizers
- Modular design (inspired by PyTorch/Keras)

### Project 2: End-to-End ML Pipeline
On a Kaggle dataset:
- EDA with pandas and visualization
- Feature engineering
- Model selection with scikit-learn
- Hyperparameter tuning
- Cross-validation
- Final evaluation and visualization
- All in a well-documented Jupyter notebook

### Project 3: Statistical Analysis Tool
Build a command-line tool (leverage your SWE skills):
- Takes CSV input
- Performs statistical tests (Phase 1 knowledge)
- Generates visualizations
- Outputs report
- Uses numpy/scipy/pandas/matplotlib
- Proper software engineering (tests, CLI, docs)

## Performance Optimization

Since you're a professional SWE:
- [ ] Profile Python code (cProfile, line_profiler)
- [ ] Understand when to use Numba for JIT compilation
- [ ] Explore Cython for critical paths
- [ ] Consider JAX for high-performance numerical computing

## Self-Assessment

Before moving to Phase 4, you should be able to:
- [ ] Write efficient vectorized numpy code without loops
- [ ] Build a complete data processing pipeline in pandas
- [ ] Implement basic ML algorithms in numpy from scratch
- [ ] Use scikit-learn's API fluently
- [ ] Create informative visualizations
- [ ] Profile and optimize numerical Python code

## Estimated Timeline

- **NumPy mastery**: 2 weeks (refresher with depth)
- **SciPy & Pandas**: 2 weeks
- **Visualization**: 1 week
- **Scikit-learn**: 2 weeks
- **Projects**: 2-3 weeks

**Total**: 9-10 weeks

## Next Steps

Move to [Phase 4: Classical ML](../04-classical-ml/README.md) to understand ML algorithms deeply.

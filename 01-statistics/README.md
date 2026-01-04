# Phase 1: Statistical Foundations

> Bridge from descriptive to inferential and Bayesian statistics

## Why This Matters for AI/ML

Modern ML is fundamentally about learning probability distributions from data. Understanding:
- **Inferential statistics** lets you evaluate model performance, conduct A/B tests, and quantify uncertainty
- **Bayesian methods** are core to probabilistic ML, variational inference, and understanding modern techniques like Bayesian optimization

## Learning Path

### 1.1 Probability Theory Deep Dive
**Building on your calc foundation**

- [ ] Review: Random variables, distributions, expectation
- [ ] Joint, marginal, and conditional distributions
- [ ] Law of large numbers and central limit theorem
- [ ] Moment generating functions
- [ ] Common distributions (Bernoulli, Binomial, Poisson, Normal, Exponential, Beta, Dirichlet)

**Key Insights for ML**:
- Most loss functions are derived from probability distributions
- Understanding distributions explains why certain models work

**Resources**:
- Book: "All of Statistics" by Wasserman (chapters 1-4)
- Practice: Implement distributions from scratch in numpy

### 1.2 Inferential Statistics
**New territory - focus here first**

- [ ] Point estimation (MLE, method of moments)
- [ ] Properties of estimators (bias, variance, consistency, efficiency)
- [ ] Confidence intervals (construction and interpretation)
- [ ] Hypothesis testing (p-values, Type I/II errors, power)
- [ ] Multiple testing and corrections (Bonferroni, FDR)
- [ ] Bootstrap and resampling methods

**Key Insights for ML**:
- MLE is the foundation of most ML training algorithms
- Bootstrap methods are used for model evaluation and uncertainty quantification
- Understanding p-values helps avoid common pitfalls in model evaluation

**Resources**:
- Book: "All of Statistics" by Wasserman (chapters 5-10)
- Interactive: Seeing Theory (https://seeing-theory.brown.edu/)
- Practice problems: Khan Academy Inferential Statistics

**Exercises**:
- [ ] Implement MLE from scratch for common distributions
- [ ] Build a bootstrap confidence interval calculator
- [ ] Design and analyze an A/B test

### 1.3 Bayesian Statistics
**The paradigm shift**

- [ ] Bayes' theorem and Bayesian reasoning
- [ ] Prior, likelihood, posterior
- [ ] Conjugate priors
- [ ] Bayesian inference vs frequentist inference
- [ ] Maximum a posteriori (MAP) estimation
- [ ] Markov Chain Monte Carlo (MCMC) basics
- [ ] Variational inference intuition

**Key Insights for ML**:
- Regularization = Bayesian priors
- Dropout = variational inference approximation
- Bayesian optimization for hyperparameter tuning
- Understanding uncertainty in predictions

**Resources**:
- Book: "Bayesian Methods for Hackers" (free online, Python-based)
- Book: "Think Bayes" by Allen Downey (practical approach)
- Video: 3Blue1Brown Bayes Theorem
- Library: PyMC for practical Bayesian modeling

**Exercises**:
- [ ] Implement Bayesian parameter estimation for a simple model
- [ ] Compare MLE vs MAP on a small dataset
- [ ] Build a naive Bayes classifier from scratch
- [ ] Explore MCMC sampling with PyMC

### 1.4 Statistical Learning Theory Basics
**Bridge to ML**

- [ ] Bias-variance decomposition
- [ ] Overfitting and underfitting
- [ ] Cross-validation
- [ ] Information criteria (AIC, BIC)
- [ ] Regularization from a statistical perspective

## Projects

### Project 1: A/B Testing Framework
Build a complete A/B testing system with:
- Sample size calculation
- Multiple testing correction
- Bayesian credible intervals
- Visualization dashboard

### Project 2: Statistical Inference Library
Create a numpy-based library implementing:
- Common distribution PDFs/CDFs
- MLE estimators
- Bootstrap methods
- Hypothesis tests

### Project 3: Bayesian Model Comparison
Implement and compare:
- Frequentist linear regression
- Bayesian linear regression with conjugate priors
- Bayesian linear regression with MCMC
- Visualize posterior distributions

## Exercises & Self-Assessment

**See [exercises.md](./exercises.md) for detailed practice problems, coding exercises, and checkpoints.**

The exercises file includes:
- Concept checks for each topic
- Coding exercises with implementation challenges
- Applied problems with real-world scenarios
- Comprehensive checkpoint project (A/B Testing Framework)
- Self-assessment criteria

Before moving to Phase 2, you should be able to:
- [ ] Explain when and why to use Bayesian vs frequentist approaches
- [ ] Derive MLE for a simple model from first principles
- [ ] Interpret confidence intervals and credible intervals correctly
- [ ] Implement bootstrap from scratch
- [ ] Understand the connection between regularization and priors
- [ ] Explain bias-variance tradeoff with mathematical rigor

## Estimated Timeline

- **Probability refresher**: 1-2 weeks (since you have calc background)
- **Inferential statistics**: 3-4 weeks (new material, practice needed)
- **Bayesian statistics**: 3-4 weeks (paradigm shift, lots of practice)
- **Projects**: 2-3 weeks

**Total**: 9-13 weeks of focused study

## Next Steps

Once comfortable here, move to [Phase 2: Math Foundations](../02-math-foundations/README.md) to refresh linear algebra and learn optimization theory.

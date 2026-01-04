# Phase 1: Statistics - Exercises & Self-Assessment

> Concrete exercises to verify understanding before moving to Phase 2

## How to Use This

Each section has:
- **Concept Checks**: Quick questions to test understanding
- **Coding Exercises**: Implement concepts from scratch
- **Applied Problems**: Real-world scenarios
- **Checkpoint**: Must-pass assessment before moving on

**Grading yourself**: If you can't explain the concept clearly or solve 80%+ of exercises, review that section.

---

## 1.1 Probability Theory - Exercises

### Concept Checks ✓

**CC1.1**: Explain the difference between a probability mass function (PMF) and a probability density function (PDF). Why can a PDF have values > 1?

**CC1.2**: You flip a fair coin 100 times. What's the expected number of heads? What's the variance? What distribution describes this?

**CC1.3**: Given random variable X ~ N(5, 9), what are μ and σ? What's P(X > 8) approximately?

**CC1.4**: If X and Y are independent random variables, is E[XY] = E[X]E[Y]? What about if they're dependent?

**CC1.5**: Explain the Central Limit Theorem in your own words. Why does it matter for ML?

### Coding Exercises 💻

**CE1.1**: Implement from scratch (no scipy.stats)
```python
def bernoulli_pmf(k, p):
    """Compute P(X = k) for X ~ Bernoulli(p)"""
    # YOUR CODE HERE
    pass

def binomial_pmf(k, n, p):
    """Compute P(X = k) for X ~ Binomial(n, p)"""
    # YOUR CODE HERE
    pass

def normal_pdf(x, mu, sigma):
    """Compute PDF at x for X ~ N(mu, sigma^2)"""
    # YOUR CODE HERE
    pass
```

**CE1.2**: Visualize the CLT
```python
# Sample from a heavily skewed distribution (e.g., exponential)
# Take sample means of size n = 5, 10, 30, 100
# Plot histograms showing convergence to normal
# YOUR CODE HERE
```

**CE1.3**: Implement a sampler
```python
def sample_from_distribution(pmf_func, values, n_samples):
    """
    Sample from a discrete distribution defined by pmf_func
    using inverse transform sampling or rejection sampling
    """
    # YOUR CODE HERE
    pass
```

### Applied Problems 🎯

**AP1.1**: You're analyzing click-through rates on a website. Clicks follow a Binomial distribution. If you have 1000 visitors and 5% click rate:
- What's the expected number of clicks?
- What's the probability of getting exactly 50 clicks?
- What's the probability of getting 60 or more clicks?

**AP1.2**: Response times for your API follow an exponential distribution with mean 200ms. What's the probability a request takes longer than 500ms?

### Checkpoint 🚦

**Can you confidently:**
- [ ] Explain and compute expectations and variances?
- [ ] Work with common distributions (Bernoulli, Binomial, Normal, Exponential)?
- [ ] Understand and apply the CLT?
- [ ] Implement basic probability functions in code?

---

## 1.2 Inferential Statistics - Exercises

### Concept Checks ✓

**CC2.1**: What's the difference between a parameter and a statistic? Give examples.

**CC2.2**: Explain Maximum Likelihood Estimation. Why is it called "maximum" and "likelihood"?

**CC2.3**: What does a 95% confidence interval mean? (Be precise - this is commonly misunderstood!)

**CC2.4**: Explain the difference between Type I and Type II errors. Which is worse depends on what?

**CC2.5**: What's a p-value? What does p < 0.05 actually tell you? What does it NOT tell you?

**CC2.6**: Why do we need multiple testing corrections? Explain the multiple testing problem.

### Coding Exercises 💻

**CE2.1**: Implement MLE for a Bernoulli distribution
```python
def mle_bernoulli(data):
    """
    Given a list of 0s and 1s, estimate p using MLE

    Args:
        data: list of 0s and 1s
    Returns:
        p_hat: MLE estimate of p
    """
    # YOUR CODE HERE
    # Derive this mathematically first!
    pass
```

**CE2.2**: Implement MLE for a Normal distribution
```python
def mle_normal(data):
    """
    Estimate mu and sigma using MLE

    Returns:
        mu_hat, sigma_hat
    """
    # YOUR CODE HERE
    # Derive the formulas first!
    pass
```

**CE2.3**: Bootstrap confidence intervals
```python
def bootstrap_ci(data, statistic, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval for any statistic

    Args:
        data: original sample
        statistic: function to compute (e.g., np.mean, np.median)
        n_bootstrap: number of bootstrap samples
        confidence: confidence level

    Returns:
        (lower_bound, upper_bound)
    """
    # YOUR CODE HERE
    pass

# Test it:
# data = np.random.exponential(2, 100)
# ci = bootstrap_ci(data, np.median)
# Does the true median (2 * ln(2)) fall in the interval?
```

**CE2.4**: Hypothesis testing from scratch
```python
def t_test_one_sample(data, mu0, alternative='two-sided'):
    """
    Perform one-sample t-test
    H0: mu = mu0

    Returns:
        t_statistic, p_value
    """
    # YOUR CODE HERE
    # Compute t-statistic
    # Compute p-value using t-distribution
    pass
```

**CE2.5**: Multiple testing correction
```python
def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction to a list of p-values"""
    # YOUR CODE HERE
    pass

def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction"""
    # YOUR CODE HERE
    pass
```

### Applied Problems 🎯

**AP2.1**: A/B Test Analysis
You run an A/B test on a website:
- Control: 1000 visitors, 45 conversions
- Treatment: 1000 visitors, 62 conversions

Questions:
1. What are the MLE estimates for conversion rates?
2. Construct 95% confidence intervals for each (use bootstrap)
3. Perform a hypothesis test: is treatment better than control?
4. What's your conclusion? What would you tell the product manager?

**AP2.2**: Multiple Testing Problem
You're analyzing gene expression data with 10,000 genes. You perform a t-test for each gene comparing diseased vs healthy tissue.
- At α = 0.05, how many false positives do you expect under the null?
- You find 300 significant p-values. Apply Bonferroni correction. How many remain?
- Apply Benjamini-Hochberg at FDR = 0.05. How many remain?
- Which correction is more appropriate here and why?

**AP2.3**: Power Analysis
You're designing an experiment to detect a 10% improvement in click-through rate (from 5% to 5.5%). You want:
- α = 0.05 (significance level)
- Power = 0.80 (probability of detecting the effect)

How many samples do you need? (Use simulation or power formulas)

### Checkpoint 🚦

**Can you confidently:**
- [ ] Derive and implement MLE for simple distributions?
- [ ] Correctly interpret confidence intervals?
- [ ] Perform and interpret hypothesis tests?
- [ ] Explain when and why to use multiple testing corrections?
- [ ] Implement bootstrap from scratch?
- [ ] Analyze A/B test results properly?

**CRITICAL UNDERSTANDING CHECK:**
Explain why this statement is **wrong**: "A 95% CI means there's a 95% probability the true parameter is in this interval."

(If you can't explain why this is wrong, review confidence intervals!)

---

## 1.3 Bayesian Statistics - Exercises

### Concept Checks ✓

**CC3.1**: State Bayes' theorem and explain each term (prior, likelihood, posterior, evidence).

**CC3.2**: What's the fundamental difference between Bayesian and frequentist inference?

**CC3.3**: What is a conjugate prior? Why are they useful?

**CC3.4**: Explain the difference between a 95% confidence interval (frequentist) and a 95% credible interval (Bayesian).

**CC3.5**: When would you prefer MAP estimation over MLE?

**CC3.6**: How does the posterior behave as you get more data? (Think about prior vs likelihood influence)

### Coding Exercises 💻

**CE3.1**: Bayesian inference for a coin flip
```python
def bayesian_coin_flip(prior_a, prior_b, data):
    """
    Bayesian inference for coin flip probability
    Prior: Beta(prior_a, prior_b)
    Data: list of 0s and 1s

    Returns:
        posterior_a, posterior_b (parameters of posterior Beta distribution)
    """
    # YOUR CODE HERE
    # Use conjugacy: Beta is conjugate prior for Bernoulli
    pass

def plot_bayesian_update(data):
    """
    Visualize how posterior updates as we see more data
    Start with uniform prior Beta(1,1)
    """
    # YOUR CODE HERE
    # Plot prior, likelihood, and posterior at different data points
    pass
```

**CE3.2**: Implement MAP estimation
```python
def map_estimate_normal(data, prior_mu, prior_sigma, data_sigma):
    """
    MAP estimate for mean of normal distribution

    Prior: mu ~ N(prior_mu, prior_sigma^2)
    Likelihood: X_i ~ N(mu, data_sigma^2)

    Returns:
        mu_map: MAP estimate
    """
    # YOUR CODE HERE
    # Derive the formula first!
    # Compare with MLE (sample mean)
    pass
```

**CE3.3**: Simple MCMC sampler
```python
def metropolis_hastings(target_log_prob, initial, n_samples, proposal_std=1.0):
    """
    Metropolis-Hastings MCMC sampler

    Args:
        target_log_prob: function that computes log probability
        initial: starting point
        n_samples: number of samples
        proposal_std: std of proposal distribution (Gaussian random walk)

    Returns:
        samples: array of samples from target distribution
    """
    # YOUR CODE HERE
    pass

# Test it on a simple distribution (e.g., standard normal)
# Check convergence with trace plots
```

**CE3.4**: Naive Bayes classifier
```python
class NaiveBayesClassifier:
    """
    Implement Naive Bayes for binary classification
    Assume features are continuous (use Gaussian likelihood)
    """

    def fit(self, X, y):
        """Estimate parameters from training data"""
        # YOUR CODE HERE
        # Estimate class priors P(y)
        # Estimate feature means and variances for each class
        pass

    def predict_proba(self, X):
        """Return class probabilities using Bayes theorem"""
        # YOUR CODE HERE
        pass

    def predict(self, X):
        """Return predicted class"""
        # YOUR CODE HERE
        pass
```

### Applied Problems 🎯

**AP3.1**: Medical Test Interpretation
A medical test for a rare disease:
- Disease prevalence: 1 in 1000
- Test sensitivity (true positive rate): 99%
- Test specificity (true negative rate): 95%

You test positive. What's the probability you actually have the disease?

(This is a classic Bayesian problem - the answer will surprise you!)

**AP3.2**: A/B Test with Prior Information
You're running an A/B test. From historical data, you know conversion rates are typically around 5% (model as Beta(5, 95) prior).

New data:
- Control: 100 visitors, 4 conversions
- Treatment: 100 visitors, 8 conversions

Using Bayesian inference:
1. Compute posterior distributions for both
2. What's the probability that treatment is better than control?
3. Compute 95% credible intervals
4. Compare with frequentist approach - what's different?

**AP3.3**: Regularization as Prior
You're fitting a linear regression with L2 regularization (Ridge):

Loss = MSE + λ * ||w||²

Show mathematically that this is equivalent to MAP estimation with a Gaussian prior on weights. What prior corresponds to L1 regularization (Lasso)?

### Checkpoint 🚦

**Can you confidently:**
- [ ] Apply Bayes' theorem to real problems?
- [ ] Understand conjugate priors and use them?
- [ ] Explain Bayesian vs frequentist philosophies?
- [ ] Implement basic Bayesian inference?
- [ ] Understand the connection between regularization and priors?
- [ ] Interpret Bayesian results correctly?

**CRITICAL UNDERSTANDING CHECK:**
Implement the medical test problem (AP3.1) from scratch and verify your answer. If the probability seems too low, you understand base rates! If not, review Bayes' theorem carefully.

---

## Final Phase 1 Checkpoint 🎯

### Comprehensive Project: A/B Testing Framework

Build a production-ready A/B testing framework that includes:

**Required Features:**
1. **Frequentist Analysis**
   - MLE estimates with confidence intervals
   - Hypothesis testing (t-test, proportions test)
   - Power analysis and sample size calculation
   - Multiple testing correction

2. **Bayesian Analysis**
   - Prior specification
   - Posterior inference
   - Credible intervals
   - Probability that B > A

3. **Visualization**
   - Distribution plots
   - Confidence/credible intervals
   - Posterior distributions
   - Power curves

4. **Software Engineering**
   - Clean API design
   - Unit tests
   - Documentation
   - CLI or web interface

**Success Criteria:**
- All statistical methods implemented from scratch (numpy only, no scipy.stats)
- Both frequentist and Bayesian approaches
- Properly handles edge cases
- Well-tested and documented

**Time estimate**: 2-3 weeks

### Knowledge Self-Assessment

You're ready for Phase 2 if you can:

**Probability & Distributions:**
- [ ] Derive expectation and variance for common distributions
- [ ] Implement and sample from distributions
- [ ] Apply CLT to real problems

**Inferential Statistics:**
- [ ] Derive MLE for simple distributions
- [ ] Implement confidence intervals (parametric and bootstrap)
- [ ] Perform and interpret hypothesis tests correctly
- [ ] Understand and apply multiple testing corrections
- [ ] Explain p-values and their limitations

**Bayesian Statistics:**
- [ ] Apply Bayes' theorem to real problems
- [ ] Use conjugate priors
- [ ] Understand prior, likelihood, posterior relationship
- [ ] Implement basic Bayesian inference
- [ ] Explain connection to regularization

**Practical Skills:**
- [ ] Analyze A/B test results both ways
- [ ] Implement statistical methods from scratch
- [ ] Visualize statistical concepts
- [ ] Explain concepts to non-technical stakeholders

**Red Flags (means you need more review):**
- ❌ Can't explain why p < 0.05 doesn't mean 95% confidence
- ❌ Can't derive MLE for Bernoulli/Normal from first principles
- ❌ Don't understand the difference between Bayesian and frequentist
- ❌ Can't implement bootstrap from scratch
- ❌ Don't see the connection between regularization and priors

### Quick Self-Test (20 minutes)

1. Derive MLE for Bernoulli distribution from first principles
2. Implement bootstrap CI in 10 lines of code
3. Explain Bayes' theorem and apply it to the medical test problem
4. What's wrong with this: "p = 0.03 means there's 97% chance the null is false"?
5. How is L2 regularization related to Bayesian inference?

If you can do all 5 quickly and correctly, you're ready for Phase 2! 🎉

---

## Solutions and Hints

Solutions are intentionally not provided here - the learning comes from struggling with the problems. However:

**Where to check your work:**
- Implement both with your code and scipy.stats - compare results
- For statistical tests: cross-reference with R or statsmodels
- For Bayesian problems: use PyMC to verify your analytical solutions
- For conceptual questions: "All of Statistics" or "Think Stats" have answers

**Getting unstuck:**
- Re-read the relevant section in your textbook
- Watch StatQuest or 3Blue1Brown videos
- Derive mathematically before coding
- Start with simple examples then generalize
- Draw pictures and visualizations

**Discussion:**
Consider creating a study group or finding an accountability partner to discuss these problems!

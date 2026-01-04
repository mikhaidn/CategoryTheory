# Exercise & Self-Assessment Guide

> How to use the exercises to ensure mastery before progressing

## Overview

Each phase directory contains an `exercises.md` file with concrete problems to verify your understanding. These aren't optional - they're designed to ensure you've truly mastered each phase before moving on.

## Exercise Structure

### 1. Concept Checks ✓
**Quick theoretical questions** to test understanding.

**Example:** "What's the difference between a 95% confidence interval and a 95% credible interval?"

**Purpose:** Ensure you understand the theory, not just the code.

**Time:** 2-5 minutes each

**When to use:** After reading each section, before implementing

### 2. Coding Exercises 💻
**Implementation challenges** from scratch (usually numpy only).

**Example:** "Implement bootstrap confidence intervals from scratch"

**Purpose:** Deep understanding through implementation

**Time:** 30 minutes to 2 hours each

**When to use:** After concept checks pass

### 3. Applied Problems 🎯
**Real-world scenarios** that combine multiple concepts.

**Example:** "Analyze this A/B test dataset using both frequentist and Bayesian approaches"

**Purpose:** Learn to apply knowledge to practical problems

**Time:** 1-4 hours each

**When to use:** After mastering the basics

### 4. Must-Implement Algorithms ✓
**Core algorithms** you MUST build from scratch.

**Example Phase 4:** Logistic regression, decision trees, k-means, PCA

**Purpose:** You don't understand it if you can't implement it

**Time:** 2-8 hours each

**When to use:** Throughout the phase

### 5. Checkpoints 🚦
**Self-assessment criteria** before moving to next phase.

**Format:**
- [ ] Can you do X?
- [ ] Understand Y deeply?
- [ ] Implemented Z from scratch?

**Purpose:** Honest self-evaluation

**Time:** 5 minutes to review

**When to use:** Before starting next phase

### 6. Final Comprehensive Project
**Large capstone project** integrating all phase concepts.

**Examples:**
- Phase 1: A/B Testing Framework
- Phase 4: Mini-sklearn library
- Phase 6: Production RAG system
- Phase 7: Full MLOps pipeline

**Purpose:** Demonstrate mastery, build portfolio

**Time:** 2-8 weeks

**When to use:** End of phase, before progressing

## How to Use Exercises

### Recommended Flow

1. **Read the phase README** to understand the concepts

2. **Do concept checks** for each section
   - If you can't answer, re-read the material
   - Don't move on until you understand

3. **Implement coding exercises**
   - Start from scratch (numpy only when specified)
   - No copy-pasting from Stack Overflow
   - Understand every line you write

4. **Gradient checking** (for ML algorithms)
   - Always verify your backprop/gradients
   - Compare analytical vs numerical gradients

5. **Work through applied problems**
   - These simulate real-world usage
   - Document your approach

6. **Check off the checkpoint list**
   - Be honest with yourself
   - If <80% checked, review that section

7. **Complete the final project**
   - This is the real test
   - Production-quality code
   - Comprehensive documentation

8. **Self-test** (at end of exercises.md)
   - Timed quick assessment
   - If you pass, you're ready!

### The 80% Rule

You should be able to complete **at least 80%** of exercises before moving to the next phase. It's okay to skip a few harder ones, but:

- ❌ Can't implement MLE from scratch? → Not ready
- ✅ Struggled with one advanced Bayesian problem? → Probably fine

### When to Ask for Help

1. **After trying** for at least 30 minutes
2. **After checking** textbooks and documentation
3. **After breaking down** the problem into smaller pieces

Good places:
- Stack Overflow (search first!)
- r/learnmachinelearning
- Study groups
- Office hours (if in a course)

### Solutions

Solutions are **intentionally not provided**. Here's how to check your work:

**For coding exercises:**
```python
# Your implementation
result_yours = your_function(data)

# Reference implementation (e.g., scipy, sklearn)
result_reference = scipy.stats.function(data)

# Compare (should be very close)
assert np.allclose(result_yours, result_reference)
```

**For concept checks:**
- Check textbooks (page numbers in resources.md)
- Cross-reference with StatQuest videos
- Ask GPT-4 to explain (but don't just accept the answer)

**For applied problems:**
- Compare with Kaggle notebooks
- Verify metrics make sense
- Get code review from peers

## Time Estimates

### Per Phase Breakdown

**Phase 1 (Statistics): 9-13 weeks**
- Concept checks: 1-2 hours per section
- Coding exercises: 20-30 hours total
- Applied problems: 10-15 hours
- Final project: 2-3 weeks

**Phase 2 (Math): 8-12 weeks**
- Similar time split
- More emphasis on derivations

**Phase 3 (Python): 9-10 weeks**
- Less theory, more practice
- Lots of hands-on coding

**Phase 4 (Classical ML): 18-21 weeks**
- Many algorithms to implement
- Final project is substantial (mini-sklearn)

**Phase 5 (Deep Learning): 23-31 weeks**
- Building from scratch takes time
- Reproducing papers is hard
- Worth it for deep understanding

**Phase 6 (Modern AI): 23-32 weeks**
- More project-based
- Experimentation time varies
- Can overlap with Phase 7

**Phase 7 (MLOps): 19-27 weeks**
- Leverage your SWE skills
- Many components to integrate
- Production system takes time

### Weekly Time Commitment

Assumes **10-15 hours/week** of focused study:
- Reading/watching: 3-5 hours
- Coding exercises: 5-8 hours
- Applied problems: 2-4 hours

At **20+ hours/week**, you can cut timeline in half.

## Red Flags

**You're NOT ready to move on if:**

❌ Can't explain concepts clearly to someone else

❌ Can't implement core algorithms from memory

❌ Haven't completed the final project

❌ Just copy-pasted code without understanding

❌ Can't pass the quick self-test

❌ Checkpoint list <50% checked

**It's okay to move on if:**

✅ 80%+ of exercises completed

✅ Core algorithms implemented from scratch

✅ Can derive key equations

✅ Final project completed

✅ Self-test passed

✅ Can explain concepts clearly

## Motivation

### Why So Many Exercises?

**Because ML is learned by doing.** You can read all the textbooks, but until you:
- Derive the math
- Implement from scratch
- Debug why it's not working
- Apply to real problems

...you don't **really** understand it.

### Why "From Scratch"?

**Because frameworks hide complexity.** Using `sklearn.LogisticRegression()` is easy. Understanding:
- Why cross-entropy loss?
- How does gradient descent work here?
- What's the gradient of logistic loss?
- Why does regularization help?

...requires implementation.

**After** you implement from scratch, frameworks make sense. You know what's happening under the hood.

### Your SWE Advantage

As a professional SWE, you already know:
- ✅ How to debug
- ✅ How to write tests
- ✅ How to read documentation
- ✅ How to break down problems

Apply these skills to learning ML! Write tests for your implementations. Debug systematically. Read the papers like documentation.

## Tips for Success

### 1. Start Small
Don't try to implement ResNet from scratch on day 1. Build up gradually:
- Simple linear regression
- Then logistic regression
- Then neural network (1 layer)
- Then deep network
- Then CNN
- ...

### 2. Verify Everything
**Gradient checking** is your friend:
```python
def gradient_check(f, x, epsilon=1e-5):
    analytical_grad = your_backward_pass(x)
    numerical_grad = (f(x + eps) - f(x - eps)) / (2 * eps)
    assert np.allclose(analytical_grad, numerical_grad)
```

### 3. Visualize
Plot everything:
- Loss curves
- Gradient norms
- Learned features
- Decision boundaries
- Attention weights

Visualization builds intuition.

### 4. Compare with Libraries
Your implementation should match sklearn/scipy (within numerical precision):
```python
assert np.allclose(your_result, sklearn_result, rtol=1e-5)
```

If it doesn't, debug!

### 5. Document Your Learnings
Keep a learning journal:
- What you learned
- What was hard
- Aha moments
- Mistakes made

Writing solidifies understanding.

### 6. Build Portfolio
All your exercise code → GitHub:
- Clean, documented code
- Notebooks with explanations
- README for each project

This becomes your portfolio.

### 7. Teach Others
Best way to learn:
- Write blog posts
- Answer Stack Overflow questions
- Explain to a study partner
- Create tutorial videos

Teaching forces clarity.

## Getting Unstuck

### Debugging Checklist

**For ML code that doesn't work:**

1. **Start simple**
   - Overfit on 1 example first
   - Then 10 examples
   - Then full dataset

2. **Check shapes**
   - Print tensor shapes everywhere
   - Most bugs are shape mismatches

3. **Verify gradients**
   - Gradient checking
   - Compare with autograd

4. **Visualize**
   - Plot loss curves
   - Plot predictions
   - Plot learned weights

5. **Compare with reference**
   - sklearn/scipy/PyTorch implementation
   - Should match!

6. **Simplify**
   - Remove complexity until it works
   - Add back piece by piece

### When You're Truly Stuck

1. **Break it down** into smaller pieces
2. **Implement** the smallest piece
3. **Test** that piece
4. **Repeat**

Example: Implementing neural network
- Start: Just forward pass, 1 layer
- Test: Compare with manual calculation
- Add: Backward pass
- Test: Gradient checking
- Add: Second layer
- Test: Gradient checking again
- ...

### Resources for Help

- **Documentation**: Read it carefully
- **Textbooks**: Derive along with them
- **Papers**: Read the original papers
- **Videos**: StatQuest, 3Blue1Brown, etc.
- **Code**: Read sklearn/PyTorch source code
- **Forums**: Stack Overflow, Reddit
- **Study groups**: Find accountability partners

## Final Thoughts

These exercises are **hard**. They're supposed to be. Machine learning is a complex field, and mastering it takes time and effort.

But if you work through these systematically:
- ✅ You'll have **deep understanding** (not surface-level)
- ✅ You'll be able to **implement papers** from scratch
- ✅ You'll **debug** models effectively
- ✅ You'll **know when to use** which technique
- ✅ You'll have a **strong portfolio**
- ✅ You'll be **job-ready** as an ML engineer

**Take your time. Do the work. Don't skip exercises.**

The difference between someone who reads about ML and someone who truly understands it is the willingness to implement from scratch and work through hard problems.

You've got this! 🚀

---

**Questions about the exercises?** Open an issue or discussion in the repo!

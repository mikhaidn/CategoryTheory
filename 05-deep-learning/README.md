# Phase 5: Deep Learning Theory & Practice

> Understand neural networks from first principles to modern architectures

## Prerequisites
- Phases 1-4 complete (especially matrix calculus from Phase 2)
- Comfortable with optimization and backpropagation math
- NumPy proficiency

## Strategy

**Build intuition through implementation**: Implement everything from scratch first (numpy), then use frameworks (PyTorch/TensorFlow).

## Learning Path

### 5.1 Neural Network Fundamentals

#### Theory
- [ ] **Perceptron and multilayer perceptron**
  - Biological inspiration (briefly)
  - Universal approximation theorem
  - Limitations of shallow networks
- [ ] **Activation functions**
  - Sigmoid, tanh, ReLU, Leaky ReLU, GELU
  - Why non-linearity is necessary
  - Dying ReLU problem
- [ ] **Forward propagation**
  - Matrix formulation
  - Computational graph concept
- [ ] **Backpropagation**
  - Chain rule for derivatives (Phase 2 matrix calculus!)
  - Derive from scratch for a 2-layer network
  - Computational graph perspective
  - Automatic differentiation concept

#### Implementation
- [ ] Implement forward pass in numpy
- [ ] Implement backpropagation in numpy
- [ ] Train a simple network on MNIST
- [ ] Visualize learned features

**Resources**:
- Blog: "Yes you should understand backprop" by Andrej Karpathy
- Video: 3Blue1Brown neural network series
- Course: Stanford CS231n (lectures 1-4)
- Interactive: TensorFlow Playground

**Project**: Build a neural network library from scratch (numpy only) with:
- [ ] Modular layer design
- [ ] Multiple activation functions
- [ ] Various loss functions
- [ ] Gradient checking to verify backprop
- [ ] Train on MNIST

### 5.2 Training Deep Networks

#### Optimization
- [ ] **Gradient descent variants** (from Phase 2, now applied)
  - SGD, momentum, Nesterov
  - Adam, AdamW, AdaGrad, RMSprop
- [ ] **Learning rate strategies**
  - Fixed, decay, cosine annealing
  - Learning rate warmup
  - Cyclical learning rates
- [ ] **Batch normalization**
  - Why it works (multiple theories)
  - Implementation details
  - Layer normalization, group normalization
- [ ] **Regularization**
  - L1/L2 (Bayesian interpretation from Phase 1!)
  - Dropout (Bayesian connection!)
  - Early stopping
  - Data augmentation
  - Weight decay

#### Challenges
- [ ] Vanishing/exploding gradients
- [ ] Overfitting in deep networks
- [ ] Mode collapse and local minima
- [ ] Debugging training (loss curves, gradient norms)

#### Initialization
- [ ] Xavier/Glorot initialization
- [ ] He initialization
- [ ] Why initialization matters

**Resources**:
- Book: "Deep Learning" by Goodfellow, Bengio, Courville (chapters 6-8)
- Course: fast.ai (practical deep learning)
- Blog: Distill.pub articles on optimization

**Exercises**:
- [ ] Implement different optimizers from scratch
- [ ] Experiment with initialization strategies
- [ ] Visualize gradient flow through deep networks
- [ ] Implement batch normalization

### 5.3 Convolutional Neural Networks (CNNs)

#### Theory
- [ ] **Convolution operation**
  - Mathematical definition
  - Why convolutions for images (inductive bias)
  - Receptive fields
- [ ] **Pooling layers**
  - Max pooling, average pooling
  - Spatial pyramid pooling
- [ ] **CNN architectures**
  - LeNet (historical)
  - AlexNet (the revolution)
  - VGG (depth matters)
  - ResNet (skip connections solve vanishing gradients!)
  - Inception/GoogLeNet (multiple scales)
  - EfficientNet (compound scaling)

#### Modern Techniques
- [ ] Transfer learning and fine-tuning
- [ ] Data augmentation for images
- [ ] Object detection (R-CNN family, YOLO)
- [ ] Semantic segmentation (U-Net, FCN)

**Resources**:
- Course: Stanford CS231n (Convolutional Neural Networks for Visual Recognition)
- Papers: AlexNet, ResNet, Inception, EfficientNet
- Practice: fast.ai course

**Projects**:
- [ ] Implement a simple CNN from scratch
- [ ] Visualize CNN filters and activations
- [ ] Image classification on CIFAR-10
- [ ] Transfer learning project (use pretrained model)
- [ ] Object detection or segmentation project

### 5.4 Recurrent Neural Networks (RNNs)

#### Theory
- [ ] **Sequence modeling motivation**
  - Why feedforward networks fail for sequences
  - Time as a dimension
- [ ] **RNN architecture**
  - Recurrent connections
  - Unrolling through time
  - Backpropagation through time (BPTT)
- [ ] **LSTM & GRU**
  - Solving vanishing gradients for sequences
  - Gates mechanism
  - When to use LSTM vs GRU
- [ ] **Bidirectional RNNs**
  - Forward and backward context
- [ ] **Attention mechanism** (precursor to Transformers)
  - Sequence-to-sequence models
  - Attention weights interpretation

**Resources**:
- Blog: "Understanding LSTM Networks" by Christopher Olah
- Course: Stanford CS224n (NLP with Deep Learning, first half)

**Projects**:
- [ ] Implement RNN from scratch
- [ ] Implement LSTM from scratch
- [ ] Text generation (character-level or word-level)
- [ ] Sentiment analysis
- [ ] Time series forecasting

### 5.5 Transformers & Attention
**The current paradigm**

#### Theory
- [ ] **Self-attention mechanism**
  - Query, key, value formulation
  - Scaled dot-product attention
  - Why attention works
- [ ] **Multi-head attention**
  - Learning multiple representations
  - Parallel attention heads
- [ ] **Positional encoding**
  - Why transformers need position information
  - Sinusoidal encoding
  - Learned positional embeddings
- [ ] **Transformer architecture**
  - Encoder-decoder structure
  - Feed-forward layers
  - Layer normalization
  - Residual connections

#### Modern Variants
- [ ] BERT (bidirectional encoder)
- [ ] GPT (autoregressive decoder)
- [ ] T5 (text-to-text framework)
- [ ] Vision Transformers (ViT)

**Resources**:
- Paper: "Attention Is All You Need" (original Transformer paper)
- Blog: "The Illustrated Transformer" by Jay Alammar
- Course: Stanford CS224n (NLP, second half)
- Interactive: Transformer Explainer

**Projects**:
- [ ] Implement self-attention from scratch
- [ ] Implement a mini-transformer
- [ ] Fine-tune BERT for text classification
- [ ] Train a small language model
- [ ] Explore Vision Transformers

### 5.6 Generative Models
**Creating new data**

#### Variational Autoencoders (VAEs)
- [ ] Encoder-decoder architecture
- [ ] Latent space representation
- [ ] Reparameterization trick
- [ ] ELBO (connects to Phase 1 Bayesian inference!)
- [ ] Disentangled representations

#### Generative Adversarial Networks (GANs)
- [ ] Generator and discriminator
- [ ] Minimax game formulation
- [ ] Training challenges (mode collapse, etc.)
- [ ] Variants: DCGAN, StyleGAN, conditional GANs

#### Diffusion Models
- [ ] Forward and reverse diffusion process
- [ ] Training objective
- [ ] Why they work better than GANs for some tasks
- [ ] DALL-E 2, Stable Diffusion

**Resources**:
- Blog: "Understanding Variational Autoencoders" by Irhum Shafkat
- Paper: "Tutorial on Variational Autoencoders" by Doersch
- Blog: "Generative Adversarial Networks" by Ian Goodfellow
- Course: Stanford CS236 (Deep Generative Models)

**Projects**:
- [ ] Implement VAE on MNIST
- [ ] Implement simple GAN
- [ ] Experiment with diffusion models (using libraries)
- [ ] Generate synthetic images

### 5.7 Deep Learning Frameworks

#### PyTorch (Recommended)
- [ ] Tensors and autograd
- [ ] Building models (nn.Module)
- [ ] Training loops
- [ ] Data loaders and transforms
- [ ] Distributed training
- [ ] TorchScript for deployment
- [ ] PyTorch Lightning (high-level wrapper)

#### TensorFlow/Keras (Alternative)
- [ ] Similar concepts, different API
- [ ] TensorFlow 2.0+ (eager execution)
- [ ] Keras functional and subclassing API
- [ ] TensorFlow Serving for deployment

**Strategy**: Pick one (PyTorch recommended), get fluent, understand the other

**Resources**:
- Official PyTorch tutorials
- Book: "Deep Learning with PyTorch" by Stevens et al.
- fast.ai course (PyTorch-based)

### 5.8 Advanced Topics

- [ ] Graph neural networks (GNNs)
- [ ] Reinforcement learning basics (DQN, policy gradients)
- [ ] Meta-learning and few-shot learning
- [ ] Neural architecture search
- [ ] Model compression (quantization, pruning, distillation)
- [ ] Interpretability and explainability

**Resources**: Research papers, conferences (NeurIPS, ICML, ICLR)

## Key Projects

### Project 1: Deep Learning Library
Build a mini PyTorch from scratch:
- Automatic differentiation (autograd)
- Common layers (Linear, Conv2d, LSTM)
- Optimizers
- Training utilities
- Clean API design (leverage your SWE skills)

### Project 2: Reproduce a Research Paper
Pick a classic paper (ResNet, BERT, etc.):
- Implement from scratch
- Train on appropriate dataset
- Match reported results
- Document learnings

### Project 3: End-to-End Deep Learning Application
Build a complete application:
- Data collection and preprocessing
- Model architecture design
- Training with experiment tracking
- Deployment (connects to Phase 7!)
- Web interface
- All production-ready code

## Self-Assessment

Before moving to modern AI:
- [ ] Can implement backpropagation from scratch
- [ ] Understand why deep learning works (theory + intuition)
- [ ] Comfortable with PyTorch or TensorFlow
- [ ] Can debug training issues
- [ ] Understand CNNs, RNNs, Transformers deeply
- [ ] Have trained non-trivial models from scratch
- [ ] Can read and implement research papers

## Estimated Timeline

- **Fundamentals & backprop**: 3-4 weeks
- **Training techniques**: 2-3 weeks
- **CNNs**: 3-4 weeks
- **RNNs**: 2-3 weeks
- **Transformers**: 3-4 weeks
- **Generative models**: 2-3 weeks
- **Frameworks**: 2 weeks (learning while doing)
- **Projects**: 6-8 weeks

**Total**: 23-31 weeks

## Next Steps

Proceed to [Phase 6: Modern AI & LLMs](../06-modern-ai/README.md) to understand current state-of-the-art.

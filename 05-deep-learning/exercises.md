# Phase 5: Deep Learning - Exercises & Self-Assessment

> Master neural networks from first principles to modern architectures

## Core Philosophy

**Learn by building**: Implement everything from scratch first (numpy), then use frameworks (PyTorch/TensorFlow).

---

## 5.1 Neural Network Fundamentals - Exercises

### Must-Implement ✓

**Neural Network from Scratch**
- [ ] Forward propagation (matrix formulation)
- [ ] Backpropagation (derive from first principles)
- [ ] Multiple activation functions (sigmoid, tanh, ReLU)
- [ ] Multiple loss functions (MSE, cross-entropy)
- [ ] Train on MNIST
- [ ] Gradient checking to verify correctness

### Coding Exercises 💻

**CE1.1**: Build a neural network library
```python
class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

    def backward(self, grad):
        # YOUR CODE HERE
        # Compute gradients w.r.t. weights, biases, and input
        pass

class ReLU(Layer):
    # YOUR CODE HERE
    pass

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        # YOUR CODE HERE
        pass

    def backward(self, loss_grad):
        # YOUR CODE HERE
        pass

    def train_step(self, x, y, lr):
        # YOUR CODE HERE
        pass
```

**CE1.2**: Gradient checking
```python
def gradient_check(layer, x, epsilon=1e-5):
    """
    Verify backprop implementation using finite differences

    For each weight w:
    - Compute analytical gradient (from backprop)
    - Compute numerical gradient: (f(w+ε) - f(w-ε)) / (2ε)
    - Compare (should be very close)
    """
    # YOUR CODE HERE
    pass
```

### Applied Problems 🎯

**AP1.1**: MNIST from scratch
- Build 2-layer network (numpy only)
- Train on MNIST
- Achieve >95% test accuracy
- Visualize learned features
- Plot loss curves and accuracy

**AP1.2**: Implement different architectures
- Vary depth (2, 3, 5 hidden layers)
- Vary width (32, 64, 128 units)
- Compare performance and training time
- Observe overfitting patterns

### Checkpoint 🚦
- [ ] Derived backprop from first principles?
- [ ] Implemented neural network from scratch?
- [ ] Gradient checking passes?
- [ ] Trained on MNIST successfully?

---

## 5.2 Training Deep Networks - Exercises

### Must-Implement ✓

**Optimization**
- [ ] Implement SGD, momentum, Adam (from Phase 2, apply to neural nets)
- [ ] Learning rate schedules (decay, cosine annealing)

**Regularization**
- [ ] Dropout (understand why it works!)
- [ ] Batch normalization
- [ ] Early stopping
- [ ] Data augmentation

### Coding Exercises 💻

**CE2.1**: Implement regularization techniques
```python
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, training=True):
        # YOUR CODE HERE
        # Different behavior for train vs test!
        pass

    def backward(self, grad):
        # YOUR CODE HERE
        pass

class BatchNorm(Layer):
    # YOUR CODE HERE
    # Track running mean/variance for inference
    pass
```

**CE2.2**: Initialization experiments
```python
# Compare different initialization schemes:
# - Zero initialization
# - Random normal (small, large variance)
# - Xavier/Glorot initialization
# - He initialization

# Plot: gradient norms through layers
# Observe: vanishing/exploding gradients
```

### Applied Problems 🎯

**AP2.1**: Debug poor training
Given a network that trains poorly:
- Diagnose the issue (learning rate, initialization, etc.)
- Fix it
- Document what you learned

### Checkpoint 🚦
- [ ] Understand optimization for deep networks?
- [ ] Implemented dropout and batch norm?
- [ ] Can debug training issues?

---

## 5.3 Convolutional Neural Networks - Exercises

### Must-Implement ✓

**CNN Basics**
- [ ] Implement 2D convolution operation
- [ ] Implement max pooling
- [ ] Build a simple CNN (Conv-Pool-FC)
- [ ] Train on CIFAR-10

### Coding Exercises 💻

**CE3.1**: Convolution from scratch
```python
def conv2d(x, kernel, stride=1, padding=0):
    """
    2D convolution (numpy only)

    x: (batch, height, width, in_channels)
    kernel: (kernel_h, kernel_w, in_channels, out_channels)
    """
    # YOUR CODE HERE
    pass

def conv2d_backward(grad, x, kernel):
    """
    Backprop through convolution
    Compute gradients w.r.t. input and kernel
    """
    # YOUR CODE HERE
    pass
```

**CE3.2**: Build CNN with PyTorch/TensorFlow
```python
# Now that you understand it, use a framework
# Implement LeNet or a simple modern architecture
# Train on CIFAR-10
# Achieve >75% accuracy
```

### Applied Problems 🎯

**AP3.1**: Image classification
- Pick dataset: CIFAR-10, CIFAR-100, or Tiny ImageNet
- Design CNN architecture
- Data augmentation
- Achieve competitive accuracy
- Visualize filters and activations

**AP3.2**: Transfer learning
- Use pretrained ResNet or VGG
- Fine-tune on small dataset (e.g., Cats vs Dogs)
- Compare with training from scratch
- Analyze when transfer learning helps

### Checkpoint 🚦
- [ ] Understand convolution operation deeply?
- [ ] Implemented CNN from scratch?
- [ ] Can use pretrained models effectively?

---

## 5.4 Recurrent Neural Networks - Exercises

### Must-Implement ✓

**RNN/LSTM**
- [ ] Implement vanilla RNN
- [ ] Implement LSTM
- [ ] Understand gates mechanism
- [ ] Train on sequence task

### Coding Exercises 💻

**CE4.1**: RNN from scratch
```python
class RNN:
    def forward(self, x_sequence):
        """
        x_sequence: list of inputs [x_1, x_2, ..., x_T]
        """
        # YOUR CODE HERE
        pass

    def backward(self, grad_sequence):
        """
        Backpropagation through time (BPTT)
        """
        # YOUR CODE HERE
        pass

class LSTM:
    # YOUR CODE HERE
    # Implement forget gate, input gate, output gate
    pass
```

### Applied Problems 🎯

**AP4.1**: Text generation
- Character-level or word-level model
- Train on text corpus (e.g., Shakespeare)
- Generate new text
- Experiment with temperature

**AP4.2**: Sentiment analysis
- Use LSTM for binary classification
- Train on IMDB reviews or Twitter data
- Compare with simple baselines

### Checkpoint 🚦
- [ ] Implemented RNN/LSTM from scratch?
- [ ] Understand BPTT?
- [ ] Applied to sequence tasks?

---

## 5.5 Transformers & Attention - Exercises

### Must-Implement ✓

**Attention Mechanism**
- [ ] Implement scaled dot-product attention
- [ ] Implement multi-head attention
- [ ] Understand positional encoding
- [ ] Build a mini-transformer

### Coding Exercises 💻

**CE5.1**: Attention from scratch
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_model)
    Returns: (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    pass

class MultiHeadAttention:
    # YOUR CODE HERE
    pass

class TransformerBlock:
    # YOUR CODE HERE
    # Self-attention + FFN + residual + layer norm
    pass
```

### Applied Problems 🎯

**AP5.1**: Train a small language model
- Build GPT-like architecture (decoder-only)
- Train on text corpus
- Generate text
- Compare with RNN/LSTM

**AP5.2**: Fine-tune BERT
- Use Hugging Face transformers
- Fine-tune on text classification task
- Achieve SOTA performance
- Analyze attention weights

### Checkpoint 🚦
- [ ] Understand attention mechanism deeply?
- [ ] Implemented transformer components?
- [ ] Can use pretrained transformers?

---

## 5.6 Generative Models - Exercises

### Applied Problems 🎯

**AP6.1**: Variational Autoencoder
- Implement VAE on MNIST
- Understand reparameterization trick
- Visualize latent space
- Generate new samples

**AP6.2**: Generative Adversarial Network
- Implement simple GAN
- Train on MNIST or CIFAR-10
- Deal with training challenges
- Generate images

---

## Final Phase 5 Checkpoint 🎯

### Comprehensive Project: Deep Learning Library

Build a mini-PyTorch from scratch:

**Core Components:**
1. **Automatic Differentiation**
   - Computational graph
   - Forward and backward pass
   - Gradient accumulation

2. **Layers**
   - Linear, Conv2d, RNN, LSTM
   - Activation functions
   - Normalization layers
   - Dropout

3. **Optimizers**
   - SGD, Momentum, Adam
   - Learning rate scheduling

4. **Loss Functions**
   - MSE, Cross-Entropy
   - Custom losses

5. **Utilities**
   - Data loaders
   - Model checkpointing
   - Training loops

**Plus**: Train a non-trivial model using your framework!

**Time estimate**: 6-8 weeks

### Alternative Project: Reproduce a Research Paper

Pick a seminal paper:
- ResNet, DenseNet, or U-Net (CNNs)
- LSTM improvements or attention variants (RNNs)
- BERT or GPT-2 (small version) (Transformers)

**Reproduce:**
- Implement architecture exactly
- Train on appropriate dataset
- Match reported results (within reason)
- Document findings

### Knowledge Self-Assessment

You're ready for Phase 6 if:

**Implementation:**
- [ ] Built neural network library from scratch
- [ ] Implemented backprop with gradient checking
- [ ] Can implement papers from reading them
- [ ] Comfortable with PyTorch or TensorFlow

**Understanding:**
- [ ] Understand backpropagation at a deep level
- [ ] Know how attention works (mathematically)
- [ ] Can explain why deep learning works
- [ ] Understand training dynamics

**Application:**
- [ ] Trained models on MNIST, CIFAR-10, text
- [ ] Used transfer learning effectively
- [ ] Fine-tuned pretrained models
- [ ] Debugged training issues

### Quick Self-Test (1 hour)

1. Derive backprop for a 2-layer network (full derivation)
2. Implement scaled dot-product attention from memory
3. Explain why batch normalization helps training
4. When to use CNN vs RNN vs Transformer?
5. Implement dropout correctly (train vs test mode)

All correct + comprehensive project? → Phase 6! 🚀

---

**Remember**: Deep learning is about understanding the fundamentals. Master backpropagation before using frameworks blindly.

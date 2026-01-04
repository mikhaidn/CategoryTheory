# Phase 6: Modern AI & LLMs - Exercises & Self-Assessment

> Master current state-of-the-art AI systems and applications

## Focus: Practical Application + Deep Understanding

This phase is more project-focused since you're working with large models that require frameworks.

---

## 6.1 Large Language Models - Exercises

### Concept Checks ✓

**CC1.1**: Explain the GPT architecture. How does it differ from BERT?

**CC1.2**: What are emergent abilities in LLMs? Give examples.

**CC1.3**: Explain tokenization. Why does it matter for model performance?

**CC1.4**: What is RLHF? Explain the three-step process (SFT, reward modeling, RL).

**CC1.5**: Compare DPO vs RLHF. What are the tradeoffs?

### Coding Exercises 💻

**CE1.1**: Tokenization exploration
```python
# Compare different tokenizers:
# - tiktoken (OpenAI)
# - sentencepiece (Google)
# - Hugging Face tokenizers

# Analyze:
# - Vocabulary size
# - Tokens per word
# - Multilingual performance
# - Special tokens handling
```

**CE1.2**: Fine-tune a small LM
```python
from transformers import GPT2LMHeadModel, Trainer

# Fine-tune GPT-2 small on your own corpus
# Tasks to try:
# - Domain-specific language model (e.g., medical, legal)
# - Code generation
# - Story generation

# Evaluate perplexity before/after
```

### Applied Problems 🎯

**AP1.1**: Build a domain-specific LM
- Pick a domain (code, science, law, etc.)
- Collect/curate training data
- Fine-tune GPT-2 or similar
- Evaluate on domain-specific tasks
- Compare with base model

**AP1.2**: Analyze scaling laws
- Train models of different sizes (if compute allows)
- Or use existing checkpoints
- Plot performance vs parameters/data/compute
- Verify scaling law predictions

### Checkpoint 🚦
- [ ] Understand LLM architectures?
- [ ] Can fine-tune models?
- [ ] Know tokenization deeply?
- [ ] Understand training process?

---

## 6.2 Prompt Engineering - Exercises

### Practical Exercises 💻

**PE2.1**: Prompting techniques comparison
```python
# For a given task, implement:
# 1. Zero-shot prompting
# 2. Few-shot prompting
# 3. Chain-of-thought
# 4. Self-consistency
# 5. ReAct

# Compare results systematically
# Measure: accuracy, consistency, cost
```

**PE2.2**: Build a prompt optimization framework
```python
class PromptOptimizer:
    """
    Given a task and test cases,
    automatically find better prompts
    """
    def __init__(self, task_description, test_cases):
        pass

    def generate_candidate_prompts(self):
        # YOUR CODE HERE
        pass

    def evaluate_prompt(self, prompt):
        # YOUR CODE HERE
        pass

    def optimize(self):
        # YOUR CODE HERE
        pass
```

### Applied Problems 🎯

**AP2.1**: Complex reasoning task
- Pick a task requiring multi-step reasoning
- Implement chain-of-thought prompting
- Compare with direct prompting
- Measure accuracy improvement

**AP2.2**: Build a prompt library
- Create reusable prompt templates
- Version control
- A/B testing framework
- Documentation

### Checkpoint 🚦
- [ ] Mastered prompting techniques?
- [ ] Built systematic evaluation?
- [ ] Can optimize prompts?

---

## 6.3 Retrieval-Augmented Generation - Exercises

### Must-Build ✓

**RAG System Components**
- [ ] Document ingestion and chunking
- [ ] Embedding generation
- [ ] Vector database integration
- [ ] Retrieval strategies
- [ ] Generation with context

### Coding Exercises 💻

**CE3.1**: Build a RAG system from scratch
```python
class RAGSystem:
    def __init__(self, embedding_model, llm, vector_db):
        # YOUR CODE HERE
        pass

    def ingest_documents(self, documents):
        """
        Chunk, embed, and store documents
        """
        # YOUR CODE HERE
        pass

    def retrieve(self, query, k=5):
        """
        Find k most relevant chunks
        """
        # YOUR CODE HERE
        pass

    def generate(self, query, context):
        """
        Generate answer using retrieved context
        """
        # YOUR CODE HERE
        pass

    def query(self, question):
        """
        End-to-end: retrieve + generate
        """
        # YOUR CODE HERE
        pass
```

**CE3.2**: Advanced RAG techniques
```python
# Implement:
# 1. Multi-query retrieval
# 2. Reranking
# 3. Hypothetical document embeddings (HyDE)
# 4. Contextual compression

# Compare performance on your use case
```

### Applied Problems 🎯

**AP3.1**: Documentation Q&A system
- Pick a large documentation (e.g., Python docs, TensorFlow docs)
- Build RAG system
- Evaluate on hand-crafted Q&A pairs
- Measure: answer quality, retrieval accuracy

**AP3.2**: Compare RAG vs fine-tuning
- Same task, two approaches
- Measure: accuracy, cost, latency, update frequency
- When to use each?

**AP3.3**: RAG evaluation framework
```python
# Build automated evaluation:
# - Retrieval metrics (precision, recall, MRR)
# - Generation metrics (BLEU, ROUGE, or LLM-as-judge)
# - End-to-end metrics
# - Latency and cost tracking
```

### Checkpoint 🚦
- [ ] Built a working RAG system?
- [ ] Understand embedding models?
- [ ] Can evaluate RAG quality?
- [ ] Know when to use RAG vs alternatives?

---

## 6.4 Fine-Tuning & Model Adaptation - Exercises

### Must-Implement ✓

**LoRA Fine-Tuning**
- [ ] Fine-tune with LoRA on a task
- [ ] Compare with full fine-tuning (if compute allows)
- [ ] Analyze parameter efficiency

### Coding Exercises 💻

**CE4.1**: Fine-tune with LoRA
```python
from peft import LoraConfig, get_peft_model

# Fine-tune a model with LoRA:
# 1. Choose base model (e.g., Llama-2-7B)
# 2. Create instruction dataset
# 3. Configure LoRA (rank, alpha, target modules)
# 4. Train
# 5. Merge and save

# Compare:
# - Training time
# - Memory usage
# - Final performance
# vs full fine-tuning
```

**CE4.2**: Create instruction dataset
```python
class InstructionDatasetCreator:
    """
    Tools for creating instruction-tuning datasets
    """
    def generate_from_examples(self, examples):
        # Use GPT-4 to generate more examples
        pass

    def format_for_training(self, data):
        # Alpaca, ShareGPT, or custom format
        pass

    def quality_filter(self, dataset):
        # Filter low-quality examples
        pass
```

### Applied Problems 🎯

**AP4.1**: Domain expert fine-tuning
- Pick a domain (medical, legal, code, etc.)
- Curate high-quality instruction data
- Fine-tune with LoRA or QLoRA
- Evaluate rigorously
- Compare with base model and GPT-4

**AP4.2**: Fine-tuning vs RAG comparison
For the same task:
- Approach 1: RAG with base model
- Approach 2: Fine-tuned model
- Approach 3: Fine-tuned model + RAG

Measure: accuracy, latency, cost, maintenance

### Checkpoint 🚦
- [ ] Successfully fine-tuned models?
- [ ] Understand LoRA/QLoRA?
- [ ] Can create instruction datasets?
- [ ] Know when to fine-tune vs RAG?

---

## 6.5 AI Agents - Exercises

### Must-Build ✓

**ReAct Agent**
- [ ] Implement reasoning-action loop
- [ ] Tool integration
- [ ] Error handling and retries
- [ ] Memory system

### Coding Exercises 💻

**CE5.1**: Build a ReAct agent
```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []

    def run(self, task):
        """
        Thought → Action → Observation loop
        """
        while not self.is_task_complete():
            # 1. Think about next step
            thought = self.think()

            # 2. Decide on action
            action = self.decide_action(thought)

            # 3. Execute action
            observation = self.execute(action)

            # 4. Update memory
            self.update_memory(thought, action, observation)

        return self.final_answer()

    # YOUR CODE HERE for each method
```

**CE5.2**: Tool integration
```python
class Tool:
    def __init__(self, name, description, function):
        self.name = name
        self.description = description
        self.function = function

    def execute(self, *args, **kwargs):
        # YOUR CODE HERE
        pass

# Create tools:
# - Web search
# - Calculator
# - Code execution
# - API calls
# - Database queries
```

### Applied Problems 🎯

**AP5.1**: Build a research assistant
- Takes a research question
- Searches the web
- Reads papers/articles
- Synthesizes information
- Provides cited answer

**AP5.2**: Coding agent
- Generates code based on description
- Executes code in sandbox
- Fixes errors based on output
- Iterates until success

**AP5.3**: Multi-agent system
- Build system with specialized agents
- Coordinator agent
- Task decomposition
- Agent collaboration
- Consensus building

### Checkpoint 🚦
- [ ] Built a working agent?
- [ ] Integrated multiple tools?
- [ ] Handled errors gracefully?
- [ ] Understand agent architectures?

---

## 6.6 Evaluation & Safety - Exercises

### Must-Implement ✓

**Evaluation Framework**
- [ ] Automatic metrics (BLEU, ROUGE, BERTScore)
- [ ] LLM-as-judge
- [ ] Human evaluation setup
- [ ] Benchmark testing

**Safety Measures**
- [ ] Input/output filtering
- [ ] Guardrails
- [ ] Red teaming
- [ ] Bias detection

### Coding Exercises 💻

**CE6.1**: Build evaluation suite
```python
class LLMEvaluator:
    def evaluate_on_benchmark(self, model, benchmark):
        # MMLU, HellaSwag, TruthfulQA, etc.
        pass

    def llm_as_judge(self, responses, criteria):
        # Use GPT-4 to evaluate other models
        pass

    def human_evaluation_interface(self):
        # Simple UI for human raters
        pass
```

**CE6.2**: Safety testing
```python
class SafetyTester:
    def test_jailbreaks(self, model):
        # Test against known jailbreak attempts
        pass

    def test_bias(self, model):
        # Test for demographic biases
        pass

    def test_hallucinations(self, model):
        # Test factual accuracy
        pass
```

---

## Final Phase 6 Checkpoint 🎯

### Comprehensive Project Options

**Option 1: Production RAG System**
- Full-stack application
- Document ingestion pipeline
- Vector database (Pinecone/Qdrant)
- Multiple retrieval strategies
- LLM integration (OpenAI/Anthropic/local)
- Web UI (Streamlit/Gradio)
- Evaluation framework
- Monitoring and logging
- Deployed to production

**Option 2: Fine-Tuned Domain Expert**
- Curate high-quality domain dataset (1000+ examples)
- Fine-tune with LoRA/QLoRA
- Comprehensive evaluation
- Compare with RAG approach
- A/B test in real usage
- Deploy and serve
- Monitor performance

**Option 3: Autonomous Agent System**
- Multi-step planning
- Tool use (search, calculator, code, APIs)
- Memory management
- Error recovery
- Web interface
- Real-world task completion
- Comprehensive testing

**All projects must include:**
- Clean, production-quality code
- Comprehensive tests
- Documentation
- Evaluation metrics
- Deployment
- Cost analysis

**Time estimate**: 6-8 weeks

### Knowledge Self-Assessment

You're ready for Phase 7 if:

**LLMs:**
- [ ] Understand modern LLM architectures
- [ ] Can fine-tune models effectively
- [ ] Master prompt engineering

**RAG:**
- [ ] Built production RAG system
- [ ] Understand embeddings deeply
- [ ] Can evaluate RAG quality

**Agents:**
- [ ] Built working agent with tools
- [ ] Understand ReAct framework
- [ ] Handle errors and edge cases

**Safety:**
- [ ] Know common failure modes
- [ ] Implemented guardrails
- [ ] Can evaluate model safety

**Practical:**
- [ ] Deployed AI application to production
- [ ] Understand cost-performance tradeoffs
- [ ] Can debug LLM issues

### Quick Self-Test (1 hour)

1. Design a RAG system for a specific use case (architecture diagram)
2. Write prompts for chain-of-thought reasoning
3. When to use RAG vs fine-tuning? List 5 criteria
4. Implement a simple ReAct agent (pseudocode)
5. How would you evaluate an AI agent?

All correct + comprehensive project? → Phase 7! 🚀

(Or start Phase 7 in parallel - MLOps concepts apply throughout!)

---

**Note**: This phase overlaps heavily with production concerns. Phase 7 (MLOps) should be started alongside Phase 6 projects.

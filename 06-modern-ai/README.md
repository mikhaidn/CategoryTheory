# Phase 6: Modern AI & Large Language Models

> Understand current state-of-the-art AI systems and applications

## Prerequisites
- Phase 5 complete (especially Transformers)
- Understanding of attention mechanisms
- Comfortable with deep learning frameworks

## Why This Phase Matters

This is where AI is today (2026) and where it's heading. Focus on:
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Fine-tuning and adaptation
- Multimodal models
- AI agents and reasoning

## Learning Path

### 6.1 Large Language Models (LLMs)

#### Architecture Deep Dive
- [ ] **Transformer variants for language**
  - GPT architecture (decoder-only, autoregressive)
  - BERT architecture (encoder-only, bidirectional)
  - T5 architecture (encoder-decoder)
  - Modern trends (decoder-only dominance)
- [ ] **Scaling laws**
  - How performance scales with model size, data, compute
  - Chinchilla paper (optimal compute allocation)
  - Emergent abilities at scale
- [ ] **Tokenization**
  - Byte-pair encoding (BPE)
  - WordPiece, SentencePiece
  - Impact on model performance
  - Multilingual considerations

#### Training LLMs
- [ ] **Pre-training**
  - Next-token prediction objective
  - Data curation and quality
  - Compute requirements (astronomical!)
  - Distributed training at scale
- [ ] **Instruction tuning**
  - Supervised fine-tuning (SFT)
  - Instruction datasets
  - Chain-of-thought prompting in training
- [ ] **Reinforcement Learning from Human Feedback (RLHF)**
  - Reward modeling
  - Proximal Policy Optimization (PPO)
  - Direct Preference Optimization (DPO) - newer alternative
  - Constitutional AI

**Resources**:
- Paper: "Attention Is All You Need" (review from Phase 5)
- Paper: "Language Models are Few-Shot Learners" (GPT-3)
- Paper: "Training language models to follow instructions" (InstructGPT)
- Paper: "Direct Preference Optimization" (DPO)
- Blog: Jay Alammar's illustrated guides (GPT-2, GPT-3, BERT)
- Course: Stanford CS224n latest lectures on LLMs

**Exercises**:
- [ ] Explore different tokenizers (tiktoken, sentencepiece)
- [ ] Fine-tune a small language model (GPT-2, BERT)
- [ ] Analyze scaling law papers and reproduce plots
- [ ] Study open-source LLMs (Llama, Mistral, etc.)

### 6.2 Prompt Engineering & In-Context Learning

#### Prompting Techniques
- [ ] **Zero-shot prompting**
  - Direct instruction
  - Task description design
- [ ] **Few-shot prompting**
  - In-context learning
  - Example selection strategies
  - Order effects
- [ ] **Chain-of-thought (CoT)**
  - Step-by-step reasoning
  - Zero-shot CoT ("Let's think step by step")
  - Self-consistency
- [ ] **Advanced techniques**
  - Tree of Thoughts
  - ReAct (Reasoning + Acting)
  - Self-refinement
  - Constitutional AI prompting

#### Prompt Optimization
- [ ] Automated prompt engineering
- [ ] Prompt templates and versioning
- [ ] Evaluation frameworks
- [ ] LangChain, LlamaIndex for prompt management

**Resources**:
- Guide: "Prompt Engineering Guide" (github.com/dair-ai/Prompt-Engineering-Guide)
- Papers: Chain-of-Thought, ReAct, Tree of Thoughts
- Blog: OpenAI prompt engineering guide
- Tools: LangChain documentation

**Projects**:
- [ ] Build a prompt optimization framework
- [ ] Implement chain-of-thought prompting system
- [ ] Create a prompt template library
- [ ] Benchmark different prompting strategies

### 6.3 Retrieval-Augmented Generation (RAG)

#### Core Concepts
- [ ] **Why RAG?**
  - Hallucination mitigation
  - Grounding in external knowledge
  - Dynamic information updates
  - Domain-specific knowledge
- [ ] **Components**
  - Document ingestion and chunking
  - Embedding models
  - Vector databases
  - Retrieval strategies
  - Generation with retrieved context

#### Embeddings
- [ ] **Text embeddings**
  - Sentence transformers
  - OpenAI embeddings, Cohere embeddings
  - BGE, E5 models
  - Evaluating embedding quality
- [ ] **Embedding techniques**
  - Contrastive learning
  - Fine-tuning embeddings for your domain

#### Vector Databases
- [ ] **Options**
  - Pinecone, Weaviate, Qdrant, Milvus
  - PostgreSQL with pgvector
  - Chroma, FAISS (for smaller scale)
- [ ] **Vector search**
  - Approximate nearest neighbors (ANN)
  - HNSW, IVF algorithms
  - Similarity metrics (cosine, euclidean, dot product)
  - Hybrid search (vector + keyword)

#### Advanced RAG
- [ ] Multi-query retrieval
- [ ] Reranking strategies
- [ ] Contextual compression
- [ ] Hypothetical document embeddings (HyDE)
- [ ] Graph-based RAG
- [ ] Agentic RAG (iterative retrieval)

**Resources**:
- Blog: "RAG from Scratch" by LangChain
- Course: Deeplearning.AI RAG courses
- Tools: LlamaIndex, LangChain, Haystack
- Papers: Latest RAG research (ACL, EMNLP)

**Projects**:
- [ ] Build a RAG system for documentation Q&A
- [ ] Implement multiple retrieval strategies and compare
- [ ] Create a domain-specific chatbot with RAG
- [ ] Build a RAG evaluation framework

### 6.4 Fine-Tuning & Model Adaptation

#### Fine-Tuning Approaches
- [ ] **Full fine-tuning**
  - When to use (sufficient data & compute)
  - Catastrophic forgetting
  - Continual learning strategies
- [ ] **Parameter-efficient fine-tuning (PEFT)**
  - LoRA (Low-Rank Adaptation) - most popular
  - QLoRA (quantized LoRA)
  - Prefix tuning
  - Adapter layers
- [ ] **Instruction tuning**
  - Creating instruction datasets
  - Alpaca, Dolly approaches
  - Self-instruct

#### Practical Considerations
- [ ] When to fine-tune vs RAG vs prompting
- [ ] Data requirements and quality
- [ ] Evaluation strategies
- [ ] Deployment of fine-tuned models

**Resources**:
- Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- Library: Hugging Face PEFT
- Library: Axolotl (fine-tuning framework)
- Course: Deeplearning.AI fine-tuning courses

**Projects**:
- [ ] Fine-tune a model with LoRA
- [ ] Create a custom instruction dataset
- [ ] Compare full fine-tuning vs LoRA vs prompting
- [ ] Fine-tune for a specific domain (legal, medical, code, etc.)

### 6.5 Multimodal Models

#### Vision-Language Models
- [ ] **CLIP** (Contrastive Language-Image Pre-training)
  - Joint embedding space
  - Zero-shot image classification
  - Image search
- [ ] **Vision-Language Models (VLMs)**
  - LLaVA, GPT-4V, Gemini
  - Image understanding and generation
  - Visual instruction following

#### Text-to-Image Models
- [ ] Diffusion models (DALL-E, Stable Diffusion, Midjourney)
- [ ] Prompt engineering for image generation
- [ ] Fine-tuning diffusion models (LoRA for Stable Diffusion)
- [ ] ControlNet (conditional generation)

#### Other Modalities
- [ ] Audio (Whisper for speech recognition)
- [ ] Video understanding
- [ ] Multimodal embeddings

**Resources**:
- Paper: "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Blog: "How DALL-E 2 Works" (Assemblyai)
- Tools: Hugging Face diffusers library
- Papers: LLaVA, Flamingo

**Projects**:
- [ ] Build an image search engine with CLIP
- [ ] Fine-tune Stable Diffusion for specific style
- [ ] Create a visual question answering system
- [ ] Multimodal RAG (text + images)

### 6.6 AI Agents & Reasoning

#### Agent Architectures
- [ ] **ReAct framework**
  - Reasoning and acting interleaved
  - Tool use
  - Observation-thought-action loop
- [ ] **Function calling**
  - Structured outputs from LLMs
  - Tool integration
  - OpenAI function calling, Anthropic tool use
- [ ] **Multi-agent systems**
  - Agent collaboration
  - AutoGen, CrewAI frameworks
  - Debate and consensus

#### Tool Use & Code Execution
- [ ] LLMs calling external APIs
- [ ] Code generation and execution
- [ ] Sandboxed environments
- [ ] Error handling and retry logic

#### Planning & Reasoning
- [ ] Task decomposition
- [ ] Multi-step reasoning
- [ ] Memory systems (short-term, long-term)
- [ ] Self-critique and refinement

**Resources**:
- Paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
- Framework: LangGraph (agent orchestration)
- Framework: AutoGen (Microsoft)
- Blog: "LLM Powered Autonomous Agents" by Lilian Weng

**Projects**:
- [ ] Build a ReAct agent with tool use
- [ ] Create a coding agent (generates and executes code)
- [ ] Multi-agent system for complex tasks
- [ ] Personal assistant agent with memory

### 6.7 Evaluation & Safety

#### Evaluation
- [ ] **Automatic metrics**
  - BLEU, ROUGE (limitations for generation)
  - BERTScore
  - Model-based evaluation (LLM-as-judge)
- [ ] **Human evaluation**
  - Rating scales
  - Preference comparisons
  - Inter-annotator agreement
- [ ] **Benchmarks**
  - MMLU, HellaSwag, TruthfulQA
  - HumanEval (code)
  - Domain-specific benchmarks

#### Safety & Alignment
- [ ] **Risks**
  - Hallucinations
  - Bias and fairness
  - Toxicity and harmful content
  - Jailbreaking and adversarial prompts
  - Privacy concerns
- [ ] **Mitigation strategies**
  - Input/output filtering
  - Constitutional AI
  - Red teaming
  - Guardrails (NeMo Guardrails, Guardrails AI)
- [ ] **Responsible AI practices**
  - Model cards and documentation
  - Testing for bias
  - User consent and transparency
  - Monitoring in production

**Resources**:
- Paper: "Anthropic's Constitutional AI"
- Blog: "Red Teaming Language Models" (various sources)
- Tool: LlamaGuard, Guardrails AI
- Standards: AI incident database

### 6.8 Emerging Topics (2025-2026)

- [ ] **Long context models**
  - 100k+ token contexts (Gemini, Claude)
  - Efficient attention mechanisms
  - Needle-in-haystack evaluation
- [ ] **Mixture of Experts (MoE)**
  - Sparse models
  - Routing mechanisms
  - Mixtral, GPT-4 (rumored)
- [ ] **Small language models (SLMs)**
  - Phi-3, Gemma, Llama-3-8B
  - On-device AI
  - Distillation techniques
- [ ] **Multimodal agents**
  - Operating computer interfaces
  - Robotics applications
- [ ] **Reasoning models**
  - OpenAI o1, o3
  - Test-time compute scaling
  - RL for reasoning

## Key Projects

### Project 1: Production RAG System
Full-stack RAG application:
- Document ingestion pipeline
- Vector database setup
- Multiple retrieval strategies
- LLM integration
- Web UI
- Evaluation framework
- Monitoring (Phase 7 skills!)
- Deploy to production

### Project 2: Fine-Tuned Domain Expert
- Curate domain-specific dataset
- Fine-tune with LoRA
- Compare with RAG approach
- Evaluate rigorously
- Deploy and serve
- A/B test against base model

### Project 3: Autonomous Agent
Build an agent that:
- Breaks down complex tasks
- Uses multiple tools (search, calculator, code execution, APIs)
- Maintains conversation memory
- Self-critiques and refines
- Web interface for interaction
- Production-ready (error handling, logging, monitoring)

### Project 4: Multimodal Application
- Image + text understanding
- Visual question answering or image search
- Integration with LLM for reasoning
- User-friendly interface
- Deploy

## Self-Assessment

Before considering yourself proficient in modern AI:
- [ ] Understand LLM architectures and training processes
- [ ] Can build and evaluate RAG systems
- [ ] Comfortable fine-tuning models with PEFT
- [ ] Have built at least one AI agent
- [ ] Understand prompt engineering deeply
- [ ] Familiar with multimodal models
- [ ] Know how to evaluate and ensure safety
- [ ] Can deploy AI applications to production (with Phase 7)
- [ ] Stay updated with latest research

## Estimated Timeline

- **LLMs fundamentals**: 3-4 weeks
- **Prompt engineering**: 2 weeks
- **RAG systems**: 3-4 weeks
- **Fine-tuning**: 2-3 weeks
- **Multimodal models**: 2-3 weeks
- **AI agents**: 3-4 weeks
- **Evaluation & safety**: 2 weeks
- **Emerging topics**: Ongoing
- **Projects**: 6-8 weeks

**Total**: 23-32 weeks

Note: This phase overlaps heavily with Phase 7 (MLOps). Many projects will use both skill sets.

## Staying Current

AI moves fast. To stay updated:
- [ ] Follow key researchers on Twitter/X
- [ ] Read papers on arXiv (cs.CL, cs.AI, cs.LG)
- [ ] Subscribe to newsletters (The Batch, TLDR AI, Last Week in AI)
- [ ] Watch conference talks (NeurIPS, ICML, ICLR, ACL, EMNLP)
- [ ] Experiment with new models (Hugging Face)
- [ ] Participate in communities (r/MachineLearning, Eleuther AI Discord)

## Resources

### Papers (Essential Reading)
- Attention Is All You Need (Transformers)
- BERT, GPT-2, GPT-3 papers
- InstructGPT (RLHF)
- Llama 2 (open-source LLM)
- Constitutional AI
- LoRA, QLoRA
- RAG papers

### Courses
- Stanford CS224n (NLP with Deep Learning)
- Stanford CS25 (Transformers United)
- Deeplearning.AI short courses (LangChain, RAG, fine-tuning)

### Books
- "Build a Large Language Model (From Scratch)" by Sebastian Raschka (2024)
- "Natural Language Processing with Transformers" by Tunstall et al.

### Tools & Frameworks
- Hugging Face Transformers
- LangChain / LlamaIndex
- vLLM (efficient inference)
- Ollama (local LLMs)
- OpenAI API, Anthropic API
- Vector databases (Pinecone, Qdrant, etc.)

## Next Steps

You're now at the frontier! Continue to:
1. Build real applications (combine Phase 6 + Phase 7)
2. Read cutting-edge research
3. Contribute to open-source AI projects
4. Consider specializing in a domain (AI for healthcare, finance, education, etc.)
5. Share your knowledge (blog, teach, speak)

Your unique combination of SWE skills + AI knowledge is extremely valuable!

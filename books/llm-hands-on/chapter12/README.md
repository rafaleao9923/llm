# Chapter 12 - Fine-tuning Generation Models

## 🎯 **Three-Step LLM Training Pipeline**

### **Stage 1: Language Modeling (Pretraining)**
- **Self-supervised learning**: Predict next token without labels
- **Massive datasets**: Trained on large text corpora (Wikipedia, books)
- **Base model output**: Foundation model with linguistic knowledge
- **Problem**: Cannot follow instructions, just completes text

### **Stage 2: Supervised Fine-Tuning (SFT)**
- **Instruction following**: Adapt base model to follow user commands
- **Labeled data**: Question-response pairs for training
- **Next-token prediction**: Based on user input instead of raw text
- **Result**: Chat-capable model that responds to instructions

### **Stage 3: Preference Tuning (Alignment)**
- **Human preference alignment**: Match expected AI behavior patterns
- **Quality improvement**: Better responses aligned with human values
- **Safety considerations**: Incorporate AI safety and ethical guidelines
- **Output**: Production-ready aligned model

```mermaid
flowchart TD
    A[Raw Text Data] --> B[Language Modeling]
    B --> C[Base Model]
    C --> D[Supervised Fine-Tuning]
    D --> E[Instruction Model]
    E --> F[Preference Tuning]
    F --> G[Aligned Model]
    
    H[Instruction Data] --> D
    I[Preference Data] --> F
    
    J[Problem: Text Completion] --> C
    K[Solution: Instruction Following] --> E
    L[Solution: Human Alignment] --> G
```

## 🔧 **Parameter-Efficient Fine-Tuning (PEFT)**

### **Full Fine-Tuning Challenges:**
- **High computational cost**: Update all model parameters
- **Memory requirements**: Significant VRAM needed for large models
- **Storage overhead**: Multiple model copies for different tasks
- **Training time**: Slow convergence for large models

### **Adapters Architecture:**
- **Modular components**: Small trainable layers inserted in Transformer blocks
- **Frozen base model**: Original weights remain unchanged
- **Task specialization**: Different adapters for different tasks
- **Performance**: 3.6% parameters achieve 99.6% of full fine-tuning performance

```mermaid
graph TD
    A[Transformer Block] --> B[Attention Layer]
    B --> C[Adapter 1]
    C --> D[Feed Forward]
    D --> E[Adapter 2]
    E --> F[Output]
    
    G[Base Model] --> H[Frozen Weights]
    I[Adapter Module] --> J[Trainable Weights]
    
    K[Medical Adapter] --> L[Medical Classification]
    M[NER Adapter] --> N[Named Entity Recognition]
```

### **Low-Rank Adaptation (LoRA):**
- **Matrix decomposition**: Large matrices → two smaller matrices
- **Efficiency gain**: 100 parameters → 20 parameters (5x reduction)
- **Frozen base**: Original model weights remain unchanged
- **Intrinsic dimensionality**: Language models have low-rank structure
- **Flexible targeting**: Choose specific layers to fine-tune

### **LoRA Mathematics:**
- **Original matrix**: 10×10 = 100 parameters
- **Decomposed matrices**: (10×2) + (2×10) = 40 parameters
- **Memory savings**: 60% reduction in trainable parameters
- **GPT-3 example**: 150M parameters → 197K parameters per block

```mermaid
mindmap
  root((LoRA Process))
    Matrix Decomposition
      Large Weight Matrix
      Two Smaller Matrices
      Rank Approximation
      Parameter Reduction
    Training Process
      Frozen Base Model
      Trainable LoRA Weights
      Gradient Updates
      Weight Combination
    Benefits
      Memory Efficiency
      Faster Training
      Task Modularity
      Storage Savings
```

## ⚡ **QLoRA: Quantized LoRA**

### **Quantization Strategy:**
- **Bit reduction**: 16-bit → 4-bit representation
- **Memory compression**: ~75% memory reduction
- **Blockwise quantization**: Group similar weights for accurate representation
- **Distribution-aware mapping**: Account for weight frequency distribution

### **QLoRA Implementation Steps:**
- **4-bit quantization**: Use normalized float representation (nf4)
- **Double quantization**: Apply nested quantization for further compression
- **LoRA adaptation**: Fine-tune low-rank matrices on quantized base
- **Merge process**: Combine LoRA weights with base model after training

```mermaid
flowchart LR
    A[16-bit Model] --> B[4-bit Quantization]
    B --> C[LoRA Decomposition]
    C --> D[Fine-tuning]
    D --> E[Weight Merging]
    E --> F[Final Model]
    
    G[Memory: 4GB] --> H[Memory: 1GB]
    I[Trainable: All Params] --> J[Trainable: LoRA Only]
```

## 📊 **Evaluation Methodologies**

### **Word-Level Metrics:**
- **Perplexity**: Measures prediction confidence (lower is better)
- **ROUGE**: Text summarization quality scoring
- **BLEU**: Machine translation evaluation
- **BERTScore**: Semantic similarity using BERT embeddings

### **Benchmark Evaluation:**
- **MMLU**: 57 multitask language understanding tasks
- **GLUE**: General language understanding evaluation
- **TruthfulQA**: Measures factual accuracy and truthfulness
- **GSM8k**: Grade-school math word problems
- **HumanEval**: Programming problem evaluation

### **Automated vs Human Evaluation:**
- **LLM-as-a-judge**: Use separate LLM to evaluate outputs
- **Pairwise comparison**: Compare two model outputs side-by-side
- **Chatbot Arena**: Human voting on anonymous model outputs
- **Elo rating system**: Chess-like ranking based on win rates

```mermaid
graph TD
    A[Evaluation Methods] --> B[Word-Level Metrics]
    A --> C[Benchmarks]
    A --> D[Human Evaluation]
    
    B --> E[Perplexity]
    B --> F[ROUGE/BLEU]
    B --> G[BERTScore]
    
    C --> H[MMLU]
    C --> I[GSM8k]
    C --> J[HumanEval]
    
    D --> K[Chatbot Arena]
    D --> L[Expert Review]
    D --> M[Domain Testing]
```

## 🎯 **Preference Tuning & Alignment**

### **Reinforcement Learning from Human Feedback (RLHF):**
- **Reward model training**: Learn to score generation quality
- **PPO optimization**: Proximal Policy Optimization for alignment
- **Three-stage process**: Collect preferences → Train reward model → Fine-tune LLM
- **Multiple rewards**: Separate models for helpfulness and safety

### **Direct Preference Optimization (DPO):**
- **No reward model**: LLM judges its own output quality
- **Reference comparison**: Compare frozen vs trainable model outputs
- **Token-level optimization**: Calculate probability shifts per token
- **Stability advantage**: More stable than PPO, simpler implementation

### **Preference Data Structure:**
- **Prompt + Accepted + Rejected**: Training triplets for alignment
- **Human labeling**: Evaluators choose preferred responses
- **Quality scoring**: Rate generations on helpfulness/safety
- **Comparative ranking**: Better vs worse rather than absolute scores

```mermaid
flowchart TD
    A[Instruction Model] --> B{Preference Method}
    B -->|RLHF| C[Train Reward Model]
    B -->|DPO| D[Direct Optimization]
    
    C --> E[PPO Fine-tuning]
    D --> F[Reference Comparison]
    
    E --> G[Aligned Model]
    F --> G
    
    H[Preference Data] --> I[Prompt]
    H --> J[Accepted Response]
    H --> K[Rejected Response]
    
    I --> C
    J --> C
    K --> C
```

## 🔄 **Complete Fine-Tuning Workflow**

### **SFT Implementation:**
- **Data templating**: Format conversations with chat templates
- **Model quantization**: 4-bit compression with BitsAndBytesConfig
- **LoRA configuration**: Set rank, alpha, target modules
- **Training setup**: SFTTrainer with optimized hyperparameters

### **DPO Implementation:**
- **Alignment data**: Prompt-accepted-rejected triplets
- **Model preparation**: Load SFT model + quantization + LoRA
- **DPO training**: DPOTrainer with beta parameter tuning
- **Adapter merging**: Combine SFT and DPO adapters sequentially

### **Key Hyperparameters:**
- **LoRA rank (r)**: 4-64, controls adaptation capacity
- **LoRA alpha**: 2×rank, balances original vs new knowledge
- **Learning rate**: 2e-4 for SFT, 1e-5 for DPO
- **Beta (DPO)**: 0.1, controls preference strength

```mermaid
mindmap
  root((Complete Pipeline))
    Data Preparation
      Instruction Templates
      Chat Formatting
      Preference Pairs
      Quality Filtering
    Model Setup
      Base Model Loading
      Quantization Config
      LoRA Configuration
      Training Arguments
    Training Process
      SFT Phase
      Model Evaluation
      DPO Phase
      Adapter Merging
    Quality Assurance
      Benchmark Testing
      Human Evaluation
      Safety Alignment
      Performance Metrics
```
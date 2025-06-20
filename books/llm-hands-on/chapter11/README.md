# Chapter 11 - Fine-tuning Representation Models for Classification

## 🎯 **Core Concept**
- **Fine-tuning vs Frozen Models**: Update pretrained model weights during training instead of keeping them frozen
- **Joint Learning**: Both representation model and classification head learn together
- **Better Performance**: Fine-tuning typically outperforms frozen pretrained models
- **Task-Specific Adaptation**: Models adapt to specific classification tasks

## 🏗️ **Supervised Classification Architecture**

### **Fine-Tuning Process:**
- **BERT + Classification Head**: Pretrained BERT with additional neural network layer
- **End-to-End Training**: Both components updated simultaneously through backpropagation
- **Shared Learning**: Classification head and BERT learn complementary representations
- **Performance Boost**: F1 score improvement from 0.80 (frozen) to 0.85 (fine-tuned)

```mermaid
flowchart TD
    A[Input Text] --> B[BERT Model]
    B --> C[Classification Head]
    C --> D[Predictions]
    E[Loss Function] --> F[Backpropagation]
    F --> G[Update BERT Weights]
    F --> H[Update Classifier Weights]
    G --> B
    H --> C
```

### **Layer Freezing Strategies:**
- **Full Fine-tuning**: All layers trainable (best performance)
- **Partial Freezing**: Freeze early layers, train later ones
- **Classification Head Only**: Freeze BERT entirely (fastest but lowest performance)
- **Performance Trade-off**: More trainable layers = better results but slower training

```mermaid
graph TD
    A[BERT Architecture] --> B[Embedding Layer]
    A --> C[Encoder Blocks 0-9]
    A --> D[Encoder Blocks 10-11]
    A --> E[Classification Head]
    
    F{Freezing Strategy} --> G[Freeze All BERT]
    F --> H[Freeze Blocks 0-9]
    F --> I[Train Everything]
    
    G --> J[F1: 0.63]
    H --> K[F1: 0.80]
    I --> L[F1: 0.85]
```

## 🎯 **Few-Shot Classification with SetFit**

### **SetFit Three-Step Process:**
- **Step 1**: Generate positive/negative sentence pairs from labeled examples
- **Step 2**: Fine-tune SentenceTransformer model using contrastive learning
- **Step 3**: Train classifier on fine-tuned embeddings

### **Data Generation Strategy:**
- **Positive Pairs**: Sentences within same class (similar)
- **Negative Pairs**: Sentences from different classes (dissimilar)
- **Pair Multiplication**: 16 samples per class → 1,280 training pairs
- **Contrastive Learning**: Embeddings learn class-relevant similarities

```mermaid
mindmap
  root((SetFit Process))
    Step 1: Data Generation
      In-class Pairs (Positive)
      Out-class Pairs (Negative)
      16 samples → 1,280 pairs
    Step 2: Fine-tune Embeddings
      SentenceTransformer Model
      Contrastive Learning
      Task-specific Representations
    Step 3: Train Classifier
      Logistic Regression
      Fine-tuned Embeddings
      Classification Head
    Results
      32 labeled examples
      F1 Score: 0.85
      Impressive Performance
```

### **SetFit Advantages:**
- **Data Efficiency**: High performance with minimal labeled data
- **Quick Training**: Faster than full model fine-tuning
- **Flexible Architecture**: Works with any SentenceTransformer model
- **Zero-shot Capability**: Can generate synthetic examples from label names

## 🔄 **Continued Pretraining with MLM**

### **Three-Step Training Pipeline:**
- **Step 1**: General pretraining (already done - BERT)
- **Step 2**: Domain-specific continued pretraining (MLM)
- **Step 3**: Task-specific fine-tuning (classification)

### **Masked Language Modeling Process:**
- **Token Masking**: Randomly mask 15% of tokens in sentences
- **Whole-Word Masking**: Alternative approach masking entire words
- **Domain Adaptation**: Update representations for domain-specific vocabulary
- **Vocabulary Specialization**: Model learns domain-specific word relationships

```mermaid
flowchart LR
    A[General BERT] --> B[Continue Pretraining]
    B --> C[Domain-Specific BERT]
    C --> D[Fine-tune for Classification]
    D --> E[Task-Specific Model]
    
    F[Movie Reviews MLM] --> G["horrible [MASK]"]
    G --> H[movie, film, comedy]
    
    I[General MLM] --> J["horrible [MASK]"]
    J --> K[idea, dream, day]
```

### **MLM Benefits:**
- **Domain Vocabulary**: Better understanding of domain-specific terms
- **Improved Performance**: Enhanced classification accuracy on specialized text
- **Organizational Adaptation**: Can adapt to company-specific language/jargon
- **Incremental Learning**: Build on existing pretrained knowledge

## 🏷️ **Named-Entity Recognition (NER)**

### **Token-Level Classification:**
- **Individual Token Prediction**: Classify each token rather than entire documents
- **Entity Categories**: Person (PER), Organization (ORG), Location (LOC), Miscellaneous (MISC)
- **BIO Tagging**: B (Beginning), I (Inside), O (Outside) entity labeling scheme
- **Phrase Recognition**: Identify multi-token entities as cohesive units

### **Data Preprocessing Challenges:**
- **Word-to-Token Alignment**: Labels at word level, predictions at token level
- **Subword Tokenization**: Single words split into multiple tokens
- **Label Propagation**: B-PER → I-PER for token continuations
- **Special Token Handling**: [CLS] and [SEP] tokens get -100 labels

(using - for " ... to fix error mermaid)

```mermaid
graph TD
    A[-Dean Palmer-] --> B[Word-Level Labels]
    B --> C["B-PER I-PER"]
    
    D[Tokenization] --> E[-Dean-, -Palmer-]
    E --> F[Token Alignment]
    F --> G[-B-PER-, -ER-]
    
    H["homer"] --> I[Subword Split]
    I --> J[-home-, -##r-]
    J --> K[Label Alignment]
    K --> L[-O-, -O-]
```

### **NER Architecture Differences:**
- **Token Classification Model**: `AutoModelForTokenClassification` instead of sequence classification
- **Per-Token Predictions**: Output predictions for each input token
- **Sequence Evaluation**: Use `seqeval` for entity-level F1 scoring
- **DataCollator**: `DataCollatorForTokenClassification` for proper batching

## ⚡ **Performance Optimization Strategies**

### **Training Efficiency:**
- **Layer Freezing**: Trade performance for speed by freezing early layers
- **Batch Size Tuning**: Balance memory usage and training stability
- **Learning Rate Selection**: Critical for both convergence and performance
- **Epoch Management**: Monitor for overfitting vs underfitting

### **Model Selection Criteria:**
- **Task Complexity**: Simple tasks may not need full fine-tuning
- **Data Availability**: Few-shot methods for limited labeled data
- **Domain Specificity**: Continued pretraining for specialized vocabularies
- **Computational Resources**: Balance performance gains with training costs

```mermaid
flowchart TD
    A{Classification Task} --> B{Data Available?}
    B -->|Limited| C[SetFit Few-Shot]
    B -->|Abundant| D{Domain-Specific?}
    D -->|Yes| E[Continued Pretraining + Fine-tuning]
    D -->|No| F[Direct Fine-tuning]
    
    G{Computational Budget?} --> H{Performance Requirements?}
    H -->|High| I[Full Fine-tuning]
    H -->|Medium| J[Partial Layer Freezing]
    H -->|Low| K[Classification Head Only]
```
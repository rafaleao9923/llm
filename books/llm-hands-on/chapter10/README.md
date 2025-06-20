# Chapter 10 - Creating Text Embedding Models

## 🎯 **Embedding Models Overview**
- **Foundation technology**: Powers classification, search, clustering, RAG, and memory systems
- **Core purpose**: Convert unstructured text to numerical representations (embeddings)
- **Semantic capture**: Represent meaning and context in high-dimensional space
- **Similarity principle**: Similar documents → closer vectors, dissimilar → farther apart

```mermaid
flowchart TD
    A[Unstructured Text] --> B[Embedding Model]
    B --> C[Numerical Vectors]
    C --> D[Semantic Similarity]
    
    E[Document 1] --> F[Vector 1]
    G[Document 2] --> H[Vector 2]
    F --> I[Distance Calculation]
    H --> I
    I --> J[Similarity Score]
```

## 🔄 **Contrastive Learning Foundation**

### **Core Principle:**
- **Similarity learning**: Train models to recognize what makes documents similar/different
- **Comparative understanding**: Learn through contrasts (Why P and not Q?)
- **Feature discovery**: Identify distinctive characteristics through comparison
- **Historical foundation**: Word2vec was early contrastive learning success

### **Contrastive Explanation:**
- **Context importance**: Understanding requires alternatives and contrasts
- **Example**: "Why dog not cat?" → Learn distinguishing features
- **Practical benefit**: More informative than isolated feature learning
- **Training efficiency**: Learn from both positive and negative examples

```mermaid
graph TD
    A[Similar Documents] --> B[Positive Pairs]
    C[Dissimilar Documents] --> D[Negative Pairs]
    
    B --> E[Minimize Distance]
    D --> F[Maximize Distance]
    
    E --> G[Embedding Model]
    F --> G
    G --> H[Semantic Representations]
```

## 🏗️ **SBERT Architecture**

### **Problem with Original BERT:**
- **Cross-encoder overhead**: n×(n-1)/2 computations for similarity
- **No embeddings**: Outputs similarity scores, not reusable vectors
- **Computational explosion**: 49,995,000 operations for 10,000 sentences
- **Poor averaged embeddings**: Simple averaging worse than GloVe

### **SBERT Solution:**
- **Siamese architecture**: Two identical BERT models with shared weights
- **Mean pooling**: Average token embeddings for fixed-size sentence vectors
- **Bi-encoder design**: Generate embeddings that can be compared directly
- **Efficiency**: Single forward pass per sentence, reusable embeddings

```mermaid
graph LR
    A[Sentence 1] --> B[BERT Encoder]
    C[Sentence 2] --> D[BERT Encoder]
    
    B --> E[Mean Pooling]
    D --> F[Mean Pooling]
    
    E --> G[Embedding 1]
    F --> H[Embedding 2]
    
    G --> I[Similarity Loss]
    H --> I
    
    J[Shared Weights] --> B
    J --> D
```

## 📊 **Training Data: NLI Datasets**

### **Natural Language Inference:**
- **Three relationships**: Entailment, contradiction, neutral
- **Contrastive pairs**: Entailment = positive, contradiction = negative
- **MNLI corpus**: 392,702 annotated sentence pairs from GLUE benchmark
- **Quality data**: Human-labeled relationship examples

### **Data Preparation:**
```python
# Convert NLI labels to contrastive pairs
# 0 = entailment (positive), 1 = neutral, 2 = contradiction (negative)
mapping = {0: 1, 1: 0, 2: 0}  # entailment=1, others=0

train_dataset = Dataset.from_dict({
    "sentence1": dataset["premise"],
    "sentence2": dataset["hypothesis"], 
    "label": [float(mapping[label]) for label in dataset["label"]]
})
```

## 🔧 **Model Training Process**

### **Base Model Selection:**
- **BERT base uncased**: Great starting point for embeddings
- **MPNet base**: Often better performance for sentence embeddings
- **All layers trainable**: Don't freeze layers for best performance
- **Pretrained foundation**: Leverage existing language understanding

### **Training Pipeline:**
```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer

# 1. Load base model
embedding_model = SentenceTransformer('bert-base-uncased')

# 2. Define loss function
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

# 3. Train with SentenceTransformerTrainer
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
```

## 📏 **Evaluation Framework**

### **STSB Benchmark:**
- **Semantic Textual Similarity**: Human-labeled sentence pairs
- **Similarity scores**: 1-5 scale normalized to 0-1
- **Evaluation metrics**: Pearson/Spearman correlation with cosine similarity
- **Fast evaluation**: Quick performance assessment during training

### **MTEB Benchmark:**
- **Comprehensive evaluation**: 58 datasets across 8 tasks and 112 languages
- **Industry standard**: Public leaderboard for model comparison
- **Multiple metrics**: Accuracy, F1, evaluation time
- **Production relevance**: Includes latency measurements

```mermaid
graph TD
    A[Trained Model] --> B[STSB Evaluation]
    A --> C[MTEB Evaluation]
    
    B --> D[Pearson Correlation]
    B --> E[Spearman Correlation]
    
    C --> F[Classification Tasks]
    C --> G[Clustering Tasks]
    C --> H[Retrieval Tasks]
    C --> I[Speed Metrics]
```

## 🎯 **Loss Functions Comparison**

### **Softmax Loss:**
- **Classification approach**: Treat similarity as classification problem
- **Baseline performance**: Historical method, not recommended
- **Example performance**: 0.59 Pearson correlation
- **Limited effectiveness**: Better alternatives available

### **Cosine Similarity Loss:**
- **Intuitive approach**: Directly optimize cosine similarity scores
- **Semantic similarity**: Works well for similarity tasks
- **Continuous scores**: Handles similarity degrees (0-1 range)
- **Performance**: 0.72 Pearson correlation (significant improvement)

### **Multiple Negatives Ranking (MNR) Loss:**
- **State-of-the-art**: Best performing loss function
- **In-batch negatives**: Use other examples in batch as negatives
- **InfoNCE/NTXentLoss**: Also known by these names
- **Performance**: 0.80 Pearson correlation (best results)

```mermaid
graph TD
    A[Anchor Sentence] --> B[Positive Example]
    A --> C[Negative Examples]
    
    D[Batch Processing] --> E[In-Batch Negatives]
    E --> F[Cross-Entropy Loss]
    
    G[Larger Batches] --> H[More Negatives]
    H --> I[Harder Task]
    I --> J[Better Performance]
```

### **Negative Examples Quality:**
- **Easy negatives**: Random sampling from different domains
- **Semi-hard negatives**: Similar topics but wrong answers
- **Hard negatives**: Very similar but incorrect answers
- **Performance impact**: Hard negatives → better model quality

## 🚀 **Fine-tuning Strategies**

### **Supervised Fine-tuning:**
- **Pretrained base**: Start with existing embedding model
- **Domain adaptation**: Adapt to specific use cases
- **Less data needed**: Leverage existing knowledge
- **Example performance**: 0.85 Pearson correlation

```python
# Fine-tune pretrained model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
```

### **Augmented SBERT:**
- **Data augmentation**: Generate more training data from limited labels
- **Four-step process**: Cross-encoder training → data generation → labeling → bi-encoder training
- **Small data solution**: Work with thousands instead of millions of examples
- **Quality preservation**: 0.71 score with only 20% of original data

#### Augmented SBERT Pipeline:
```mermaid
flowchart TD
    A[Small Gold Dataset] --> B[Train Cross-Encoder]
    C[Unlabeled Data] --> D[Generate Pairs]
    B --> E[Label with Cross-Encoder]
    D --> E
    E --> F[Silver Dataset]
    A --> G[Combine Gold + Silver]
    F --> G
    G --> H[Train Bi-Encoder]
```

## 🔬 **Unsupervised Learning: TSDAE**

### **Transformer-based Sequential Denoising Auto-Encoder:**
- **No labels needed**: Pure unsupervised learning approach
- **Denoising task**: Remove words, reconstruct original sentence
- **Encoder-decoder**: Encoder generates embeddings, decoder reconstructs
- **Training objective**: Better embeddings → better reconstruction

### **TSDAE Process:**
1. **Add noise**: Randomly remove words from input sentence
2. **Encode**: Generate sentence embedding from damaged sentence
3. **Decode**: Reconstruct original sentence from embedding
4. **Optimize**: Improve embeddings to enable better reconstruction

```mermaid
graph LR
    A[Original Sentence] --> B[Add Noise]
    B --> C[Damaged Sentence]
    C --> D[Encoder]
    D --> E[Sentence Embedding]
    E --> F[Decoder]
    F --> G[Reconstructed Sentence]
    A --> H[Loss Calculation]
    G --> H
```

### **Implementation Details:**
- **CLS pooling**: Use [CLS] token instead of mean pooling
- **Tied weights**: Share encoder/decoder embedding parameters
- **Performance**: 0.70 Pearson correlation (impressive for unsupervised)
- **Domain adaptation**: Excellent for adapting to new domains

## 🎯 **Domain Adaptation**

### **Adaptive Pretraining:**
- **Target domain focus**: Train on domain-specific unlabeled data
- **Two-stage process**: Unsupervised pretraining → supervised fine-tuning
- **Out-domain tolerance**: Can fine-tune with data from different domains
- **Cost effective**: Leverage unlabeled data for domain adaptation

### **Domain Adaptation Pipeline:**
```mermaid
flowchart TD
    A[Target Domain Text] --> B[TSDAE Pretraining]
    B --> C[Domain-Adapted Model]
    D[Labeled Data] --> E[Supervised Fine-tuning]
    C --> E
    E --> F[Final Domain-Specific Model]
```

## 🛠️ **Practical Implementation**

### **Data Requirements:**
- **Quality over quantity**: High-quality pairs more important than volume
- **Hard negatives**: Significantly improve performance
- **Batch size effects**: Larger batches better for MNR loss
- **Domain relevance**: Training data should match target use case

### **Training Tips:**
- **Mixed precision**: Use fp16 for memory efficiency
- **Warmup steps**: Gradually increase learning rate
- **Evaluation frequency**: Monitor performance during training
- **Early stopping**: Save best model based on validation performance

### **Performance Comparison:**

| Method | Training Data | Performance | Notes |
|--------|---------------|-------------|-------|
| Softmax Loss | 50K MNLI | 0.59 | Baseline approach |
| Cosine Loss | 50K MNLI | 0.72 | Good for similarity tasks |
| MNR Loss | 16K entailment | 0.80 | State-of-the-art |
| Fine-tuned | 50K MNLI | 0.85 | Pretrained base |
| Augmented SBERT | 10K + 40K silver | 0.71 | Small data solution |
| TSDAE | 50K unlabeled | 0.70 | Unsupervised |

## 🔑 **Key Implementation Patterns**

### **Model Architecture:**
```python
# Create custom embedding model
word_embedding_model = models.Transformer("bert-base-uncased")
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), 
    "mean"  # or "cls" for TSDAE
)
embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

### **Loss Function Selection:**
```python
# Choose appropriate loss based on data
if similarity_scores_available:
    loss = losses.CosineSimilarityLoss(model=embedding_model)
elif pairs_available:
    loss = losses.MultipleNegativesRankingLoss(model=embedding_model)
elif unsupervised:
    loss = losses.DenoisingAutoEncoderLoss(embedding_model, tie_encoder_decoder=True)
```

## 🎯 **Chapter Summary**

### **Embedding Model Creation:**
- **Contrastive learning**: Foundation technique for similarity learning
- **Architecture choices**: SBERT provides efficient embedding generation
- **Loss functions**: MNR loss typically performs best
- **Data quality**: Hard negatives significantly improve performance

### **Training Strategies:**
- **From scratch**: Use large datasets with quality contrastive pairs
- **Fine-tuning**: Leverage pretrained models for efficiency
- **Data augmentation**: Augmented SBERT for limited labeled data
- **Unsupervised**: TSDAE for domain adaptation without labels

### **Best Practices:**
- **Evaluation**: Use comprehensive benchmarks like MTEB
- **Performance monitoring**: Track multiple metrics during training
- **Resource management**: Clear VRAM between experiments
- **Domain adaptation**: Combine unsupervised pretraining with supervised fine-tuning
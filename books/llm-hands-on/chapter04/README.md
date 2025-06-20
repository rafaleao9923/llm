# Chapter 4 - Text Classification

## 📊 **Text Classification Overview**
- **Core task**: Assign labels/classes to input text automatically
- **Wide applications**: Sentiment analysis, intent detection, entity extraction, language detection
- **Two main approaches**: Representation models vs Generative models
- **Pretrained focus**: Leverage existing trained models rather than training from scratch

```mermaid
flowchart TD
    A[Text Input] --> B{Model Type}
    B -->|Representation| C[Task-Specific Model]
    B -->|Representation| D[Embedding Model]
    B -->|Generative| E[Encoder-Decoder Model]
    B -->|Generative| F[Decoder-Only Model]
    C --> G[Direct Classification]
    D --> H[Features + Classifier]
    E --> I[Text-to-Text Generation]
    F --> J[Prompt-Based Classification]
```

## 🎬 **Dataset: Movie Review Sentiment**
- **Rotten Tomatoes dataset**: 5,331 positive + 5,331 negative movie reviews
- **Binary classification**: Positive (1) vs Negative (0) sentiment
- **Data splits**: Train (8,530), validation (1,066), test (1,066)
- **Hugging Face Hub**: Easy loading with datasets package

## 🏗️ **Model Architecture Approaches**

### **🔹 Representation Models:**
- **Foundation**: BERT-like encoder-only architectures
- **Two flavors**: Task-specific models vs Embedding models
- **Size advantage**: Significantly smaller than generative models
- **Performance**: Excel at classification and understanding tasks

### **🔹 Task-Specific Models:**
- **Direct classification**: Model outputs class probabilities directly
- **Pretraining**: Fine-tuned on specific tasks (e.g., sentiment analysis)
- **Ready-to-use**: No additional training required
- **Example**: Twitter-RoBERTa-base for sentiment analysis

### **🔹 Embedding Models:**
- **Feature extraction**: Generate general-purpose vector representations
- **Two-step process**: Embeddings → Classifier training
- **Flexibility**: Same embeddings usable for multiple tasks
- **Example**: sentence-transformers/all-mpnet-base-v2

```mermaid
mindmap
  root((Text Classification))
    Representation Models
      Task-Specific
        Direct output
        Pre-fine-tuned
        Domain specific
      Embedding
        Feature extraction
        General purpose
        Two-step process
    Generative Models
      Encoder-Decoder
        T5/Flan-T5
        Text-to-text
        Instruction following
      Decoder-Only
        GPT models
        Prompt engineering
        API access
```

## 🎯 **Model Selection Strategy**

### **BERT Family Evolution:**
- **BERT (2018)**: Original encoder-only foundation
- **RoBERTa (2019)**: Robustly optimized BERT pretraining
- **DistilBERT (2019)**: Smaller, faster, lighter version
- **ALBERT (2019)**: Lite BERT with parameter sharing
- **DeBERTa (2020)**: Disentangled attention improvements

### **Selection Criteria:**
- **Language compatibility**: Target language support
- **Model size**: Inference speed vs accuracy tradeoff
- **Domain relevance**: Training data similarity to use case
- **Performance benchmarks**: MTEB leaderboard for embeddings

## 🔧 **Task-Specific Model Implementation**

### **Pipeline Setup:**
- **Transformers pipeline**: Simplified model loading and inference
- **Tokenizer integration**: Automatic text-to-token conversion
- **GPU utilization**: CUDA device mapping for speed
- **Batch processing**: Efficient handling of multiple inputs

### **Token Handling Benefits:**
- **Unknown word robustness**: Subword tokenization handles unseen words
- **Compositional understanding**: Combine subword pieces for meaning
- **Vocabulary efficiency**: Reuse common prefixes/suffixes

### **Performance Results:**
- **F1 Score**: 0.80 weighted average
- **Cross-domain**: Twitter model → movie reviews (good generalization)
- **Evaluation metrics**: Precision, recall, accuracy, F1 score

```mermaid
graph TD
    A[Movie Review Text] --> B[Tokenizer]
    B --> C[Token IDs]
    C --> D[Twitter-RoBERTa Model]
    D --> E[Class Probabilities]
    E --> F[Prediction: Pos/Neg]
    F --> G[F1: 0.80]
```

## 📈 **Classification Metrics Explained**

### **Confusion Matrix Foundation:**
- **True Positive (TP)**: Correctly predicted positive
- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive
- **False Negative (FN)**: Incorrectly predicted negative

### **Key Metrics:**
- **Precision**: TP/(TP+FP) - Accuracy of positive predictions
- **Recall**: TP/(TP+FN) - Coverage of actual positives
- **F1 Score**: 2×(Precision×Recall)/(Precision+Recall) - Balanced measure
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Overall correctness

## 🎯 **Embedding-Based Classification**

### **Two-Step Approach:**
1. **Feature extraction**: Text → embeddings (frozen model)
2. **Classification**: Embeddings → labels (trainable classifier)

### **Benefits:**
- **Cost efficiency**: No GPU needed for classifier training
- **Model flexibility**: Same embeddings for multiple tasks
- **Compute distribution**: Heavy lifting on GPU, training on CPU

### **Implementation Process:**
```python
# Step 1: Generate embeddings
embeddings = model.encode(texts)  # Shape: (n_samples, 768)

# Step 2: Train classifier
clf = LogisticRegression()
clf.fit(train_embeddings, labels)

# Step 3: Predict
predictions = clf.predict(test_embeddings)
```

### **Performance Results:**
- **F1 Score**: 0.85 (better than task-specific model!)
- **Embedding dimension**: 768 values per document
- **Training efficiency**: Fast CPU-based classifier training

## 🚀 **Zero-Shot Classification**

### **No Labeled Data Approach:**
- **Label descriptions**: "A negative movie review", "A positive movie review"
- **Cosine similarity**: Compare document embeddings to label embeddings
- **Creative labeling**: More specific descriptions improve performance

### **Implementation Steps:**
1. **Describe labels**: Convert class names to descriptive sentences
2. **Embed everything**: Both documents and label descriptions
3. **Calculate similarity**: Cosine similarity between doc-label pairs
4. **Assign labels**: Highest similarity determines classification

### **Mathematical Foundation:**
- **Cosine similarity**: cos(θ) = (A·B)/(||A||×||B||)
- **Angle interpretation**: Smaller angle = higher similarity
- **Range**: -1 to 1 (1 = identical direction, 0 = orthogonal)

### **Performance Results:**
- **F1 Score**: 0.78 (impressive for zero training data!)
- **Creativity matters**: Better label descriptions → better results
- **Practical value**: Quick feasibility testing without labeling effort

```mermaid
flowchart LR
    A[Document] --> B[Embed Document]
    C[Label Descriptions] --> D[Embed Labels]
    B --> E[Cosine Similarity]
    D --> E
    E --> F[Highest Similarity]
    F --> G[Predicted Label]
```

## 🤖 **Generative Model Classification**

### **Sequence-to-Sequence Nature:**
- **Input**: Text sequence
- **Output**: Text sequence (not direct class probabilities)
- **Prompt dependency**: Require instructions to understand task
- **Prompt engineering**: Iterative prompt improvement process

## 📝 **Text-to-Text Transfer Transformer (T5)**

### **Architecture:**
- **Encoder-decoder**: 12 encoders + 12 decoders (original Transformer style)
- **Everything as text**: All tasks converted to text-to-text format
- **Unified training**: Multiple tasks trained simultaneously

### **Training Process:**
1. **Pretraining**: Masked language modeling with token spans
2. **Multi-task fine-tuning**: Convert all tasks to instruction format
3. **Flan-T5**: Extended with 1000+ instruction tasks

### **Implementation:**
```python
# Load model
pipe = pipeline("text2text-generation", model="google/flan-t5-small")

# Create prompt
prompt = "Is the following sentence positive or negative? "
input_text = prompt + document

# Generate prediction
output = pipe(input_text)
prediction = 0 if output[0]["generated_text"] == "negative" else 1
```

### **Performance Results:**
- **F1 Score**: 0.84 (strong performance)
- **Model sizes**: small/base/large/xl/xxl variants available
- **Instruction following**: Good at understanding task requirements

## 💬 **ChatGPT Classification**

### **Training Pipeline:**
1. **Instruction tuning**: Manual instruction-output pairs
2. **Preference tuning**: Ranked outputs for human preference alignment
3. **Final model**: Optimized for human-like responses

### **API Implementation:**
- **OpenAI API**: External service access (not local model)
- **Cost considerations**: Pay-per-use pricing model
- **Rate limiting**: Manage request frequency to avoid errors
- **Prompt template**: Structured instructions for classification

### **Prompt Engineering:**
```python
prompt = """Predict whether the following document is a positive or negative
movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any
other answers."""
```

### **Performance Results:**
- **F1 Score**: 0.91 (highest performance)
- **Data contamination risk**: Unknown if test data was in training
- **Cost**: ~3 cents for full test dataset
- **Evaluation challenges**: Closed-source model limitations

```mermaid
graph TD
    A[Instruction Data] --> B[Instruction Tuning]
    B --> C[Base ChatGPT]
    D[Preference Data] --> E[Preference Tuning]
    C --> E
    E --> F[Final ChatGPT]
    F --> G[API Access]
    G --> H[Classification Task]
```

## 📊 **Performance Comparison Summary**

| Method | Model Type | F1 Score | Training Required | Cost |
|--------|------------|----------|------------------|------|
| Task-Specific | RoBERTa | 0.80 | None | Local GPU |
| Embedding + Classifier | MPNet + LogReg | 0.85 | Lightweight | GPU + CPU |
| Zero-Shot Embedding | MPNet | 0.78 | None | GPU |
| Flan-T5 | T5-small | 0.84 | None | Local GPU |
| ChatGPT | GPT-3.5 | 0.91 | None | API costs |

## 🔑 **Key Takeaways**

### **Model Selection Strategy:**
- **Task-specific**: Best for single, well-defined tasks
- **Embedding**: Most flexible for multiple tasks
- **Zero-shot**: Quick feasibility testing
- **Generative**: Powerful but require prompt engineering

### **Practical Considerations:**
- **Resource constraints**: Local vs API access tradeoffs
- **Performance requirements**: Speed vs accuracy balance
- **Data availability**: Labeled vs unlabeled scenarios
- **Cost sensitivity**: Training time vs inference costs

### **Best Practices:**
- **Baseline comparison**: Always compare against classical methods (TF-IDF + LogReg)
- **Prompt iteration**: Invest time in prompt engineering for generative models
- **Evaluation rigor**: Proper train/validation/test splits
- **Domain matching**: Consider training data similarity to target domain
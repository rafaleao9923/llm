# Chapter 2 - Tokens and Embeddings

## 🔤 **Tokens and Embeddings Overview**
- **Central concepts** for understanding LLM functionality and construction
- **Text processing pipeline**: Raw text → Tokens → Embeddings → Model computation
- **Foundation knowledge** for language model applications and future developments

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenization]
    B --> C[Token IDs]
    C --> D[Embeddings]
    D --> E[Language Model]
    E --> F[Output Generation]
```

## 🔧 **LLM Tokenization Process**

### **High-Level Tokenization Flow:**
- **Input processing**: Text prompt broken into pieces before model sees it
- **Token generation**: Models output one token at a time (streaming effect)
- **Bidirectional process**: Tokenizers handle both input and output conversion
- **Integer representation**: Models work with token IDs, not raw text

### **Token ID Structure:**
- **Unique identifiers**: Each token has specific ID in vocabulary table
- **Special tokens**: Beginning markers (<s>), assistants markers, separators
- **Decoding process**: Token IDs converted back to readable text
- **Sequential processing**: Each token processed in order

```mermaid
graph TD
    A[Raw Text Input] --> B[Tokenizer]
    B --> C[Token IDs Array]
    C --> D[Language Model]
    D --> E[Output Token IDs]
    E --> F[Tokenizer Decode]
    F --> G[Human-Readable Output]
```

## 📝 **Tokenization Methods**

### **🔹 Four Major Approaches:**

**Word Tokens:**
- **Complete words**: Each word = one token
- **Vocabulary explosion**: Many similar words (apologize, apology, apologetic)
- **Unknown word problem**: Can't handle new/unseen words
- **Legacy usage**: Common in word2vec, less used in modern LLMs

**Subword Tokens:**
- **Partial words**: "apologizing" → "apolog" + "izing"
- **Vocabulary efficiency**: Reuse common prefixes/suffixes across words
- **New word handling**: Break unknown words into known subparts
- **Context optimization**: More text fits in limited context windows

**Character Tokens:**
- **Individual letters**: "play" → "p" + "l" + "a" + "y"
- **Universal coverage**: Can represent any text with basic alphabet
- **Modeling difficulty**: Requires learning to spell before learning language
- **Sequence length**: Much longer sequences for same content

**Byte Tokens:**
- **Unicode bytes**: Raw byte-level representation
- **Tokenization-free**: No vocabulary limitations
- **Multilingual strength**: Universal across all languages
- **Research direction**: "ByT5" and "CANINE" models explore this approach

```mermaid
mindmap
  root((Tokenization Types))
    Word Level
      Complete Words
      Large Vocabulary
      Unknown Word Issues
      word2vec Era
    Subword Level
      BPE Method
      WordPiece Method
      Balanced Efficiency
      Modern Standard
    Character Level
      Individual Letters
      Universal Coverage
      Longer Sequences
      Spelling Learning
    Byte Level
      Unicode Bytes
      Language Agnostic
      Tokenization Free
      Research Frontier
```

## 🔍 **Real-World Tokenizer Comparison**

### **BERT (2018) - WordPiece:**
- **Vocabulary**: 30,522 tokens
- **Casing**: Uncased (lowercase) vs Cased versions
- **Special tokens**: [CLS], [SEP], [UNK], [MASK], [PAD]
- **Limitations**: No emoji/Chinese support, loses newlines
- **Token format**: "##" prefix for subword continuation

### **GPT-2 (2019) - BPE:**
- **Vocabulary**: 50,257 tokens
- **Improvements**: Preserves capitalization and newlines
- **Unicode handling**: Breaks down emojis/foreign chars into multiple tokens
- **Whitespace**: Represents spaces and indentation explicitly
- **Context awareness**: Better for code and structured text

### **GPT-4 (2023) - Advanced BPE:**
- **Vocabulary**: ~100,000 tokens
- **Efficiency**: Fewer tokens per word ("CAPITALIZATION" = 2 tokens vs 4)
- **Code optimization**: Single tokens for common programming constructs
- **Whitespace mastery**: Dedicated tokens for space sequences (1-83 spaces)
- **Fill-in-middle**: Special tokens for code completion tasks

### **Specialized Models:**

**StarCoder2 (Code-focused):**
- **Individual digits**: Each number digit = separate token
- **Repository context**: Special tokens for filenames, repo names
- **Mathematical precision**: Better number representation for calculations

**Galactica (Science-focused):**
- **Citation tokens**: [START_REF] and [END_REF] wrappers
- **Reasoning tokens**: <work> for step-by-step thinking
- **Domain optimization**: Scientific paper and formula handling

```mermaid
graph TD
    A[Tokenizer Evolution] --> B[BERT 2018]
    A --> C[GPT-2 2019]
    A --> D[GPT-4 2023]
    A --> E[Specialized Models]
    
    B --> F[Basic WordPiece]
    C --> G[BPE + Unicode]
    D --> H[Efficient BPE]
    E --> I[Domain-Specific]
    
    F --> J[Limited Scope]
    G --> K[Better Coverage]
    H --> L[Optimized Performance]
    I --> M[Task-Specialized]
```

## 🎯 **Tokenizer Design Factors**

### **🔹 Three Key Determinants:**

**Tokenization Method:**
- **Algorithm choice**: BPE, WordPiece, SentencePiece, Unigram
- **Optimization approach**: Different ways to build efficient vocabulary
- **Performance tradeoffs**: Speed vs accuracy vs coverage

**Initialization Parameters:**
- **Vocabulary size**: 30K-100K+ tokens (trending larger)
- **Special tokens**: Task-specific markers and control tokens
- **Capitalization handling**: Preserve vs normalize case
- **Language coverage**: Monolingual vs multilingual support

**Training Dataset Domain:**
- **Text type**: General web text, scientific papers, code repositories
- **Language distribution**: English-only vs multilingual corpora
- **Quality filtering**: Clean text vs raw internet data
- **Domain expertise**: Legal, medical, technical specialization

## 🔢 **Token Embeddings in Language Models**

### **Embedding Matrix Structure:**
- **Vocabulary mapping**: Each token ID → embedding vector
- **Random initialization**: Embeddings start random, learn during training
- **Shared across positions**: Same token = same initial embedding everywhere
- **Model component**: Embeddings are downloadable part of pretrained models

### **Contextualized Embeddings:**
- **Dynamic representation**: Same word gets different embeddings based on context
- **Layer processing**: Raw embeddings transformed through model layers
- **Application power**: Enables NER, summarization, classification tasks
- **Multimodal bridge**: Text embeddings connect to image generation systems

```mermaid
flowchart TD
    A[Token ID: 'bank'] --> B{Context Analysis}
    B -->|Financial Context| C[Bank Embedding v1]
    B -->|River Context| D[Bank Embedding v2]
    C --> E[Downstream Task]
    D --> E
    
    F[Static Embedding] --> G[Same everywhere]
    H[Contextualized Embedding] --> I[Context-dependent]
```

## 📊 **Text Embeddings for Documents**

### **Document-Level Representation:**
- **Single vector**: Entire sentence/document → one embedding
- **Semantic capture**: Meaning representation for comparison/search
- **Application enablement**: Powers semantic search, RAG, clustering
- **Model specialization**: sentence-transformers library and models

### **Common Creation Methods:**
- **Token averaging**: Average all token embeddings in sequence
- **Specialized training**: Models trained specifically for sentence embeddings
- **CLS token**: Use special classification token from BERT-style models
- **Pooling strategies**: Max pooling, attention-weighted averaging

## 🎵 **Word2Vec and Beyond LLMs**

### **Word2Vec Algorithm Core:**
- **Skip-gram training**: Predict if words are neighbors in context
- **Negative sampling**: Add random non-neighbor examples for training
- **Contrastive learning**: Positive examples vs negative examples
- **Sliding window**: Generate training pairs from text sequences

### **Training Process:**
1. **Text corpus**: Large dataset (Wikipedia, news, etc.)
2. **Window sliding**: Extract word pairs within context window
3. **Positive examples**: Actual neighboring words in text
4. **Negative sampling**: Random word pairs as negative examples
5. **Neural network**: Train classifier to distinguish positive/negative
6. **Embedding learning**: Embeddings updated to improve classification

```mermaid
flowchart TD
    A[Text: 'The cat sat on mat'] --> B[Sliding Window]
    B --> C[Positive Pairs: cat-sat, sat-on]
    B --> D[Negative Pairs: cat-random, sat-random]
    C --> E[Neural Network Training]
    D --> E
    E --> F[Updated Embeddings]
    F --> G[Semantic Similarity]
```

## 🎶 **Embeddings for Recommendation Systems**

### **Music Playlist Embeddings:**
- **Song as tokens**: Treat songs like words in vocabulary
- **Playlists as sentences**: Playlists provide context for song relationships
- **Similarity learning**: Songs appearing together = similar embeddings
- **Recommendation engine**: Find nearest neighbors in embedding space

### **Implementation Process:**
1. **Playlist dataset**: Collections of songs grouped by human curation
2. **Song tokenization**: Each song gets unique ID in vocabulary
3. **Context window**: Songs within same playlist = positive examples
4. **Word2Vec training**: Apply same algorithm to song sequences
5. **Similarity queries**: Find songs with closest embedding vectors
6. **Genre clustering**: Similar genres naturally group together

### **Real-World Results:**
- **Michael Jackson** → Prince, Madonna, other MJ songs
- **2Pac** → Nas, Notorious B.I.G., other rap artists
- **Metallica** → Van Halen, Guns N' Roses, heavy metal bands

## 🔑 **Key Technical Insights**

### **Tokenization Impact on Performance:**
- **Code models**: Need proper indentation token handling
- **Multilingual**: Byte-level fallback for unseen characters
- **Domain adaptation**: Specialized vocabularies improve task performance
- **Context efficiency**: Better tokenization = more content in limited windows

### **Embedding Quality Factors:**
- **Training data**: Domain-relevant corpora produce better embeddings
- **Model architecture**: Transformer layers create contextual representations
- **Task alignment**: Embeddings trained for specific use cases perform better
- **Vector dimensions**: Higher dimensions capture more nuanced relationships
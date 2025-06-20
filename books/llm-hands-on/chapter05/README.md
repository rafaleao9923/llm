# Chapter 5 - Text Clustering and Topic Modeling

## 🎯 **Text Clustering and Topic Modeling Overview**
- **Unsupervised learning**: Group similar texts without labeled data
- **Semantic grouping**: Based on content meaning and relationships
- **Efficient categorization**: Handle large volumes of unstructured text
- **Exploratory analysis**: Quick understanding of document collections

```mermaid
flowchart TD
    A[Unstructured Text Collection] --> B[Text Clustering]
    B --> C[Semantic Groups]
    C --> D[Topic Modeling]
    D --> E[Labeled Topics with Keywords]
```

## 📊 **Dataset: ArXiv Research Papers**
- **ArXiv platform**: Open-access scholarly articles repository
- **Focus domain**: Computation and Language (cs.CL)
- **Dataset size**: 44,949 abstracts from 1991-2024
- **Use case**: Academic paper clustering and topic discovery

## 🔄 **Common Text Clustering Pipeline**

### **Three-Step Process:**
1. **Document embedding**: Convert text to numerical representations
2. **Dimensionality reduction**: Compress high-dimensional embeddings
3. **Clustering**: Group similar documents together

```mermaid
graph LR
    A[Text Documents] --> B[Embedding Model]
    B --> C[Vector Embeddings]
    C --> D[UMAP Reduction]
    D --> E[Reduced Embeddings]
    E --> F[HDBSCAN Clustering]
    F --> G[Document Clusters]
```

## 🎯 **Step 1: Document Embedding**

### **Embedding Model Selection:**
- **MTEB leaderboard**: Performance benchmark for embedding models
- **Model choice**: "thenlper/gte-small" for clustering optimization
- **Performance balance**: Good clustering scores + fast inference
- **Semantic similarity**: Optimized for finding related documents

### **Implementation:**
```python
embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(abstracts)
# Result: (44949, 384) - 44,949 docs × 384-dimensional embeddings
```

### **Semantic Representation:**
- **Vector dimensions**: 384 numerical values per document
- **Meaning capture**: Embeddings represent document semantics
- **Feature space**: Vectors serve as clustering features

## 📉 **Step 2: Dimensionality Reduction**

### **High-Dimensional Challenge:**
- **Curse of dimensionality**: Exponential growth in possible values
- **Clustering difficulty**: Hard to find meaningful clusters in high dimensions
- **Solution**: Compress to lower-dimensional space while preserving structure

### **UMAP (Uniform Manifold Approximation and Projection):**
- **Nonlinear handling**: Better than PCA for complex relationships
- **Parameter tuning**: n_components=5, min_dist=0.0, metric='cosine'
- **Information tradeoff**: Some data loss but improved clustering

```mermaid
graph TD
    A[384-Dimensional Space] --> B[UMAP Reduction]
    B --> C[5-Dimensional Space]
    D[High Complexity] --> E[Manageable Complexity]
    F[Difficult Clustering] --> G[Clear Cluster Patterns]
```

### **Implementation Details:**
```python
umap_model = UMAP(
    n_components=5,        # Target dimensions
    min_dist=0.0,         # Tight clusters
    metric='cosine',      # High-dim friendly
    random_state=42       # Reproducibility
)
reduced_embeddings = umap_model.fit_transform(embeddings)
```

## 🎯 **Step 3: Clustering with HDBSCAN**

### **Density-Based Clustering:**
- **No predefined clusters**: Algorithm determines optimal number
- **Outlier detection**: Points that don't belong to any cluster
- **Hierarchical approach**: Builds cluster hierarchy
- **Parameter**: min_cluster_size controls cluster granularity

### **Advantages over K-Means:**
- **Automatic cluster count**: No need to specify number of clusters
- **Irregular shapes**: Handles non-spherical cluster shapes
- **Outlier tolerance**: Doesn't force all points into clusters
- **Noise handling**: Natural outlier detection capability

### **Results Analysis:**
```python
hdbscan_model = HDBSCAN(min_cluster_size=50)
clusters = hdbscan_model.fit(reduced_embeddings)
# Result: 156 clusters generated
```

## 👁️ **Cluster Visualization and Inspection**

### **Manual Inspection:**
- **Cluster examination**: Review documents in each cluster
- **Pattern identification**: Find common themes in grouped documents
- **Example**: Cluster 0 contains sign language translation papers

### **2D Visualization:**
- **UMAP reduction**: Compress to 2 dimensions for plotting
- **Scatter plot**: Each point represents a document
- **Color coding**: Same color = same cluster
- **Outlier highlighting**: Gray points for unclustered documents

```mermaid
graph TD
    A[384D Embeddings] --> B[UMAP to 2D]
    B --> C[Scatter Plot]
    C --> D[Colored Clusters]
    C --> E[Gray Outliers]
    D --> F[Visual Pattern Recognition]
```

## 🏷️ **From Clustering to Topic Modeling**

### **Topic Definition:**
- **Manual labeling**: Assign names to clusters based on content
- **Keyword extraction**: Use most representative words
- **Theme identification**: Find abstract topics in document collections

### **Traditional Approaches:**
- **Latent Dirichlet Allocation (LDA)**: Probabilistic topic modeling
- **Bag-of-words limitation**: Ignores context and meaning
- **Word distribution**: Topics as probability distributions over vocabulary

## 🤖 **BERTopic: Modular Topic Modeling**

### **Two-Step Process:**
1. **Semantic clustering**: Embed → Reduce → Cluster (same as before)
2. **Topic representation**: Extract meaningful topic descriptions

### **c-TF-IDF Approach:**
- **Cluster-level frequency**: Count words per cluster, not per document
- **Term weighting**: Balance frequency with cluster-specificity
- **Stop word handling**: Reduce weight of common, meaningless words

```mermaid
flowchart LR
    A[Documents per Cluster] --> B[Cluster Term Frequency]
    B --> C[c-TF calculation]
    D[Term frequency across clusters] --> E[IDF calculation]
    C --> F[c-TF-IDF weights]
    E --> F
    F --> G[Topic Keywords]
```

### **Mathematical Foundation:**
- **c-TF**: Cluster Term Frequency (word count per cluster)
- **IDF**: Inverse Document Frequency (log(avg_freq/word_freq))
- **c-TF-IDF**: c-TF × IDF (weighted importance score)

### **Modularity Benefits:**
- **Lego-block design**: Each component independently replaceable
- **Model flexibility**: Swap embedding models, clustering algorithms
- **Pipeline customization**: Adapt to specific use cases
- **Future-proof**: Easy integration of new models

## 🔧 **BERTopic Implementation**

### **Basic Setup:**
```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model
).fit(abstracts, embeddings)
```

### **Topic Exploration:**
- **get_topic_info()**: Overview of all topics with keywords
- **get_topic(n)**: Detailed keywords for specific topic
- **find_topics()**: Search for topics by keyword similarity

### **Example Topics:**
- **Topic 0**: Speech recognition ("speech", "asr", "recognition")
- **Topic 22**: Topic modeling ("topic", "topics", "lda", "modeling")
- **Topic 1**: Medical NLP ("medical", "clinical", "biomedical")

```mermaid
mindmap
  root((BERTopic Features))
    Core Pipeline
      Embedding
      Dimensionality Reduction
      Clustering
      Topic Representation
    Modularity
      Swappable Components
      Custom Models
      Pipeline Flexibility
    Exploration
      Topic Search
      Visualization
      Interactive Plots
    Variants
      Guided Topics
      Hierarchical
      Dynamic
      Multimodal
```

## 🎨 **Advanced Representation Models**

### **KeyBERTInspired:**
- **Semantic similarity**: Compare document and keyword embeddings
- **Cosine similarity**: Find most representative words per topic
- **Improvement**: Better readability, removes stop words
- **Tradeoff**: May remove domain-specific abbreviations (e.g., "nmt")

### **Maximal Marginal Relevance (MMR):**
- **Diversity optimization**: Reduce keyword redundancy
- **Iterative selection**: Choose diverse but relevant keywords
- **Parameter**: Diversity setting controls keyword variety
- **Result**: Remove similar words (e.g., "summary" vs "summaries")

```python
# KeyBERTInspired example
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# MMR example
representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model.update_topics(abstracts, representation_model=representation_model)
```

## 🤖 **Generative Model Integration**

### **Text Generation Block:**
- **Label generation**: Create short topic labels instead of keywords
- **Efficiency**: One generation per topic, not per document
- **Input combination**: Use keywords + representative documents
- **Prompt engineering**: Guide model to generate meaningful labels

### **Prompt Structure:**
```python
prompt = """I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""
```

### **Model Comparison:**

**Flan-T5 Results:**
- **Topic 0**: "Speech-to-description"
- **Topic 1**: "Science/Tech" (too broad)
- **Topic 4**: "Summarization"

**GPT-3.5 Results:**
- **Topic 0**: "Leveraging External Data for Improving Low-Res..."
- **Topic 1**: "Improved Representation Learning for Biomedical..."
- **Topic 2**: "Advancements in Aspect-Based Sentiment Analysis..."

```mermaid
graph TD
    A[Topic Keywords] --> B[Generative Model]
    C[Representative Documents] --> B
    B --> D[Topic Label]
    E[Prompt Template] --> B
    
    F[Multiple Models] --> G[Flan-T5: Simple labels]
    F --> H[GPT-3.5: Detailed labels]
    F --> I[Local Models: Privacy]
```

## 📊 **Performance and Visualization**

### **Multiple Representations:**
- **Complementary views**: Keywords, labels, embeddings side-by-side
- **Model stacking**: Combine multiple representation approaches
- **Perspective variety**: Different models provide different insights

### **Interactive Visualization:**
- **Document scatter plot**: Hover to see titles and topics
- **Topic exploration**: Click to focus on specific topics
- **Relationship mapping**: Heatmaps showing topic similarities
- **Hierarchical structure**: Tree view of topic relationships

### **Advanced Plotting:**
```python
# Interactive document visualization
fig = topic_model.visualize_documents(titles, reduced_embeddings)

# Topic relationships
topic_model.visualize_heatmap(n_clusters=30)

# Hierarchical structure
topic_model.visualize_hierarchy()
```

## 🔑 **Key Implementation Strategies**

### **Pipeline Optimization:**
- **Embedding selection**: Choose models optimized for clustering tasks
- **Parameter tuning**: Adjust UMAP and HDBSCAN for dataset characteristics
- **Representation stacking**: Combine multiple representation models
- **Iterative refinement**: Test different components for best results

### **Practical Considerations:**
- **Outlier handling**: Decide whether to include or exclude outliers
- **Cluster granularity**: Balance between too few and too many clusters
- **Domain expertise**: Validate results with subject matter experts
- **Computational resources**: Consider model size vs. performance tradeoffs

### **Best Practices:**
- **Human evaluation**: Always manually inspect cluster quality
- **Multiple perspectives**: Use various representation models
- **Validation**: Check if domain-specific papers cluster correctly
- **Iterative improvement**: Refine based on clustering results

## 🎯 **Chapter Summary Insights**

### **Unsupervised Power:**
- **No labels required**: Discover patterns without prior annotation
- **Exploratory capability**: Quick understanding of large document collections
- **Flexibility**: Adapt to various domains and document types

### **Modern Approach Benefits:**
- **Semantic understanding**: Context-aware embeddings vs. bag-of-words
- **Modular design**: Mix and match best components for specific needs
- **LLM integration**: Leverage latest generative models for interpretability

### **Practical Applications:**
- **Academic research**: Discover trends in scientific literature
- **Business intelligence**: Understand customer feedback themes
- **Content organization**: Automatically categorize large text collections
- **Data exploration**: Quick insights into unfamiliar document sets
## 🎯 **Author's Core Intentions**

The authors demonstrate the evolution from traditional bag-of-words topic modeling (LDA) to modern semantic clustering using transformer embeddings. The key progression shows:

1. **Three-step pipeline** - Embed → Reduce → Cluster for semantic grouping
2. **Modular architecture** - BERTopic's Lego-block design allows swapping components
3. **Multiple representations** - Keywords, labels, and generative descriptions provide different perspectives
4. **Interactive exploration** - Visual tools for understanding document collections

The sample code reveals practical implementation patterns: UMAP for dimensionality reduction, HDBSCAN for density-based clustering, and c-TF-IDF for topic keyword extraction.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedTopicModelingSystem:
    def __init__(self):
        self.dataset = None
        self.abstracts = None
        self.titles = None
        self.embeddings = None
        self.models = {}
        self.clustering_results = {}
        self.topic_models = {}
        
    def load_data(self, dataset_name="maartengr/arxiv_nlp", sample_size=None):
        print(f"📊 Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name)["train"]
        
        if sample_size:
            indices = np.random.choice(len(self.dataset), size=sample_size, replace=False)
            self.dataset = self.dataset.select(indices)
        
        self.abstracts = self.dataset["Abstracts"]
        self.titles = self.dataset["Titles"]
        
        print(f"✅ Loaded {len(self.abstracts)} documents")
        print(f"📝 Sample abstract: '{self.abstracts[0][:200]}...'")
        
        return self.dataset
    
    def benchmark_embedding_models(self, model_configs):
        print("\n🧠 EMBEDDING MODEL BENCHMARKING")
        print("=" * 50)
        
        embedding_results = {}
        
        for model_name, model_path in model_configs.items():
            print(f"\n🔧 Testing {model_name}")
            print(f"   Model: {model_path}")
            
            try:
                model = SentenceTransformer(model_path)
                embeddings = model.encode(self.abstracts[:1000], show_progress_bar=True)
                
                embedding_info = {
                    'model_name': model_name,
                    'model_path': model_path,
                    'embedding_dim': embeddings.shape[1],
                    'embeddings': embeddings,
                    'avg_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
                    'embedding_variance': np.var(embeddings.flatten())
                }
                
                quick_clustering_score = self._quick_clustering_evaluation(embeddings)
                embedding_info['clustering_score'] = quick_clustering_score
                
                embedding_results[model_name] = embedding_info
                
                print(f"   ✅ Dimensions: {embeddings.shape[1]}")
                print(f"   📊 Clustering score: {quick_clustering_score:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                embedding_results[model_name] = {'error': str(e)}
        
        self.models['embeddings'] = embedding_results
        return embedding_results
    
    def _quick_clustering_evaluation(self, embeddings, n_clusters=10):
        if len(embeddings) < n_clusters:
            return 0.0
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        try:
            return silhouette_score(embeddings, labels)
        except:
            return 0.0
    
    def compare_dimensionality_reduction(self, embeddings, methods_config):
        print("\n📉 DIMENSIONALITY REDUCTION COMPARISON")
        print("=" * 50)
        
        reduction_results = {}
        
        for method_name, config in methods_config.items():
            print(f"\n🔧 Testing {method_name}")
            
            try:
                if method_name == 'UMAP':
                    reducer = UMAP(**config, random_state=42)
                elif method_name == 'PCA':
                    reducer = PCA(**config, random_state=42)
                else:
                    continue
                
                reduced_embeddings = reducer.fit_transform(embeddings)
                
                clustering_score = self._evaluate_clustering_quality(reduced_embeddings)
                
                reduction_results[method_name] = {
                    'method': method_name,
                    'config': config,
                    'reduced_embeddings': reduced_embeddings,
                    'output_dim': reduced_embeddings.shape[1],
                    'clustering_score': clustering_score,
                    'variance_explained': self._calculate_variance_explained(embeddings, reduced_embeddings)
                }
                
                print(f"   ✅ Output dimensions: {reduced_embeddings.shape[1]}")
                print(f"   📊 Clustering score: {clustering_score:.3f}")
                print(f"   📈 Variance explained: {reduction_results[method_name]['variance_explained']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                reduction_results[method_name] = {'error': str(e)}
        
        self.models['reduction'] = reduction_results
        return reduction_results
    
    def _evaluate_clustering_quality(self, embeddings, n_clusters_range=[5, 10, 15, 20]):
        best_score = -1
        
        for n_clusters in n_clusters_range:
            if len(embeddings) < n_clusters:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
            except:
                continue
        
        return best_score
    
    def _calculate_variance_explained(self, original_embeddings, reduced_embeddings):
        original_var = np.var(original_embeddings.flatten())
        
        try:
            pca = PCA().fit(original_embeddings)
            n_components = reduced_embeddings.shape[1]
            explained_ratio = np.sum(pca.explained_variance_ratio_[:n_components])
            return explained_ratio
        except:
            return np.var(reduced_embeddings.flatten()) / original_var
    
    def benchmark_clustering_algorithms(self, reduced_embeddings, clustering_configs):
        print("\n🎯 CLUSTERING ALGORITHM COMPARISON")
        print("=" * 50)
        
        clustering_results = {}
        
        for algo_name, config in clustering_configs.items():
            print(f"\n🔧 Testing {algo_name}")
            
            try:
                if algo_name == 'HDBSCAN':
                    clusterer = HDBSCAN(**config)
                    labels = clusterer.fit_predict(reduced_embeddings)
                elif algo_name == 'KMeans':
                    clusterer = KMeans(**config, random_state=42)
                    labels = clusterer.fit_predict(reduced_embeddings)
                elif algo_name == 'Agglomerative':
                    clusterer = AgglomerativeClustering(**config)
                    labels = clusterer.fit_predict(reduced_embeddings)
                else:
                    continue
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = list(labels).count(-1) if -1 in labels else 0
                
                clustering_metrics = self._calculate_clustering_metrics(reduced_embeddings, labels)
                
                clustering_results[algo_name] = {
                    'algorithm': algo_name,
                    'config': config,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'outlier_ratio': n_outliers / len(labels),
                    **clustering_metrics
                }
                
                print(f"   ✅ Clusters found: {n_clusters}")
                print(f"   🔍 Outliers: {n_outliers} ({clustering_results[algo_name]['outlier_ratio']:.1%})")
                print(f"   📊 Silhouette score: {clustering_metrics.get('silhouette_score', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                clustering_results[algo_name] = {'error': str(e)}
        
        self.clustering_results = clustering_results
        return clustering_results
    
    def _calculate_clustering_metrics(self, embeddings, labels):
        metrics = {}
        
        valid_labels = labels[labels != -1] if -1 in labels else labels
        valid_embeddings = embeddings[labels != -1] if -1 in labels else embeddings
        
        if len(set(valid_labels)) > 1 and len(valid_embeddings) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_labels)
            except:
                metrics['silhouette_score'] = None
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_labels)
            except:
                metrics['calinski_harabasz_score'] = None
        
        cluster_sizes = Counter(valid_labels)
        metrics['avg_cluster_size'] = np.mean(list(cluster_sizes.values()))
        metrics['std_cluster_size'] = np.std(list(cluster_sizes.values()))
        metrics['min_cluster_size'] = min(cluster_sizes.values()) if cluster_sizes else 0
        metrics['max_cluster_size'] = max(cluster_sizes.values()) if cluster_sizes else 0
        
        return metrics
    
    def build_advanced_topic_models(self, embedding_model, reduction_config, clustering_config):
        print("\n🤖 ADVANCED TOPIC MODEL CONSTRUCTION")
        print("=" * 50)
        
        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(self.abstracts, show_progress_bar=True)
        self.embeddings = embeddings
        
        umap_model = UMAP(**reduction_config, random_state=42)
        hdbscan_model = HDBSCAN(**clustering_config)
        
        representation_models = {
            'c-TF-IDF': None,
            'KeyBERT': KeyBERTInspired(),
            'MMR': MaximalMarginalRelevance(diversity=0.3),
        }
        
        topic_models = {}
        
        for repr_name, repr_model in representation_models.items():
            print(f"\n🎭 Building model with {repr_name} representation...")
            
            try:
                topic_model = BERTopic(
                    embedding_model=model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    representation_model=repr_model,
                    verbose=False
                ).fit(self.abstracts, embeddings)
                
                topic_info = topic_model.get_topic_info()
                
                model_metrics = {
                    'n_topics': len(topic_info) - 1,
                    'n_outliers': topic_info[topic_info['Topic'] == -1]['Count'].sum(),
                    'largest_topic_size': topic_info[topic_info['Topic'] != -1]['Count'].max(),
                    'avg_topic_size': topic_info[topic_info['Topic'] != -1]['Count'].mean(),
                    'topic_coherence': self._calculate_topic_coherence(topic_model),
                    'topic_diversity': self._calculate_topic_diversity(topic_model)
                }
                
                topic_models[repr_name] = {
                    'model': topic_model,
                    'topic_info': topic_info,
                    'metrics': model_metrics
                }
                
                print(f"   ✅ Topics: {model_metrics['n_topics']}")
                print(f"   📊 Avg topic size: {model_metrics['avg_topic_size']:.1f}")
                print(f"   🎯 Topic coherence: {model_metrics['topic_coherence']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                topic_models[repr_name] = {'error': str(e)}
        
        self.topic_models = topic_models
        return topic_models
    
    def _calculate_topic_coherence(self, topic_model, top_n=10):
        try:
            topics = topic_model.get_topics()
            coherence_scores = []
            
            for topic_id in list(topics.keys())[:10]:
                if topic_id == -1:
                    continue
                
                topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]
                
                if len(topic_words) > 1:
                    word_embeddings = topic_model.embedding_model.encode(topic_words)
                    similarity_matrix = np.corrcoef(word_embeddings)
                    coherence = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                    coherence_scores.append(coherence)
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
        except:
            return 0.0
    
    def _calculate_topic_diversity(self, topic_model, top_n=10):
        try:
            topics = topic_model.get_topics()
            all_words = []
            
            for topic_id in list(topics.keys())[:10]:
                if topic_id == -1:
                    continue
                topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]
                all_words.extend(topic_words)
            
            unique_words = len(set(all_words))
            total_words = len(all_words)
            
            return unique_words / total_words if total_words > 0 else 0.0
        except:
            return 0.0
    
    def create_comprehensive_visualizations(self, topic_model, reduced_embeddings=None):
        print("\n📊 COMPREHENSIVE VISUALIZATION SUITE")
        print("=" * 50)
        
        if reduced_embeddings is None:
            umap_2d = UMAP(n_components=2, random_state=42)
            reduced_embeddings = umap_2d.fit_transform(self.embeddings)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Document Clusters', 'Topic Sizes', 'Topic Hierarchy', 'Topic Similarity'],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        topics = topic_model.topics_
        topic_info = topic_model.get_topic_info()
        
        colors = px.colors.qualitative.Set3
        
        for i, topic_id in enumerate(topic_info['Topic'][:20]):
            if topic_id == -1:
                continue
                
            mask = np.array(topics) == topic_id
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=reduced_embeddings[mask, 0],
                        y=reduced_embeddings[mask, 1],
                        mode='markers',
                        name=f'Topic {topic_id}',
                        marker=dict(size=3, color=colors[i % len(colors)]),
                        text=[self.titles[j] for j in np.where(mask)[0]],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        topic_counts = topic_info[topic_info['Topic'] != -1].head(10)
        fig.add_trace(
            go.Bar(
                x=topic_counts['Count'],
                y=[f"Topic {t}" for t in topic_counts['Topic']],
                orientation='h',
                name='Topic Sizes'
            ),
            row=1, col=2
        )
        
        try:
            hierarchical_topics = topic_model.hierarchical_topics(self.abstracts)
            linkage_matrix = linkage(topic_model.umap_model.embedding_, method='ward')
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(linkage_matrix))),
                    y=linkage_matrix[:, 2],
                    mode='lines+markers',
                    name='Hierarchy'
                ),
                row=2, col=1
            )
        except:
            pass
        
        try:
            topic_embeddings = []
            topic_labels = []
            
            for topic_id in topic_info['Topic'][:10]:
                if topic_id == -1:
                    continue
                
                topic_words = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
                topic_embedding = topic_model.embedding_model.encode(' '.join(topic_words))
                topic_embeddings.append(topic_embedding)
                topic_labels.append(f'Topic {topic_id}')
            
            if topic_embeddings:
                topic_embeddings = np.array(topic_embeddings)
                umap_topics = UMAP(n_components=2, random_state=42).fit_transform(topic_embeddings)
                
                fig.add_trace(
                    go.Scatter(
                        x=umap_topics[:, 0],
                        y=umap_topics[:, 1],
                        mode='markers+text',
                        text=topic_labels,
                        textposition='middle center',
                        marker=dict(size=10),
                        name='Topic Similarity'
                    ),
                    row=2, col=2
                )
        except:
            pass
        
        fig.update_layout(height=800, showlegend=False, title_text="Topic Modeling Analysis Dashboard")
        fig.show()
        
        return fig
    
    def analyze_topic_evolution(self, topic_model, time_column='year'):
        print("\n📈 TOPIC EVOLUTION ANALYSIS")
        print("=" * 50)
        
        if time_column not in self.dataset.column_names:
            print(f"❌ Column '{time_column}' not found in dataset")
            return None
        
        topics = topic_model.topics_
        years = self.dataset[time_column]
        
        topic_evolution = {}
        
        for year in sorted(set(years)):
            year_mask = np.array(years) == year
            year_topics = [topics[i] for i in np.where(year_mask)[0]]
            year_topic_counts = Counter(year_topics)
            
            topic_evolution[year] = year_topic_counts
        
        evolution_df = pd.DataFrame(topic_evolution).fillna(0)
        
        top_topics = evolution_df.sum(axis=1).nlargest(10).index
        evolution_subset = evolution_df.loc[top_topics]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        evolution_subset.T.plot(kind='line', ax=axes[0], marker='o')
        axes[0].set_title('Topic Evolution Over Time (Top 10 Topics)')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Documents')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        evolution_pct = evolution_subset.div(evolution_subset.sum(axis=0), axis=1) * 100
        evolution_pct.T.plot(kind='area', stacked=True, ax=axes[1], alpha=0.7)
        axes[1].set_title('Topic Distribution Over Time (Percentage)')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Percentage of Documents')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        return evolution_df
    
    def compare_all_approaches(self):
        print("\n📈 COMPREHENSIVE METHOD COMPARISON")
        print("=" * 50)
        
        comparison_data = []
        
        for method_name, model_info in self.topic_models.items():
            if 'error' not in model_info:
                metrics = model_info['metrics']
                comparison_data.append({
                    'Method': method_name,
                    'Topics': metrics['n_topics'],
                    'Avg Topic Size': metrics['avg_topic_size'],
                    'Coherence': metrics['topic_coherence'],
                    'Diversity': metrics['topic_diversity'],
                    'Outlier Ratio': metrics['n_outliers'] / len(self.abstracts)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            print("🏆 Model Comparison Results:")
            print(df.round(3).to_string(index=False))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            df.plot(x='Method', y='Coherence', kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('Topic Coherence by Method')
            axes[0, 0].set_ylabel('Coherence Score')
            
            df.plot(x='Method', y='Diversity', kind='bar', ax=axes[0, 1], color='lightgreen')
            axes[0, 1].set_title('Topic Diversity by Method')
            axes[0, 1].set_ylabel('Diversity Score')
            
            df.plot(x='Method', y='Topics', kind='bar', ax=axes[1, 0], color='salmon')
            axes[1, 0].set_title('Number of Topics by Method')
            axes[1, 0].set_ylabel('Number of Topics')
            
            df.plot(x='Method', y='Outlier Ratio', kind='bar', ax=axes[1, 1], color='gold')
            axes[1, 1].set_title('Outlier Ratio by Method')
            axes[1, 1].set_ylabel('Outlier Ratio')
            
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return df
        
        return None

def main_topic_modeling_analysis():
    print("🎯 Chapter 5: Advanced Topic Modeling Analysis")
    print("=" * 60)
    
    system = AdvancedTopicModelingSystem()
    system.load_data(sample_size=5000)
    
    print("\n" + "="*60)
    print("🧠 PART 1: EMBEDDING MODEL COMPARISON")
    
    embedding_models = {
        "GTE-Small": "thenlper/gte-small",
        "MPNet": "sentence-transformers/all-mpnet-base-v2", 
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "E5-Small": "intfloat/e5-small-v2"
    }
    
    embedding_results = system.benchmark_embedding_models(embedding_models)
    
    best_embedding = max(
        [r for r in embedding_results.values() if 'error' not in r],
        key=lambda x: x['clustering_score']
    )
    
    print(f"\n🏆 Best embedding model: {best_embedding['model_name']}")
    print(f"   Clustering score: {best_embedding['clustering_score']:.3f}")
    
    print("\n" + "="*60)
    print("📉 PART 2: DIMENSIONALITY REDUCTION COMPARISON")
    
    reduction_methods = {
        'UMAP': {'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'},
        'PCA': {'n_components': 5}
    }
    
    reduction_results = system.compare_dimensionality_reduction(
        best_embedding['embeddings'], 
        reduction_methods
    )
    
    print("\n" + "="*60)
    print("🎯 PART 3: CLUSTERING ALGORITHM COMPARISON")
    
    best_reduction = max(
        [r for r in reduction_results.values() if 'error' not in r],
        key=lambda x: x['clustering_score']
    )
    
    clustering_methods = {
        'HDBSCAN': {'min_cluster_size': 50, 'metric': 'euclidean'},
        'KMeans': {'n_clusters': 20},
        'Agglomerative': {'n_clusters': 20, 'linkage': 'ward'}
    }
    
    clustering_results = system.benchmark_clustering_algorithms(
        best_reduction['reduced_embeddings'],
        clustering_methods
    )
    
    print("\n" + "="*60)
    print("🤖 PART 4: ADVANCED TOPIC MODEL CONSTRUCTION")
    
    topic_models = system.build_advanced_topic_models(
        embedding_model=best_embedding['model_path'],
        reduction_config={'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'},
        clustering_config={'min_cluster_size': 50, 'metric': 'euclidean'}
    )
    
    print("\n" + "="*60)
    print("📊 PART 5: COMPREHENSIVE VISUALIZATION")
    
    if 'c-TF-IDF' in topic_models and 'error' not in topic_models['c-TF-IDF']:
        best_model = topic_models['c-TF-IDF']['model']
        system.create_comprehensive_visualizations(best_model)
    
    print("\n" + "="*60)
    print("📈 PART 6: METHOD COMPARISON")
    
    comparison_df = system.compare_all_approaches()
    
    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    main_topic_modeling_analysis()
```

---

# Chapter 5 Advanced Topic Modeling Exercises

## 🎯 **Dynamic Topic Modeling**

### **Exercise 1: Temporal Topic Evolution**
```python
class TemporalTopicAnalyzer:
    def __init__(self, time_slices=5):
        self.time_slices = time_slices
        self.models_by_time = {}
        self.topic_alignments = {}
        
    def create_dynamic_topics(self, documents, timestamps, method='sliding_window'):
        """
        Create topic models for different time periods
        """
        sorted_indices = np.argsort(timestamps)
        sorted_docs = [documents[i] for i in sorted_indices]
        sorted_times = [timestamps[i] for i in sorted_indices]
        
        if method == 'sliding_window':
            return self._sliding_window_approach(sorted_docs, sorted_times)
        elif method == 'fixed_periods':
            return self._fixed_period_approach(sorted_docs, sorted_times)
        else:
            return self._evolutionary_approach(sorted_docs, sorted_times)
    
    def _sliding_window_approach(self, documents, timestamps):
        """
        Create overlapping time windows for smooth topic evolution
        """
        window_size = len(documents) // self.time_slices
        overlap = window_size // 2
        
        time_models = {}
        
        for i in range(0, len(documents) - window_size, overlap):
            window_docs = documents[i:i + window_size]
            window_start = timestamps[i]
            window_end = timestamps[min(i + window_size - 1, len(timestamps) - 1)]
            
            topic_model = BERTopic(
                min_topic_size=max(10, len(window_docs) // 20),
                nr_topics="auto"
            ).fit(window_docs)
            
            time_models[f"{window_start}-{window_end}"] = {
                'model': topic_model,
                'documents': window_docs,
                'start_time': window_start,
                'end_time': window_end,
                'doc_count': len(window_docs)
            }
        
        return time_models
    
    def track_topic_evolution(self, time_models, similarity_threshold=0.7):
        """
        Track how topics evolve across time periods
        """
        evolution_chains = []
        model_keys = sorted(time_models.keys())
        
        for i in range(len(model_keys) - 1):
            current_model = time_models[model_keys[i]]['model']
            next_model = time_models[model_keys[i + 1]]['model']
            
            topic_similarities = self._calculate_topic_similarities(
                current_model, next_model
            )
            
            evolution_chains.append({
                'from_period': model_keys[i],
                'to_period': model_keys[i + 1],
                'topic_matches': topic_similarities,
                'emerging_topics': self._find_emerging_topics(topic_similarities),
                'declining_topics': self._find_declining_topics(topic_similarities)
            })
        
        return evolution_chains
    
    def _calculate_topic_similarities(self, model1, model2):
        """
        Calculate similarity between topics across time periods
        """
        similarities = {}
        
        for topic1_id in model1.get_topics():
            if topic1_id == -1:
                continue
                
            topic1_words = [word for word, _ in model1.get_topic(topic1_id)[:10]]
            best_match = -1
            best_score = 0
            
            for topic2_id in model2.get_topics():
                if topic2_id == -1:
                    continue
                    
                topic2_words = [word for word, _ in model2.get_topic(topic2_id)[:10]]
                
                overlap = len(set(topic1_words) & set(topic2_words))
                jaccard_sim = overlap / len(set(topic1_words) | set(topic2_words))
                
                if jaccard_sim > best_score:
                    best_score = jaccard_sim
                    best_match = topic2_id
            
            similarities[topic1_id] = {
                'matched_topic': best_match,
                'similarity_score': best_score,
                'words_from': topic1_words,
                'words_to': [word for word, _ in model2.get_topic(best_match)[:10]] if best_match != -1 else []
            }
        
        return similarities
    
    def _find_emerging_topics(self, similarities):
        """
        Find topics that appear in the new time period
        """
        matched_topics = {sim['matched_topic'] for sim in similarities.values() if sim['matched_topic'] != -1}
        return [topic for topic in similarities.keys() if topic not in matched_topics]
    
    def _find_declining_topics(self, similarities):
        """
        Find topics that disappear or significantly change
        """
        return [topic for topic, sim in similarities.items() if sim['similarity_score'] < 0.3]

temporal_analyzer = TemporalTopicAnalyzer()
```

### **Exercise 2: Hierarchical Topic Discovery**
```python
class HierarchicalTopicExplorer:
    def __init__(self):
        self.hierarchy_levels = {}
        self.topic_tree = {}
        
    def build_topic_hierarchy(self, documents, max_levels=3):
        """
        Build hierarchical topic structure from coarse to fine-grained
        """
        hierarchy = {}
        current_docs = documents
        
        for level in range(max_levels):
            print(f"Building level {level + 1} hierarchy...")
            
            # Adjust clustering parameters by level
            min_cluster_size = max(50 // (level + 1), 10)
            
            topic_model = BERTopic(
                min_topic_size=min_cluster_size,
                nr_topics="auto"
            ).fit(current_docs)
            
            topics = topic_model.topics_
            topic_info = topic_model.get_topic_info()
            
            level_data = {
                'model': topic_model,
                'topics': topics,
                'topic_info': topic_info,
                'documents': current_docs,
                'subtopics': {}
            }
            
            # Create subtopics for next level
            if level < max_levels - 1:
                for topic_id in topic_info['Topic']:
                    if topic_id == -1:
                        continue
                    
                    topic_docs = [current_docs[i] for i, t in enumerate(topics) if t == topic_id]
                    
                    if len(topic_docs) > min_cluster_size * 2:
                        level_data['subtopics'][topic_id] = topic_docs
            
            hierarchy[f'level_{level + 1}'] = level_data
            
            # Prepare documents for next level (only large topics)
            if level < max_levels - 1 and level_data['subtopics']:
                next_level_docs = []
                for subtopic_docs in level_data['subtopics'].values():
                    next_level_docs.extend(subtopic_docs)
                current_docs = next_level_docs
            else:
                break
        
        return hierarchy
    
    def visualize_topic_tree(self, hierarchy):
        """
        Create interactive tree visualization of topic hierarchy
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = go.Figure()
        
        # Build tree structure
        node_trace = go.Scatter(
            x=[], y=[], 
            mode='markers+text',
            marker=dict(size=[], color=[], line=dict(width=2)),
            text=[], textposition="middle center",
            hovertemplate='%{text}<extra></extra>'
        )
        
        edge_trace = go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        )
        
        # Position nodes in hierarchy
        y_positions = {}
        x_positions = {}
        
        for level_name, level_data in hierarchy.items():
            level_num = int(level_name.split('_')[1])
            y_pos = level_num
            
            topic_info = level_data['topic_info']
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            for i, (_, row) in enumerate(valid_topics.iterrows()):
                topic_id = row['Topic']
                x_pos = i - len(valid_topics) / 2
                
                node_trace['x'] += (x_pos,)
                node_trace['y'] += (y_pos,)
                node_trace['marker']['size'] += (max(10, min(30, row['Count'] / 10)),)
                node_trace['marker']['color'] += (topic_id,)
                
                topic_words = level_data['model'].get_topic(topic_id)[:3]
                topic_label = f"T{topic_id}: {', '.join([w for w, _ in topic_words])}"
                node_trace['text'] += (topic_label,)
                
                x_positions[f"{level_name}_{topic_id}"] = x_pos
                y_positions[f"{level_name}_{topic_id}"] = y_pos
        
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        
        fig.update_layout(
            title="Hierarchical Topic Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Topic hierarchy from coarse (top) to fine-grained (bottom)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def analyze_topic_specificity(self, hierarchy):
        """
        Analyze how topics become more specific at deeper levels
        """
        specificity_analysis = {}
        
        for level_name, level_data in hierarchy.items():
            model = level_data['model']
            topic_specificities = []
            
            for topic_id in model.get_topics():
                if topic_id == -1:
                    continue
                
                topic_words = [word for word, _ in model.get_topic(topic_id)[:20]]
                
                # Calculate specificity metrics
                word_lengths = [len(word) for word in topic_words]
                avg_word_length = np.mean(word_lengths)
                
                # Technical term ratio (words > 6 characters)
                technical_ratio = sum(1 for word in topic_words if len(word) > 6) / len(topic_words)
                
                # Uniqueness score (how many words appear in other topics)
                other_topics_words = []
                for other_id in model.get_topics():
                    if other_id != topic_id and other_id != -1:
                        other_topics_words.extend([w for w, _ in model.get_topic(other_id)[:20]])
                
                unique_words = sum(1 for word in topic_words if word not in other_topics_words)
                uniqueness_score = unique_words / len(topic_words)
                
                topic_specificities.append({
                    'topic_id': topic_id,
                    'avg_word_length': avg_word_length,
                    'technical_ratio': technical_ratio,
                    'uniqueness_score': uniqueness_score,
                    'specificity_score': (avg_word_length + technical_ratio + uniqueness_score) / 3
                })
            
            specificity_analysis[level_name] = {
                'topics': topic_specificities,
                'avg_specificity': np.mean([t['specificity_score'] for t in topic_specificities]),
                'specificity_std': np.std([t['specificity_score'] for t in topic_specificities])
            }
        
        return specificity_analysis
```

## 🔍 **Advanced Topic Quality Evaluation**

### **Exercise 3: Topic Coherence and Quality Metrics**
```python
class TopicQualityEvaluator:
    def __init__(self):
        self.coherence_methods = ['c_v', 'c_npmi', 'c_uci', 'u_mass']
        self.quality_metrics = {}
        
    def comprehensive_topic_evaluation(self, topic_model, documents, corpus=None):
        """
        Evaluate topic quality using multiple coherence measures
        """
        if corpus is None:
            corpus = [doc.lower().split() for doc in documents]
        
        topics = topic_model.get_topics()
        evaluation_results = {}
        
        for coherence_method in self.coherence_methods:
            coherence_scores = self._calculate_coherence(
                topics, corpus, method=coherence_method
            )
            evaluation_results[coherence_method] = coherence_scores
        
        # Additional quality metrics
        evaluation_results['diversity'] = self._calculate_topic_diversity(topics)
        evaluation_results['coverage'] = self._calculate_topic_coverage(topic_model, documents)
        evaluation_results['stability'] = self._calculate_topic_stability(topic_model, documents)
        evaluation_results['interpretability'] = self._calculate_interpretability(topics)
        
        return evaluation_results
    
    def _calculate_coherence(self, topics, corpus, method='c_v', top_n=10):
        """
        Calculate topic coherence using different methods
        """
        try:
            from gensim.models import CoherenceModel
            from gensim.corpora import Dictionary
            
            # Prepare data for Gensim
            dictionary = Dictionary(corpus)
            bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
            
            topic_words = []
            for topic_id, topic in topics.items():
                if topic_id != -1:
                    words = [word for word, _ in topic[:top_n]]
                    topic_words.append(words)
            
            if not topic_words:
                return {'avg_coherence': 0.0, 'topic_scores': []}
            
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=corpus,
                corpus=bow_corpus,
                dictionary=dictionary,
                coherence=method
            )
            
            avg_coherence = coherence_model.get_coherence()
            topic_scores = coherence_model.get_coherence_per_topic()
            
            return {
                'avg_coherence': avg_coherence,
                'topic_scores': topic_scores,
                'method': method
            }
            
        except ImportError:
            # Fallback to simple word co-occurrence
            return self._simple_coherence_fallback(topics, corpus, top_n)
    
    def _simple_coherence_fallback(self, topics, corpus, top_n=10):
        """
        Simple coherence calculation without Gensim
        """
        topic_coherences = []
        
        # Create word co-occurrence matrix
        vocab = set(word for doc in corpus for word in doc)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
        
        for doc in corpus:
            for i, word1 in enumerate(doc):
                for word2 in doc[i+1:]:
                    if word1 in word_to_idx and word2 in word_to_idx:
                        idx1, idx2 = word_to_idx[word1], word_to_idx[word2]
                        cooccurrence_matrix[idx1, idx2] += 1
                        cooccurrence_matrix[idx2, idx1] += 1
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:top_n]]
            topic_indices = [word_to_idx[word] for word in topic_words if word in word_to_idx]
            
            if len(topic_indices) < 2:
                topic_coherences.append(0.0)
                continue
            
            coherence_sum = 0
            pair_count = 0
            
            for i, idx1 in enumerate(topic_indices):
                for idx2 in topic_indices[i+1:]:
                    coherence_sum += cooccurrence_matrix[idx1, idx2]
                    pair_count += 1
            
            avg_coherence = coherence_sum / pair_count if pair_count > 0 else 0
            topic_coherences.append(avg_coherence)
        
        return {
            'avg_coherence': np.mean(topic_coherences),
            'topic_scores': topic_coherences,
            'method': 'simple_cooccurrence'
        }
    
    def _calculate_topic_diversity(self, topics, top_n=10):
        """
        Calculate diversity of topics (how unique the words are)
        """
        all_words = []
        topic_words = []
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            words = [word for word, _ in topic[:top_n]]
            topic_words.append(words)
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words
    
    def _calculate_topic_coverage(self, topic_model, documents):
        """
        Calculate what percentage of documents are assigned to meaningful topics
        """
        topics = topic_model.topics_
        meaningful_assignments = sum(1 for topic in topics if topic != -1)
        total_documents = len(topics)
        
        return meaningful_assignments / total_documents if total_documents > 0 else 0.0
    
    def _calculate_topic_stability(self, topic_model, documents, n_runs=3):
        """
        Calculate topic stability across multiple runs
        """
        if len(documents) < 1000:  # Skip for small datasets
            return 0.8  # Default stability score
        
        original_topics = topic_model.get_topics()
        stability_scores = []
        
        # Sample documents for stability testing
        sample_size = min(1000, len(documents))
        
        for run in range(n_runs):
            sample_indices = np.random.choice(len(documents), sample_size, replace=False)
            sample_docs = [documents[i] for i in sample_indices]
            
            try:
                # Create new model with same parameters
                stability_model = BERTopic(
                    min_topic_size=max(10, sample_size // 50),
                    nr_topics="auto"
                ).fit(sample_docs)
                
                stability_topics = stability_model.get_topics()
                
                # Calculate topic overlap
                overlap_score = self._calculate_topic_overlap(original_topics, stability_topics)
                stability_scores.append(overlap_score)
                
            except:
                stability_scores.append(0.5)  # Default moderate stability
        
        return np.mean(stability_scores)
    
    def _calculate_topic_overlap(self, topics1, topics2, top_n=10):
        """
        Calculate overlap between two sets of topics
        """
        if not topics1 or not topics2:
            return 0.0
        
        overlap_scores = []
        
        for topic_id1, topic1 in topics1.items():
            if topic_id1 == -1:
                continue
            
            words1 = set(word for word, _ in topic1[:top_n])
            best_overlap = 0
            
            for topic_id2, topic2 in topics2.items():
                if topic_id2 == -1:
                    continue
                
                words2 = set(word for word, _ in topic2[:top_n])
                jaccard_sim = len(words1 & words2) / len(words1 | words2)
                best_overlap = max(best_overlap, jaccard_sim)
            
            overlap_scores.append(best_overlap)
        
        return np.mean(overlap_scores) if overlap_scores else 0.0
    
    def _calculate_interpretability(self, topics, top_n=10):
        """
        Calculate topic interpretability based on word clarity
        """
        interpretability_scores = []
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:top_n]]
            
            # Word length (longer words often more specific)
            avg_word_length = np.mean([len(word) for word in topic_words])
            
            # Word frequency spread (more even = better)
            word_weights = [weight for _, weight in topic[:top_n]]
            weight_entropy = -np.sum([w * np.log(w + 1e-10) for w in word_weights])
            
            # Combined interpretability score
            interpretability = (avg_word_length / 10 + weight_entropy / 5) / 2
            interpretability_scores.append(min(1.0, interpretability))
        
        return np.mean(interpretability_scores) if interpretability_scores else 0.0

# Example usage
evaluator = TopicQualityEvaluator()
```

## 🔄 **Cross-Domain Topic Transfer**

### **Exercise 4: Domain Adaptation for Topics**
```python
class CrossDomainTopicTransfer:
    def __init__(self):
        self.source_model = None
        self.target_models = {}
        self.transfer_mappings = {}
        
    def train_source_domain(self, source_documents, domain_name="source"):
        """
        Train topic model on source domain
        """
        print(f"Training source domain model: {domain_name}")
        
        self.source_model = BERTopic(
            min_topic_size=50,
            nr_topics="auto"
        ).fit(source_documents)
        
        source_topics = self.source_model.get_topics()
        
        return {
            'model': self.source_model,
            'n_topics': len(source_topics) - (1 if -1 in source_topics else 0),
            'topic_info': self.source_model.get_topic_info()
        }
    
    def adapt_to_target_domain(self, target_documents, domain_name, 
                              adaptation_method="guided_topics"):
        """
        Adapt source topics to target domain using different strategies
        """
        print(f"Adapting to target domain: {domain_name}")
        
        if adaptation_method == "guided_topics":
            return self._guided_topic_adaptation(target_documents, domain_name)
        elif adaptation_method == "transfer_learning":
            return self._transfer_learning_adaptation(target_documents, domain_name)
        elif adaptation_method == "zero_shot":
            return self._zero_shot_adaptation(target_documents, domain_name)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    def _guided_topic_adaptation(self, target_documents, domain_name):
        """
        Use source topic keywords to guide target domain clustering
        """
        if not self.source_model:
            raise ValueError("Source model must be trained first")
        
        # Extract source topic keywords
        source_topics = self.source_model.get_topics()
        seed_topic_list = []
        
        for topic_id, topic in source_topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:5]]
            seed_topic_list.append(topic_words)
        
        # Create guided BERTopic model
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
        
        target_model = BERTopic(
            min_topic_size=max(10, len(target_documents) // 50),
            seed_topic_list=seed_topic_list,
            nr_topics=len(seed_topic_list)
        ).fit(target_documents)
        
        self.target_models[domain_name] = target_model
        
        # Calculate transfer success
        transfer_quality = self._evaluate_transfer_quality(
            self.source_model, target_model
        )
        
        return {
            'model': target_model,
            'method': 'guided_topics',
            'transfer_quality': transfer_quality,
            'topic_info': target_model.get_topic_info()
        }
    
    def _transfer_learning_adaptation(self, target_documents, domain_name):
        """
        Use source embeddings as initialization for target clustering
        """
        # Extract source document embeddings
        source_embeddings = self.source_model.embedding_model.encode(
            target_documents[:1000]  # Sample for speed
        )
        
        # Use source UMAP and HDBSCAN models as initialization
        target_model = BERTopic(
            embedding_model=self.source_model.embedding_model,
            umap_model=self.source_model.umap_model,
            hdbscan_model=self.source_model.hdbscan_model,
            min_topic_size=max(10, len(target_documents) // 50)
        ).fit(target_documents)
        
        self.target_models[domain_name] = target_model
        
        transfer_quality = self._evaluate_transfer_quality(
            self.source_model, target_model
        )
        
        return {
            'model': target_model,
            'method': 'transfer_learning',
            'transfer_quality': transfer_quality,
            'topic_info': target_model.get_topic_info()
        }
    
    def _zero_shot_adaptation(self, target_documents, domain_name):
        """
        Apply source model directly to target documents
        """
        # Transform target documents using source model
        target_topics, target_probs = self.source_model.transform(target_documents)
        
        # Calculate confidence in zero-shot predictions
        confidence_scores = np.max(target_probs, axis=1)
        high_confidence_mask = confidence_scores > 0.5
        
        zero_shot_results = {
            'topics': target_topics,
            'probabilities': target_probs,
            'confidence_scores': confidence_scores,
            'high_confidence_ratio': np.mean(high_confidence_mask),
            'avg_confidence': np.mean(confidence_scores)
        }
        
        return {
            'results': zero_shot_results,
            'method': 'zero_shot',
            'source_model': self.source_model
        }
    
    def _evaluate_transfer_quality(self, source_model, target_model):
        """
        Evaluate quality of domain transfer
        """
        source_topics = source_model.get_topics()
        target_topics = target_model.get_topics()
        
        # Topic overlap analysis
        topic_overlaps = []
        
        for source_id, source_topic in source_topics.items():
            if source_id == -1:
                continue
            
            source_words = set(word for word, _ in source_topic[:10])
            best_overlap = 0
            
            for target_id, target_topic in target_topics.items():
                if target_id == -1:
                    continue
                
                target_words = set(word for word, _ in target_topic[:10])
                jaccard_sim = len(source_words & target_words) / len(source_words | target_words)
                best_overlap = max(best_overlap, jaccard_sim)
            
            topic_overlaps.append(best_overlap)
        
        return {
            'avg_topic_overlap': np.mean(topic_overlaps),
            'transfer_success_rate': sum(1 for overlap in topic_overlaps if overlap > 0.3) / len(topic_overlaps),
            'topic_preservation': np.mean(topic_overlaps)
        }
    
    def analyze_domain_differences(self, domains_data):
        """
        Analyze differences between multiple domains
        """
        domain_analysis = {}
        
        for domain_name, domain_docs in domains_data.items():
            # Basic statistics
            doc_lengths = [len(doc.split()) for doc in domain_docs]
            
            # Vocabulary analysis
            all_words = []
            for doc in domain_docs:
                all_words.extend(doc.lower().split())
            
            vocab_size = len(set(all_words))
            avg_word_freq = len(all_words) / vocab_size if vocab_size > 0 else 0
            
            domain_analysis[domain_name] = {
                'n_documents': len(domain_docs),
                'avg_doc_length': np.mean(doc_lengths),
                'vocab_size': vocab_size,
                'avg_word_frequency': avg_word_freq,
                'vocabulary_richness': vocab_size / len(all_words) if all_words else 0
            }
        
        # Cross-domain similarity
        domain_names = list(domains_data.keys())
        similarity_matrix = np.zeros((len(domain_names), len(domain_names)))
        
        for i, domain1 in enumerate(domain_names):
            for j, domain2 in enumerate(domain_names):
                if i != j:
                    sim = self._calculate_domain_similarity(
                        domains_data[domain1], domains_data[domain2]
                    )
                    similarity_matrix[i, j] = sim
                else:
                    similarity_matrix[i, j] = 1.0
        
        domain_analysis['cross_domain_similarity'] = {
            'matrix': similarity_matrix,
            'domain_names': domain_names,
            'avg_similarity': np.mean(similarity_matrix[similarity_matrix != 1.0])
        }
        
        return domain_analysis
    
    def _calculate_domain_similarity(self, docs1, docs2, sample_size=500):
        """
        Calculate similarity between two document collections
        """
        # Sample documents for efficiency
        sample1 = docs1[:sample_size] if len(docs1) > sample_size else docs1
        sample2 = docs2[:sample_size] if len(docs2) > sample_size else docs2
        
        # Extract vocabularies
        vocab1 = set()
        vocab2 = set()
        
        for doc in sample1:
            vocab1.update(doc.lower().split())
        
        for doc in sample2:
            vocab2.update(doc.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(vocab1 & vocab2)
        union = len(vocab1 | vocab2)
        
        return intersection / union if union > 0 else 0.0

# Example cross-domain analysis
transfer_system = CrossDomainTopicTransfer()
```

## 🎨 **Interactive Topic Exploration**

# Chapter 5 Advanced Topic Modeling Exercises

## 🎯 **Dynamic Topic Modeling**

### **Exercise 1: Temporal Topic Evolution**
```python
class TemporalTopicAnalyzer:
    def __init__(self, time_slices=5):
        self.time_slices = time_slices
        self.models_by_time = {}
        self.topic_alignments = {}
        
    def create_dynamic_topics(self, documents, timestamps, method='sliding_window'):
        """
        Create topic models for different time periods
        """
        sorted_indices = np.argsort(timestamps)
        sorted_docs = [documents[i] for i in sorted_indices]
        sorted_times = [timestamps[i] for i in sorted_indices]
        
        if method == 'sliding_window':
            return self._sliding_window_approach(sorted_docs, sorted_times)
        elif method == 'fixed_periods':
            return self._fixed_period_approach(sorted_docs, sorted_times)
        else:
            return self._evolutionary_approach(sorted_docs, sorted_times)
    
    def _sliding_window_approach(self, documents, timestamps):
        """
        Create overlapping time windows for smooth topic evolution
        """
        window_size = len(documents) // self.time_slices
        overlap = window_size // 2
        
        time_models = {}
        
        for i in range(0, len(documents) - window_size, overlap):
            window_docs = documents[i:i + window_size]
            window_start = timestamps[i]
            window_end = timestamps[min(i + window_size - 1, len(timestamps) - 1)]
            
            topic_model = BERTopic(
                min_topic_size=max(10, len(window_docs) // 20),
                nr_topics="auto"
            ).fit(window_docs)
            
            time_models[f"{window_start}-{window_end}"] = {
                'model': topic_model,
                'documents': window_docs,
                'start_time': window_start,
                'end_time': window_end,
                'doc_count': len(window_docs)
            }
        
        return time_models
    
    def track_topic_evolution(self, time_models, similarity_threshold=0.7):
        """
        Track how topics evolve across time periods
        """
        evolution_chains = []
        model_keys = sorted(time_models.keys())
        
        for i in range(len(model_keys) - 1):
            current_model = time_models[model_keys[i]]['model']
            next_model = time_models[model_keys[i + 1]]['model']
            
            topic_similarities = self._calculate_topic_similarities(
                current_model, next_model
            )
            
            evolution_chains.append({
                'from_period': model_keys[i],
                'to_period': model_keys[i + 1],
                'topic_matches': topic_similarities,
                'emerging_topics': self._find_emerging_topics(topic_similarities),
                'declining_topics': self._find_declining_topics(topic_similarities)
            })
        
        return evolution_chains
    
    def _calculate_topic_similarities(self, model1, model2):
        """
        Calculate similarity between topics across time periods
        """
        similarities = {}
        
        for topic1_id in model1.get_topics():
            if topic1_id == -1:
                continue
                
            topic1_words = [word for word, _ in model1.get_topic(topic1_id)[:10]]
            best_match = -1
            best_score = 0
            
            for topic2_id in model2.get_topics():
                if topic2_id == -1:
                    continue
                    
                topic2_words = [word for word, _ in model2.get_topic(topic2_id)[:10]]
                
                overlap = len(set(topic1_words) & set(topic2_words))
                jaccard_sim = overlap / len(set(topic1_words) | set(topic2_words))
                
                if jaccard_sim > best_score:
                    best_score = jaccard_sim
                    best_match = topic2_id
            
            similarities[topic1_id] = {
                'matched_topic': best_match,
                'similarity_score': best_score,
                'words_from': topic1_words,
                'words_to': [word for word, _ in model2.get_topic(best_match)[:10]] if best_match != -1 else []
            }
        
        return similarities
    
    def _find_emerging_topics(self, similarities):
        """
        Find topics that appear in the new time period
        """
        matched_topics = {sim['matched_topic'] for sim in similarities.values() if sim['matched_topic'] != -1}
        return [topic for topic in similarities.keys() if topic not in matched_topics]
    
    def _find_declining_topics(self, similarities):
        """
        Find topics that disappear or significantly change
        """
        return [topic for topic, sim in similarities.items() if sim['similarity_score'] < 0.3]

temporal_analyzer = TemporalTopicAnalyzer()
```

### **Exercise 2: Hierarchical Topic Discovery**
```python
class HierarchicalTopicExplorer:
    def __init__(self):
        self.hierarchy_levels = {}
        self.topic_tree = {}
        
    def build_topic_hierarchy(self, documents, max_levels=3):
        """
        Build hierarchical topic structure from coarse to fine-grained
        """
        hierarchy = {}
        current_docs = documents
        
        for level in range(max_levels):
            print(f"Building level {level + 1} hierarchy...")
            
            # Adjust clustering parameters by level
            min_cluster_size = max(50 // (level + 1), 10)
            
            topic_model = BERTopic(
                min_topic_size=min_cluster_size,
                nr_topics="auto"
            ).fit(current_docs)
            
            topics = topic_model.topics_
            topic_info = topic_model.get_topic_info()
            
            level_data = {
                'model': topic_model,
                'topics': topics,
                'topic_info': topic_info,
                'documents': current_docs,
                'subtopics': {}
            }
            
            # Create subtopics for next level
            if level < max_levels - 1:
                for topic_id in topic_info['Topic']:
                    if topic_id == -1:
                        continue
                    
                    topic_docs = [current_docs[i] for i, t in enumerate(topics) if t == topic_id]
                    
                    if len(topic_docs) > min_cluster_size * 2:
                        level_data['subtopics'][topic_id] = topic_docs
            
            hierarchy[f'level_{level + 1}'] = level_data
            
            # Prepare documents for next level (only large topics)
            if level < max_levels - 1 and level_data['subtopics']:
                next_level_docs = []
                for subtopic_docs in level_data['subtopics'].values():
                    next_level_docs.extend(subtopic_docs)
                current_docs = next_level_docs
            else:
                break
        
        return hierarchy
    
    def visualize_topic_tree(self, hierarchy):
        """
        Create interactive tree visualization of topic hierarchy
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = go.Figure()
        
        # Build tree structure
        node_trace = go.Scatter(
            x=[], y=[], 
            mode='markers+text',
            marker=dict(size=[], color=[], line=dict(width=2)),
            text=[], textposition="middle center",
            hovertemplate='%{text}<extra></extra>'
        )
        
        edge_trace = go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none'
        )
        
        # Position nodes in hierarchy
        y_positions = {}
        x_positions = {}
        
        for level_name, level_data in hierarchy.items():
            level_num = int(level_name.split('_')[1])
            y_pos = level_num
            
            topic_info = level_data['topic_info']
            valid_topics = topic_info[topic_info['Topic'] != -1]
            
            for i, (_, row) in enumerate(valid_topics.iterrows()):
                topic_id = row['Topic']
                x_pos = i - len(valid_topics) / 2
                
                node_trace['x'] += (x_pos,)
                node_trace['y'] += (y_pos,)
                node_trace['marker']['size'] += (max(10, min(30, row['Count'] / 10)),)
                node_trace['marker']['color'] += (topic_id,)
                
                topic_words = level_data['model'].get_topic(topic_id)[:3]
                topic_label = f"T{topic_id}: {', '.join([w for w, _ in topic_words])}"
                node_trace['text'] += (topic_label,)
                
                x_positions[f"{level_name}_{topic_id}"] = x_pos
                y_positions[f"{level_name}_{topic_id}"] = y_pos
        
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        
        fig.update_layout(
            title="Hierarchical Topic Structure",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Topic hierarchy from coarse (top) to fine-grained (bottom)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def analyze_topic_specificity(self, hierarchy):
        """
        Analyze how topics become more specific at deeper levels
        """
        specificity_analysis = {}
        
        for level_name, level_data in hierarchy.items():
            model = level_data['model']
            topic_specificities = []
            
            for topic_id in model.get_topics():
                if topic_id == -1:
                    continue
                
                topic_words = [word for word, _ in model.get_topic(topic_id)[:20]]
                
                # Calculate specificity metrics
                word_lengths = [len(word) for word in topic_words]
                avg_word_length = np.mean(word_lengths)
                
                # Technical term ratio (words > 6 characters)
                technical_ratio = sum(1 for word in topic_words if len(word) > 6) / len(topic_words)
                
                # Uniqueness score (how many words appear in other topics)
                other_topics_words = []
                for other_id in model.get_topics():
                    if other_id != topic_id and other_id != -1:
                        other_topics_words.extend([w for w, _ in model.get_topic(other_id)[:20]])
                
                unique_words = sum(1 for word in topic_words if word not in other_topics_words)
                uniqueness_score = unique_words / len(topic_words)
                
                topic_specificities.append({
                    'topic_id': topic_id,
                    'avg_word_length': avg_word_length,
                    'technical_ratio': technical_ratio,
                    'uniqueness_score': uniqueness_score,
                    'specificity_score': (avg_word_length + technical_ratio + uniqueness_score) / 3
                })
            
            specificity_analysis[level_name] = {
                'topics': topic_specificities,
                'avg_specificity': np.mean([t['specificity_score'] for t in topic_specificities]),
                'specificity_std': np.std([t['specificity_score'] for t in topic_specificities])
            }
        
        return specificity_analysis
```

## 🔍 **Advanced Topic Quality Evaluation**

### **Exercise 3: Topic Coherence and Quality Metrics**
```python
class TopicQualityEvaluator:
    def __init__(self):
        self.coherence_methods = ['c_v', 'c_npmi', 'c_uci', 'u_mass']
        self.quality_metrics = {}
        
    def comprehensive_topic_evaluation(self, topic_model, documents, corpus=None):
        """
        Evaluate topic quality using multiple coherence measures
        """
        if corpus is None:
            corpus = [doc.lower().split() for doc in documents]
        
        topics = topic_model.get_topics()
        evaluation_results = {}
        
        for coherence_method in self.coherence_methods:
            coherence_scores = self._calculate_coherence(
                topics, corpus, method=coherence_method
            )
            evaluation_results[coherence_method] = coherence_scores
        
        # Additional quality metrics
        evaluation_results['diversity'] = self._calculate_topic_diversity(topics)
        evaluation_results['coverage'] = self._calculate_topic_coverage(topic_model, documents)
        evaluation_results['stability'] = self._calculate_topic_stability(topic_model, documents)
        evaluation_results['interpretability'] = self._calculate_interpretability(topics)
        
        return evaluation_results
    
    def _calculate_coherence(self, topics, corpus, method='c_v', top_n=10):
        """
        Calculate topic coherence using different methods
        """
        try:
            from gensim.models import CoherenceModel
            from gensim.corpora import Dictionary
            
            # Prepare data for Gensim
            dictionary = Dictionary(corpus)
            bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
            
            topic_words = []
            for topic_id, topic in topics.items():
                if topic_id != -1:
                    words = [word for word, _ in topic[:top_n]]
                    topic_words.append(words)
            
            if not topic_words:
                return {'avg_coherence': 0.0, 'topic_scores': []}
            
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=corpus,
                corpus=bow_corpus,
                dictionary=dictionary,
                coherence=method
            )
            
            avg_coherence = coherence_model.get_coherence()
            topic_scores = coherence_model.get_coherence_per_topic()
            
            return {
                'avg_coherence': avg_coherence,
                'topic_scores': topic_scores,
                'method': method
            }
            
        except ImportError:
            # Fallback to simple word co-occurrence
            return self._simple_coherence_fallback(topics, corpus, top_n)
    
    def _simple_coherence_fallback(self, topics, corpus, top_n=10):
        """
        Simple coherence calculation without Gensim
        """
        topic_coherences = []
        
        # Create word co-occurrence matrix
        vocab = set(word for doc in corpus for word in doc)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
        
        for doc in corpus:
            for i, word1 in enumerate(doc):
                for word2 in doc[i+1:]:
                    if word1 in word_to_idx and word2 in word_to_idx:
                        idx1, idx2 = word_to_idx[word1], word_to_idx[word2]
                        cooccurrence_matrix[idx1, idx2] += 1
                        cooccurrence_matrix[idx2, idx1] += 1
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:top_n]]
            topic_indices = [word_to_idx[word] for word in topic_words if word in word_to_idx]
            
            if len(topic_indices) < 2:
                topic_coherences.append(0.0)
                continue
            
            coherence_sum = 0
            pair_count = 0
            
            for i, idx1 in enumerate(topic_indices):
                for idx2 in topic_indices[i+1:]:
                    coherence_sum += cooccurrence_matrix[idx1, idx2]
                    pair_count += 1
            
            avg_coherence = coherence_sum / pair_count if pair_count > 0 else 0
            topic_coherences.append(avg_coherence)
        
        return {
            'avg_coherence': np.mean(topic_coherences),
            'topic_scores': topic_coherences,
            'method': 'simple_cooccurrence'
        }
    
    def _calculate_topic_diversity(self, topics, top_n=10):
        """
        Calculate diversity of topics (how unique the words are)
        """
        all_words = []
        topic_words = []
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            words = [word for word, _ in topic[:top_n]]
            topic_words.append(words)
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words
    
    def _calculate_topic_coverage(self, topic_model, documents):
        """
        Calculate what percentage of documents are assigned to meaningful topics
        """
        topics = topic_model.topics_
        meaningful_assignments = sum(1 for topic in topics if topic != -1)
        total_documents = len(topics)
        
        return meaningful_assignments / total_documents if total_documents > 0 else 0.0
    
    def _calculate_topic_stability(self, topic_model, documents, n_runs=3):
        """
        Calculate topic stability across multiple runs
        """
        if len(documents) < 1000:  # Skip for small datasets
            return 0.8  # Default stability score
        
        original_topics = topic_model.get_topics()
        stability_scores = []
        
        # Sample documents for stability testing
        sample_size = min(1000, len(documents))
        
        for run in range(n_runs):
            sample_indices = np.random.choice(len(documents), sample_size, replace=False)
            sample_docs = [documents[i] for i in sample_indices]
            
            try:
                # Create new model with same parameters
                stability_model = BERTopic(
                    min_topic_size=max(10, sample_size // 50),
                    nr_topics="auto"
                ).fit(sample_docs)
                
                stability_topics = stability_model.get_topics()
                
                # Calculate topic overlap
                overlap_score = self._calculate_topic_overlap(original_topics, stability_topics)
                stability_scores.append(overlap_score)
                
            except:
                stability_scores.append(0.5)  # Default moderate stability
        
        return np.mean(stability_scores)
    
    def _calculate_topic_overlap(self, topics1, topics2, top_n=10):
        """
        Calculate overlap between two sets of topics
        """
        if not topics1 or not topics2:
            return 0.0
        
        overlap_scores = []
        
        for topic_id1, topic1 in topics1.items():
            if topic_id1 == -1:
                continue
            
            words1 = set(word for word, _ in topic1[:top_n])
            best_overlap = 0
            
            for topic_id2, topic2 in topics2.items():
                if topic_id2 == -1:
                    continue
                
                words2 = set(word for word, _ in topic2[:top_n])
                jaccard_sim = len(words1 & words2) / len(words1 | words2)
                best_overlap = max(best_overlap, jaccard_sim)
            
            overlap_scores.append(best_overlap)
        
        return np.mean(overlap_scores) if overlap_scores else 0.0
    
    def _calculate_interpretability(self, topics, top_n=10):
        """
        Calculate topic interpretability based on word clarity
        """
        interpretability_scores = []
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:top_n]]
            
            # Word length (longer words often more specific)
            avg_word_length = np.mean([len(word) for word in topic_words])
            
            # Word frequency spread (more even = better)
            word_weights = [weight for _, weight in topic[:top_n]]
            weight_entropy = -np.sum([w * np.log(w + 1e-10) for w in word_weights])
            
            # Combined interpretability score
            interpretability = (avg_word_length / 10 + weight_entropy / 5) / 2
            interpretability_scores.append(min(1.0, interpretability))
        
        return np.mean(interpretability_scores) if interpretability_scores else 0.0

# Example usage
evaluator = TopicQualityEvaluator()
```

## 🔄 **Cross-Domain Topic Transfer**

### **Exercise 4: Domain Adaptation for Topics**
```python
class CrossDomainTopicTransfer:
    def __init__(self):
        self.source_model = None
        self.target_models = {}
        self.transfer_mappings = {}
        
    def train_source_domain(self, source_documents, domain_name="source"):
        """
        Train topic model on source domain
        """
        print(f"Training source domain model: {domain_name}")
        
        self.source_model = BERTopic(
            min_topic_size=50,
            nr_topics="auto"
        ).fit(source_documents)
        
        source_topics = self.source_model.get_topics()
        
        return {
            'model': self.source_model,
            'n_topics': len(source_topics) - (1 if -1 in source_topics else 0),
            'topic_info': self.source_model.get_topic_info()
        }
    
    def adapt_to_target_domain(self, target_documents, domain_name, 
                              adaptation_method="guided_topics"):
        """
        Adapt source topics to target domain using different strategies
        """
        print(f"Adapting to target domain: {domain_name}")
        
        if adaptation_method == "guided_topics":
            return self._guided_topic_adaptation(target_documents, domain_name)
        elif adaptation_method == "transfer_learning":
            return self._transfer_learning_adaptation(target_documents, domain_name)
        elif adaptation_method == "zero_shot":
            return self._zero_shot_adaptation(target_documents, domain_name)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    def _guided_topic_adaptation(self, target_documents, domain_name):
        """
        Use source topic keywords to guide target domain clustering
        """
        if not self.source_model:
            raise ValueError("Source model must be trained first")
        
        # Extract source topic keywords
        source_topics = self.source_model.get_topics()
        seed_topic_list = []
        
        for topic_id, topic in source_topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:5]]
            seed_topic_list.append(topic_words)
        
        # Create guided BERTopic model
        from bertopic import BERTopic
        from bertopic.dimensionality import BaseDimensionalityReduction
        
        target_model = BERTopic(
            min_topic_size=max(10, len(target_documents) // 50),
            seed_topic_list=seed_topic_list,
            nr_topics=len(seed_topic_list)
        ).fit(target_documents)
        
        self.target_models[domain_name] = target_model
        
        # Calculate transfer success
        transfer_quality = self._evaluate_transfer_quality(
            self.source_model, target_model
        )
        
        return {
            'model': target_model,
            'method': 'guided_topics',
            'transfer_quality': transfer_quality,
            'topic_info': target_model.get_topic_info()
        }
    
    def _transfer_learning_adaptation(self, target_documents, domain_name):
        """
        Use source embeddings as initialization for target clustering
        """
        # Extract source document embeddings
        source_embeddings = self.source_model.embedding_model.encode(
            target_documents[:1000]  # Sample for speed
        )
        
        # Use source UMAP and HDBSCAN models as initialization
        target_model = BERTopic(
            embedding_model=self.source_model.embedding_model,
            umap_model=self.source_model.umap_model,
            hdbscan_model=self.source_model.hdbscan_model,
            min_topic_size=max(10, len(target_documents) // 50)
        ).fit(target_documents)
        
        self.target_models[domain_name] = target_model
        
        transfer_quality = self._evaluate_transfer_quality(
            self.source_model, target_model
        )
        
        return {
            'model': target_model,
            'method': 'transfer_learning',
            'transfer_quality': transfer_quality,
            'topic_info': target_model.get_topic_info()
        }
    
    def _zero_shot_adaptation(self, target_documents, domain_name):
        """
        Apply source model directly to target documents
        """
        # Transform target documents using source model
        target_topics, target_probs = self.source_model.transform(target_documents)
        
        # Calculate confidence in zero-shot predictions
        confidence_scores = np.max(target_probs, axis=1)
        high_confidence_mask = confidence_scores > 0.5
        
        zero_shot_results = {
            'topics': target_topics,
            'probabilities': target_probs,
            'confidence_scores': confidence_scores,
            'high_confidence_ratio': np.mean(high_confidence_mask),
            'avg_confidence': np.mean(confidence_scores)
        }
        
        return {
            'results': zero_shot_results,
            'method': 'zero_shot',
            'source_model': self.source_model
        }
    
    def _evaluate_transfer_quality(self, source_model, target_model):
        """
        Evaluate quality of domain transfer
        """
        source_topics = source_model.get_topics()
        target_topics = target_model.get_topics()
        
        # Topic overlap analysis
        topic_overlaps = []
        
        for source_id, source_topic in source_topics.items():
            if source_id == -1:
                continue
            
            source_words = set(word for word, _ in source_topic[:10])
            best_overlap = 0
            
            for target_id, target_topic in target_topics.items():
                if target_id == -1:
                    continue
                
                target_words = set(word for word, _ in target_topic[:10])
                jaccard_sim = len(source_words & target_words) / len(source_words | target_words)
                best_overlap = max(best_overlap, jaccard_sim)
            
            topic_overlaps.append(best_overlap)
        
        return {
            'avg_topic_overlap': np.mean(topic_overlaps),
            'transfer_success_rate': sum(1 for overlap in topic_overlaps if overlap > 0.3) / len(topic_overlaps),
            'topic_preservation': np.mean(topic_overlaps)
        }
    
    def analyze_domain_differences(self, domains_data):
        """
        Analyze differences between multiple domains
        """
        domain_analysis = {}
        
        for domain_name, domain_docs in domains_data.items():
            # Basic statistics
            doc_lengths = [len(doc.split()) for doc in domain_docs]
            
            # Vocabulary analysis
            all_words = []
            for doc in domain_docs:
                all_words.extend(doc.lower().split())
            
            vocab_size = len(set(all_words))
            avg_word_freq = len(all_words) / vocab_size if vocab_size > 0 else 0
            
            domain_analysis[domain_name] = {
                'n_documents': len(domain_docs),
                'avg_doc_length': np.mean(doc_lengths),
                'vocab_size': vocab_size,
                'avg_word_frequency': avg_word_freq,
                'vocabulary_richness': vocab_size / len(all_words) if all_words else 0
            }
        
        # Cross-domain similarity
        domain_names = list(domains_data.keys())
        similarity_matrix = np.zeros((len(domain_names), len(domain_names)))
        
        for i, domain1 in enumerate(domain_names):
            for j, domain2 in enumerate(domain_names):
                if i != j:
                    sim = self._calculate_domain_similarity(
                        domains_data[domain1], domains_data[domain2]
                    )
                    similarity_matrix[i, j] = sim
                else:
                    similarity_matrix[i, j] = 1.0
        
        domain_analysis['cross_domain_similarity'] = {
            'matrix': similarity_matrix,
            'domain_names': domain_names,
            'avg_similarity': np.mean(similarity_matrix[similarity_matrix != 1.0])
        }
        
        return domain_analysis
    
    def _calculate_domain_similarity(self, docs1, docs2, sample_size=500):
        """
        Calculate similarity between two document collections
        """
        # Sample documents for efficiency
        sample1 = docs1[:sample_size] if len(docs1) > sample_size else docs1
        sample2 = docs2[:sample_size] if len(docs2) > sample_size else docs2
        
        # Extract vocabularies
        vocab1 = set()
        vocab2 = set()
        
        for doc in sample1:
            vocab1.update(doc.lower().split())
        
        for doc in sample2:
            vocab2.update(doc.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(vocab1 & vocab2)
        union = len(vocab1 | vocab2)
        
        return intersection / union if union > 0 else 0.0

# Example cross-domain analysis
transfer_system = CrossDomainTopicTransfer()
```

## 🎨 **Interactive Topic Exploration**

### **Exercise 5: Advanced Visualization and Exploration Tools**
```python
class InteractiveTopicExplorer:
    def __init__(self, topic_model):
        self.topic_model = topic_model
        self.exploration_cache = {}
        
    def create_topic_dashboard(self, documents, titles=None):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        topics = self.topic_model.topics_
        topic_info = self.topic_model.get_topic_info()
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Topic Size Distribution', 'Word Frequency Heatmap',
                'Topic Similarity Network', 'Document-Topic Mapping',
                'Topic Evolution Timeline', 'Topic Quality Metrics'
            ],
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        valid_topics = topic_info[topic_info['Topic'] != -1].head(20)
        
        fig.add_trace(
            go.Bar(
                x=valid_topics['Count'],
                y=[f"Topic {t}" for t in valid_topics['Topic']],
                orientation='h',
                name='Topic Sizes'
            ),
            row=1, col=1
        )
        
        word_matrix = self._create_topic_word_matrix(valid_topics)
        fig.add_trace(
            go.Heatmap(
                z=word_matrix['values'],
                x=word_matrix['words'],
                y=word_matrix['topics'],
                colorscale='Blues',
                name='Word Frequencies'
            ),
            row=1, col=2
        )
        
        similarity_data = self._calculate_topic_similarities_for_network(valid_topics)
        fig.add_trace(
            go.Scatter(
                x=similarity_data['x'],
                y=similarity_data['y'],
                mode='markers+lines',
                marker=dict(size=10),
                name='Topic Network'
            ),
            row=2, col=1
        )
        
        if self.topic_model.umap_model:
            doc_embeddings = self.topic_model.umap_model.transform(
                self.topic_model.embedding_model.encode(documents[:1000])
            )
            doc_topics = topics[:1000]
            
            fig.add_trace(
                go.Scatter(
                    x=doc_embeddings[:, 0],
                    y=doc_embeddings[:, 1],
                    mode='markers',
                    marker=dict(
                        color=doc_topics,
                        colorscale='Set3',
                        size=3
                    ),
                    text=titles[:1000] if titles else None,
                    name='Documents'
                ),
                row=2, col=2
            )
        
        coherence_scores = self._calculate_topic_coherence_scores(valid_topics)
        fig.add_trace(
            go.Bar(
                x=[f"T{t}" for t in valid_topics['Topic']],
                y=coherence_scores,
                name='Coherence Scores'
            ),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Topic Analysis Dashboard")
        return fig
    
    def _create_topic_word_matrix(self, topic_info):
        topics_data = []
        words_data = []
        values_data = []
        
        all_words = set()
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            topic_words = self.topic_model.get_topic(topic_id)[:10]
            
            for word, score in topic_words:
                all_words.add(word)
        
        word_list = list(all_words)[:20]
        
        matrix = []
        topic_labels = []
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            topic_words_dict = dict(self.topic_model.get_topic(topic_id))
            
            row_values = []
            for word in word_list:
                row_values.append(topic_words_dict.get(word, 0))
            
            matrix.append(row_values)
            topic_labels.append(f"Topic {topic_id}")
        
        return {
            'values': matrix,
            'words': word_list,
            'topics': topic_labels
        }
    
    def _calculate_topic_similarities_for_network(self, topic_info):
        n_topics = len(topic_info)
        x_coords = []
        y_coords = []
        
        angle_step = 2 * np.pi / n_topics
        radius = 1
        
        for i in range(n_topics):
            angle = i * angle_step
            x_coords.append(radius * np.cos(angle))
            y_coords.append(radius * np.sin(angle))
        
        return {'x': x_coords, 'y': y_coords}
    
    def _calculate_topic_coherence_scores(self, topic_info):
        coherence_scores = []
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            topic_words = [word for word, _ in self.topic_model.get_topic(topic_id)[:10]]
            
            if len(topic_words) > 1:
                word_embeddings = self.topic_model.embedding_model.encode(topic_words)
                similarity_matrix = np.corrcoef(word_embeddings)
                coherence = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                coherence_scores.append(coherence)
            else:
                coherence_scores.append(0.0)
        
        return coherence_scores
    
    def create_topic_exploration_interface(self, documents):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        topic_info = self.topic_model.get_topic_info()
        valid_topics = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
        
        topic_dropdown = widgets.Dropdown(
            options=[(f"Topic {t}", t) for t in valid_topics],
            value=valid_topics[0],
            description='Topic:'
        )
        
        n_words_slider = widgets.IntSlider(
            value=10,
            min=5,
            max=20,
            step=1,
            description='Words:'
        )
        
        n_docs_slider = widgets.IntSlider(
            value=5,
            min=3,
            max=15,
            step=1,
            description='Documents:'
        )
        
        output_area = widgets.Output()
        
        def update_topic_display(change=None):
            with output_area:
                clear_output(wait=True)
                
                topic_id = topic_dropdown.value
                n_words = n_words_slider.value
                n_docs = n_docs_slider.value
                
                print(f"🎯 Topic {topic_id} Analysis")
                print("=" * 40)
                
                topic_words = self.topic_model.get_topic(topic_id)
                print(f"\n📝 Top {n_words} Keywords:")
                for word, score in topic_words[:n_words]:
                    print(f"  {word}: {score:.4f}")
                
                topic_docs = self._get_representative_documents(topic_id, documents, n_docs)
                print(f"\n📄 Representative Documents:")
                for i, doc in enumerate(topic_docs, 1):
                    print(f"  {i}. {doc[:200]}...")
                
                topic_size = topic_info[topic_info['Topic'] == topic_id]['Count'].iloc[0]
                print(f"\n📊 Topic Statistics:")
                print(f"  Documents: {topic_size}")
                print(f"  Percentage: {topic_size/len(documents)*100:.1f}%")
        
        topic_dropdown.observe(update_topic_display, names='value')
        n_words_slider.observe(update_topic_display, names='value')
        n_docs_slider.observe(update_topic_display, names='value')
        
        update_topic_display()
        
        interface = widgets.VBox([
            widgets.HBox([topic_dropdown, n_words_slider, n_docs_slider]),
            output_area
        ])
        
        return interface
    
    def _get_representative_documents(self, topic_id, documents, n_docs=5):
        topics = self.topic_model.topics_
        topic_indices = [i for i, t in enumerate(topics) if t == topic_id]
        
        if len(topic_indices) > n_docs:
            selected_indices = np.random.choice(topic_indices, n_docs, replace=False)
        else:
            selected_indices = topic_indices
        
        return [documents[i] for i in selected_indices]
    
    def export_topic_analysis_report(self, documents, output_file="topic_analysis_report.html"):
        topic_info = self.topic_model.get_topic_info()
        
        html_content = f"""
        <html>
        <head>
            <title>Topic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .topic {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; }}
                .topic-header {{ background-color: #f5f5f5; padding: 10px; margin: -20px -20px 20px -20px; }}
                .keywords {{ font-weight: bold; color: #333; }}
                .documents {{ margin-top: 15px; }}
                .doc {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Topic Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Documents: {len(documents)}</p>
            <p>Topics Found: {len(topic_info) - 1}</p>
        """
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            
            topic_words = self.topic_model.get_topic(topic_id)[:10]
            topic_docs = self._get_representative_documents(topic_id, documents, 3)
            
            html_content += f"""
            <div class="topic">
                <div class="topic-header">
                    <h2>Topic {topic_id}</h2>
                    <p>Documents: {row['Count']} ({row['Count']/len(documents)*100:.1f}%)</p>
                </div>
                
                <div class="keywords">
                    Keywords: {', '.join([f"{word} ({score:.3f})" for word, score in topic_words])}
                </div>
                
                <div class="documents">
                    <h4>Representative Documents:</h4>
            """
            
            for i, doc in enumerate(topic_docs, 1):
                html_content += f'<div class="doc">{i}. {doc[:300]}...</div>'
            
            html_content += "</div></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report exported to {output_file}")
        return output_file

explorer = InteractiveTopicExplorer(topic_model)
```

## 🚀 **Real-World Application Templates**

### **Exercise 6: Production-Ready Topic Modeling Pipeline**
```python
class ProductionTopicPipeline:
    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.models = {}
        self.preprocessing_cache = {}
        
    def _load_config(self, config_file):
        default_config = {
            'embedding_model': 'thenlper/gte-small',
            'umap_params': {'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'},
            'hdbscan_params': {'min_cluster_size': 50, 'metric': 'euclidean'},
            'min_topic_size': 20,
            'max_topics': 100,
            'preprocessing': {
                'remove_stopwords': True,
                'min_doc_length': 50,
                'max_doc_length': 5000,
                'language': 'english'
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def preprocess_documents(self, documents, custom_stopwords=None):
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        try:
            stop_words = set(stopwords.words(self.config['preprocessing']['language']))
        except:
            stop_words = set()
        
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        processed_docs = []
        
        for doc in documents:
            doc = str(doc).lower()
            doc = re.sub(r'[^a-zA-Z\s]', '', doc)
            doc = re.sub(r'\s+', ' ', doc).strip()
            
            if self.config['preprocessing']['remove_stopwords']:
                words = doc.split()
                doc = ' '.join([word for word in words if word not in stop_words])
            
            doc_length = len(doc.split())
            if (doc_length >= self.config['preprocessing']['min_doc_length'] and 
                doc_length <= self.config['preprocessing']['max_doc_length']):
                processed_docs.append(doc)
        
        print(f"Preprocessing: {len(documents)} -> {len(processed_docs)} documents")
        return processed_docs
    
    def train_production_model(self, documents, model_name="production_v1"):
        processed_docs = self.preprocess_documents(documents)
        
        embedding_model = SentenceTransformer(self.config['embedding_model'])
        
        umap_model = UMAP(**self.config['umap_params'], random_state=42)
        hdbscan_model = HDBSCAN(**self.config['hdbscan_params'])
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            min_topic_size=self.config['min_topic_size'],
            nr_topics=self.config['max_topics']
        ).fit(processed_docs)
        
        model_metadata = {
            'model': topic_model,
            'config': self.config,
            'training_docs': len(processed_docs),
            'topics_found': len(topic_model.get_topic_info()) - 1,
            'training_date': pd.Timestamp.now().isoformat(),
            'preprocessing_cache': self.preprocessing_cache
        }
        
        self.models[model_name] = model_metadata
        
        return topic_model, model_metadata
    
    def evaluate_model_performance(self, model_metadata, test_documents=None):
        topic_model = model_metadata['model']
        
        performance_metrics = {
            'model_info': {
                'n_topics': model_metadata['topics_found'],
                'training_docs': model_metadata['training_docs'],
                'outlier_ratio': self._calculate_outlier_ratio(topic_model)
            }
        }
        
        topics = topic_model.get_topics()
        
        performance_metrics['topic_quality'] = {
            'avg_topic_size': np.mean([len(topic_model.get_topic_info()[topic_model.get_topic_info()['Topic'] == t]['Count']) for t in topics if t != -1]),
            'topic_diversity': self._calculate_diversity(topics),
            'topic_coherence': self._estimate_coherence(topics, topic_model)
        }
        
        if test_documents:
            test_processed = self.preprocess_documents(test_documents)
            test_topics, test_probs = topic_model.transform(test_processed)
            
            performance_metrics['generalization'] = {
                'test_docs': len(test_processed),
                'assigned_ratio': np.mean(np.array(test_topics) != -1),
                'avg_confidence': np.mean(np.max(test_probs, axis=1))
            }
        
        return performance_metrics
    
    def _calculate_outlier_ratio(self, topic_model):
        topics = topic_model.topics_
        return np.mean(np.array(topics) == -1)
    
    def _calculate_diversity(self, topics):
        all_words = []
        for topic_id, topic in topics.items():
            if topic_id != -1:
                all_words.extend([word for word, _ in topic[:10]])
        
        return len(set(all_words)) / len(all_words) if all_words else 0
    
    def _estimate_coherence(self, topics, topic_model):
        coherence_scores = []
        
        for topic_id, topic in topics.items():
            if topic_id == -1:
                continue
            
            topic_words = [word for word, _ in topic[:5]]
            if len(topic_words) > 1:
                try:
                    embeddings = topic_model.embedding_model.encode(topic_words)
                    similarity_matrix = np.corrcoef(embeddings)
                    coherence = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                    coherence_scores.append(coherence)
                except:
                    coherence_scores.append(0.5)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def deploy_model(self, model_name, deployment_config):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_metadata = self.models[model_name]
        topic_model = model_metadata['model']
        
        deployment_package = {
            'model': topic_model,
            'config': model_metadata['config'],
            'preprocessing_function': self.preprocess_documents,
            'metadata': model_metadata,
            'deployment_config': deployment_config,
            'api_version': '1.0.0'
        }
        
        if deployment_config.get('save_path'):
            import pickle
            with open(deployment_config['save_path'], 'wb') as f:
                pickle.dump(deployment_package, f)
            print(f"Model deployed to {deployment_config['save_path']}")
        
        return deployment_package
    
    def create_inference_api(self, model_name):
        def inference_function(new_documents, return_probabilities=False):
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            topic_model = self.models[model_name]['model']
            processed_docs = self.preprocess_documents(new_documents)
            
            topics, probabilities = topic_model.transform(processed_docs)
            
            results = []
            for i, (doc, topic, probs) in enumerate(zip(processed_docs, topics, probabilities)):
                result = {
                    'document_id': i,
                    'topic': int(topic),
                    'confidence': float(np.max(probs)),
                    'top_keywords': [word for word, _ in topic_model.get_topic(topic)[:5]] if topic != -1 else []
                }
                
                if return_probabilities:
                    result['all_probabilities'] = probs.tolist()
                
                results.append(result)
            
            return results
        
        return inference_function

pipeline = ProductionTopicPipeline()
```

---

## 🎯 **Quick Start Action Plan**

### **Immediate Practice:**
1. **Run advanced topic modeling system** - Compare embedding models, reduction methods, clustering algorithms
2. **Try temporal topic analysis** - Track how topics evolve over time periods
3. **Experiment with hierarchical topics** - Build coarse-to-fine topic structures

### **This Week's Goals:**
1. **Master BERTopic modularity** - Mix and match components for optimal results
2. **Understand quality evaluation** - Use coherence, diversity, and stability metrics
3. **Practice cross-domain transfer** - Adapt topics from one domain to another

### **Advanced Projects:**
1. **Build production pipeline** - Create deployment-ready topic modeling system
2. **Implement interactive explorer** - Build dashboard for topic analysis
3. **Develop domain adaptation** - Transfer topics across different document collections

The enhanced framework transforms basic clustering into sophisticated topic modeling with quality evaluation, temporal analysis, and production deployment capabilities.

---

## 🎯 **Key Chapter 5 Insights**

### **Modular Pipeline Architecture:**
- **Embed → Reduce → Cluster** - Three-step semantic grouping process
- **UMAP over PCA** - Better preservation of local and global structure for clustering
- **HDBSCAN over K-means** - Automatic cluster detection with outlier handling
- **c-TF-IDF representation** - Cluster-level term frequency for meaningful topic keywords

### **Memory Anchors:**
- **"Embeddings capture semantics"** - Similar documents cluster together naturally
- **"Dimensionality reduction enables clustering"** - High dimensions hurt distance metrics
- **"Density-based beats centroid-based"** - HDBSCAN finds natural groupings vs forced spheres
- **"Multiple representations reveal different perspectives"** - Keywords, labels, summaries complement each other

### **Production Considerations:**
The enhanced system enables real-world deployment through:
- **Quality evaluation metrics** - Coherence, diversity, stability measurement
- **Temporal analysis** - Track topic evolution across time periods
- **Cross-domain adaptation** - Transfer topics between different document collections
- **Interactive exploration** - Dashboards and interfaces for human-in-the-loop analysis

### **Practical Applications:**
- **Research trend analysis** - Discover emerging topics in academic literature
- **Customer feedback categorization** - Automatically group support tickets and reviews
- **Content recommendation** - Find similar documents for personalization
- **Market intelligence** - Track competitor mentions and industry themes

This chapter's enhanced framework transforms unsupervised text clustering from exploratory analysis into production-ready topic discovery systems with comprehensive evaluation and deployment capabilities.
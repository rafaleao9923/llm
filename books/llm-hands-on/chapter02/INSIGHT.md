## 🎯 **Author's Core Intentions**

The authors transition from abstract tokenization concepts to concrete implementation patterns, demonstrating:

1. **Token Flow Visualization**: Raw text → Token IDs → Model processing → Output decoding
2. **Tokenizer Evolution**: Historical progression from BERT's WordPiece to GPT-4's advanced BPE
3. **Contextualized vs Static Embeddings**: Moving beyond Word2Vec to transformer-based representations
4. **Real-world Applications**: Music recommendation system as practical embedding use case

The sample code shows four critical patterns:
- **Direct tokenizer exploration** with ID-to-text mapping
- **Multi-tokenizer comparison** revealing design tradeoffs
- **Embedding extraction** from transformer models
- **Word2Vec application** to non-text domains (music playlists)

```python
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from collections import defaultdict
import json
import time

class TokenizationExplorer:
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
        self.load_tokenizers()
    
    def load_tokenizers(self):
        tokenizer_configs = {
            'bert_uncased': 'bert-base-uncased',
            'bert_cased': 'bert-base-cased', 
            'gpt2': 'gpt2',
            'gpt4': 'Xenova/gpt-4',
            'phi3': 'microsoft/Phi-3-mini-4k-instruct',
            't5': 'google/flan-t5-small'
        }
        
        for name, model_name in tokenizer_configs.items():
            try:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(model_name)
                print(f"✅ Loaded {name}")
            except:
                print(f"❌ Failed to load {name}")
    
    def analyze_tokenization_patterns(self, test_cases):
        results = {}
        
        for text_name, text in test_cases.items():
            results[text_name] = {}
            
            for tokenizer_name, tokenizer in self.tokenizers.items():
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.encode(text)
                
                results[text_name][tokenizer_name] = {
                    'tokens': tokens,
                    'token_count': len(tokens),
                    'token_ids': token_ids,
                    'efficiency': len(text) / len(tokens) if tokens else 0,
                    'vocab_size': tokenizer.vocab_size,
                    'special_tokens': len([t for t in tokens if t.startswith('[') or t.startswith('<')])
                }
        
        return results
    
    def visualize_tokenization_efficiency(self, results):
        data = []
        for text_name, tokenizer_results in results.items():
            for tokenizer_name, metrics in tokenizer_results.items():
                data.append({
                    'text': text_name,
                    'tokenizer': tokenizer_name,
                    'token_count': metrics['token_count'],
                    'efficiency': metrics['efficiency']
                })
        
        df = pd.DataFrame(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        pivot_count = df.pivot(index='text', columns='tokenizer', values='token_count')
        sns.heatmap(pivot_count, annot=True, fmt='d', ax=ax1, cmap='YlOrRd')
        ax1.set_title('Token Count by Tokenizer')
        
        pivot_eff = df.pivot(index='text', columns='tokenizer', values='efficiency')
        sns.heatmap(pivot_eff, annot=True, fmt='.2f', ax=ax2, cmap='YlGnBu')
        ax2.set_title('Characters per Token (Efficiency)')
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def decode_token_by_token(self, text, tokenizer_name):
        if tokenizer_name not in self.tokenizers:
            return
            
        tokenizer = self.tokenizers[tokenizer_name]
        token_ids = tokenizer.encode(text)
        
        print(f"🔍 Token-by-token analysis ({tokenizer_name}):")
        print(f"Input: '{text}'")
        print("-" * 50)
        
        for i, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id])
            print(f"Token {i:2d}: ID={token_id:5d} | '{token_text}'")
        
        print(f"\nTotal tokens: {len(token_ids)}")
        print(f"Reconstructed: '{tokenizer.decode(token_ids)}'")

class EmbeddingExplorer:
    def __init__(self):
        self.models = {}
        self.sentence_model = None
        self.load_models()
    
    def load_models(self):
        try:
            self.models['bert'] = AutoModel.from_pretrained('bert-base-uncased')
            self.models['bert_tokenizer'] = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("✅ Loaded BERT")
        except:
            print("❌ Failed to load BERT")
        
        try:
            self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            print("✅ Loaded sentence transformer")
        except:
            print("❌ Failed to load sentence transformer")
    
    def extract_contextualized_embeddings(self, sentences, layer_index=-1):
        if 'bert' not in self.models:
            return None
            
        model = self.models['bert']
        tokenizer = self.models['bert_tokenizer']
        
        embeddings_data = []
        
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                last_hidden_state = outputs.hidden_states[layer_index]
                cls_embedding = last_hidden_state[0, 0, :].numpy()
                mean_embedding = last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
                
                embeddings_data.append({
                    'sentence': sentence,
                    'cls_embedding': cls_embedding,
                    'mean_embedding': mean_embedding,
                    'tokens': tokenizer.tokenize(sentence)
                })
        
        return embeddings_data
    
    def compare_contextualized_vs_static(self, word_pairs):
        results = {}
        
        for word1, word2 in word_pairs:
            contexts1 = [
                f"The {word1} was very important.",
                f"She decided to {word1} the document.",
                f"The {word1} of the river was steep."
            ]
            
            contexts2 = [
                f"The {word2} was very important.",
                f"She decided to {word2} the document.", 
                f"The {word2} of the river was steep."
            ]
            
            emb1 = self.extract_contextualized_embeddings(contexts1)
            emb2 = self.extract_contextualized_embeddings(contexts2)
            
            if emb1 and emb2:
                similarities = []
                for e1 in emb1:
                    for e2 in emb2:
                        sim = cosine_similarity(
                            e1['cls_embedding'].reshape(1, -1),
                            e2['cls_embedding'].reshape(1, -1)
                        )[0, 0]
                        similarities.append(sim)
                
                results[f"{word1}_{word2}"] = {
                    'avg_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'contexts': (contexts1, contexts2)
                }
        
        return results
    
    def sentence_similarity_matrix(self, sentences):
        if not self.sentence_model:
            return None
            
        embeddings = self.sentence_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix, 
            annot=True, 
            fmt='.3f',
            xticklabels=[s[:30] + '...' if len(s) > 30 else s for s in sentences],
            yticklabels=[s[:30] + '...' if len(s) > 30 else s for s in sentences],
            cmap='coolwarm',
            center=0
        )
        plt.title('Sentence Similarity Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return similarity_matrix, embeddings

class Word2VecExplorer:
    def __init__(self):
        self.models = {}
        self.training_data = {}
    
    def create_synthetic_playlist_data(self):
        genres = {
            'rock': ['rolling_stones', 'led_zeppelin', 'queen', 'beatles', 'pink_floyd'],
            'rap': ['tupac', 'biggie', 'nas', 'jay_z', 'eminem'],
            'pop': ['madonna', 'michael_jackson', 'prince', 'whitney_houston', 'britney_spears'],
            'jazz': ['miles_davis', 'john_coltrane', 'charlie_parker', 'duke_ellington', 'ella_fitzgerald']
        }
        
        playlists = []
        
        for genre, artists in genres.items():
            for _ in range(50):
                playlist_size = np.random.randint(3, 8)
                if np.random.random() < 0.8:
                    playlist = np.random.choice(artists, playlist_size, replace=True).tolist()
                else:
                    mixed_artists = []
                    for g in genres.values():
                        mixed_artists.extend(g)
                    playlist = np.random.choice(mixed_artists, playlist_size, replace=True).tolist()
                
                playlists.append(playlist)
        
        return playlists
    
    def train_embeddings(self, sequences, embedding_type='music'):
        model = Word2Vec(
            sequences,
            vector_size=50,
            window=10,
            min_count=1,
            workers=4,
            epochs=10
        )
        
        self.models[embedding_type] = model
        self.training_data[embedding_type] = sequences
        
        return model
    
    def analyze_embeddings(self, model, item, topn=5):
        try:
            similar_items = model.wv.most_similar(item, topn=topn)
            return similar_items
        except KeyError:
            return f"'{item}' not in vocabulary"
    
    def visualize_embeddings(self, model, items_to_plot=None):
        if not items_to_plot:
            items_to_plot = list(model.wv.key_to_index.keys())[:20]
        
        embeddings = [model.wv[item] for item in items_to_plot if item in model.wv]
        valid_items = [item for item in items_to_plot if item in model.wv]
        
        if len(embeddings) < 2:
            print("Not enough items to visualize")
            return
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        for i, item in enumerate(valid_items):
            plt.annotate(item, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('2D Visualization of Embeddings (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return embeddings_2d

def main_exploration():
    print("🎵 Chapter 2: Advanced Token and Embedding Analysis")
    print("=" * 60)
    
    test_cases = {
        'simple': "Hello world!",
        'mixed_case': "English and CAPITALIZATION",
        'special_chars': "🎵 鸟 emoji test",
        'code': "def function(): return True",
        'numbers': "12.0 * 50 = 600",
        'punctuation': "Dr. Smith's cat, isn't it?"
    }
    
    print("\n🔤 PART 1: TOKENIZATION ANALYSIS")
    print("-" * 40)
    
    tokenizer_explorer = TokenizationExplorer()
    results = tokenizer_explorer.analyze_tokenization_patterns(test_cases)
    
    for text_name, text in test_cases.items():
        print(f"\n📝 Analyzing: '{text}'")
        tokenizer_explorer.decode_token_by_token(text, 'phi3')
        break
    
    print("\n📊 Efficiency comparison across tokenizers:")
    df = tokenizer_explorer.visualize_tokenization_efficiency(results)
    
    print("\n🧠 PART 2: CONTEXTUALIZED EMBEDDINGS")
    print("-" * 40)
    
    embedding_explorer = EmbeddingExplorer()
    
    test_sentences = [
        "The bank is by the river.",
        "I need to bank this check.",
        "The bank offers great loans.",
        "Fishing by the river bank.",
        "She works at the bank downtown."
    ]
    
    print("🔍 Extracting contextualized embeddings...")
    embeddings_data = embedding_explorer.extract_contextualized_embeddings(test_sentences)
    
    if embeddings_data:
        print("✅ Successfully extracted embeddings")
        for i, data in enumerate(embeddings_data[:2]):
            print(f"Sentence {i+1}: {data['sentence']}")
            print(f"Embedding shape: {data['cls_embedding'].shape}")
            print(f"First 5 dimensions: {data['cls_embedding'][:5]}")
    
    print("\n📄 PART 3: SENTENCE SIMILARITY")
    print("-" * 40)
    
    similarity_sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Dogs are great pets.",
        "Canines make wonderful companions.",
        "I love programming in Python.",
        "Coding with Python is enjoyable."
    ]
    
    similarity_matrix, embeddings = embedding_explorer.sentence_similarity_matrix(similarity_sentences)
    
    print("\n🎵 PART 4: WORD2VEC MUSIC RECOMMENDATIONS")
    print("-" * 40)
    
    w2v_explorer = Word2VecExplorer()
    
    print("🎶 Creating synthetic music playlist data...")
    playlists = w2v_explorer.create_synthetic_playlist_data()
    print(f"Generated {len(playlists)} playlists")
    print(f"Sample playlist: {playlists[0]}")
    
    print("\n🏋️ Training Word2Vec model...")
    music_model = w2v_explorer.train_embeddings(playlists, 'music')
    
    test_artists = ['rolling_stones', 'tupac', 'miles_davis']
    
    for artist in test_artists:
        print(f"\n🎤 Artists similar to {artist}:")
        similar = w2v_explorer.analyze_embeddings(music_model, artist)
        if isinstance(similar, list):
            for similar_artist, score in similar:
                print(f"  {similar_artist}: {score:.3f}")
    
    print("\n📈 Visualizing music embeddings...")
    w2v_explorer.visualize_embeddings(music_model)
    
    print("\n🔬 PART 5: EMBEDDING ARITHMETIC")
    print("-" * 40)
    
    try:
        king_vec = music_model.wv['rolling_stones']
        rock_vec = music_model.wv['led_zeppelin'] 
        pop_vec = music_model.wv['michael_jackson']
        
        result_vec = king_vec - rock_vec + pop_vec
        similar_to_result = music_model.wv.similar_by_vector(result_vec, topn=3)
        
        print("🧮 Embedding arithmetic:")
        print("rolling_stones - led_zeppelin + michael_jackson =")
        for artist, score in similar_to_result:
            print(f"  {artist}: {score:.3f}")
            
    except KeyError as e:
        print(f"⚠️ Arithmetic failed: {e}")

if __name__ == "__main__":
    main_exploration()
```

---

# Chapter 2 Hands-On Practice Exercises

## 🔤 **Tokenization Mastery Drills**

### **Exercise 1: Tokenizer Detective**
```python
def tokenizer_investigation(text, tokenizer_names):
    results = {}
    for name in tokenizer_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokens = tokenizer.tokenize(text)
        
        analysis = {
            'token_count': len(tokens),
            'unknown_tokens': sum(1 for t in tokens if '[UNK]' in t or '<unk>' in t),
            'special_tokens': sum(1 for t in tokens if t.startswith(('[', '<'))),
            'subword_tokens': sum(1 for t in tokens if '##' in t or 'Ġ' in t),
            'efficiency_score': len(text) / len(tokens)
        }
        results[name] = analysis
    
    return results

challenge_texts = [
    "SupercalifragilisticexpialidociousAntidisestablishmentarianism",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "🚀✨ AI/ML 🤖 = Artificial Intelligence & Machine Learning! 💯",
    "北京大学 東京大学 서울대학교 Université de Paris",
    "COVID-19 affected e-commerce (25% ↑) vs. brick-and-mortar (-15% ↓)"
]

tokenizers = ["bert-base-uncased", "gpt2", "microsoft/Phi-3-mini-4k-instruct"]
```

### **Exercise 2: Token Efficiency Optimizer**
```python
def find_optimal_tokenizer(corpus, tokenizer_candidates):
    scores = {}
    
    for tokenizer_name in tokenizer_candidates:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        total_tokens = 0
        total_chars = 0
        
        for text in corpus:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)
            total_chars += len(text)
        
        efficiency = total_chars / total_tokens
        vocab_coverage = calculate_vocab_coverage(corpus, tokenizer)
        
        scores[tokenizer_name] = {
            'efficiency': efficiency,
            'vocab_coverage': vocab_coverage,
            'composite_score': efficiency * vocab_coverage
        }
    
    return sorted(scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)

def calculate_vocab_coverage(corpus, tokenizer):
    unique_tokens = set()
    for text in corpus:
        unique_tokens.update(tokenizer.tokenize(text))
    return len(unique_tokens) / tokenizer.vocab_size
```

### **Exercise 3: Custom Tokenizer Trainer**
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_custom_bpe_tokenizer(corpus, vocab_size=1000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>"])
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer

domain_corpus = [
    "Machine learning models require large datasets for training.",
    "Neural networks use backpropagation for parameter optimization.",
    "Transformer architectures revolutionized natural language processing.",
    "Attention mechanisms enable models to focus on relevant information."
]

custom_tokenizer = train_custom_bpe_tokenizer(domain_corpus)
```

## 🧠 **Embedding Analysis Challenges**

### **Challenge 1: Context Sensitivity Detector**
```python
def measure_context_sensitivity(word, contexts, model_name="bert-base-uncased"):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    embeddings = []
    
    for context in contexts:
        inputs = tokenizer(context, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        
        word_tokens = tokenizer.tokenize(word)
        context_tokens = tokenizer.tokenize(context)
        
        word_positions = find_word_positions(word_tokens, context_tokens)
        
        if word_positions:
            word_embedding = outputs.last_hidden_state[0, word_positions[0]:word_positions[-1]+1].mean(dim=0)
            embeddings.append(word_embedding.numpy())
    
    if len(embeddings) > 1:
        similarity_matrix = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        return 1 - avg_similarity
    
    return 0

polysemous_words = ["bank", "bark", "bat", "fair", "light"]
contexts_per_word = {
    "bank": [
        "I deposited money at the bank.",
        "We sat by the river bank.",
        "The plane had to bank to the left."
    ],
    "bark": [
        "The dog's bark was loud.",
        "Tree bark protects the trunk.",
        "The captain would bark orders."
    ]
}
```

### **Challenge 2: Semantic Cluster Analysis**
```python
def analyze_semantic_clusters(word_list, model):
    embeddings = [model.encode(word) for word in word_list]
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    best_k = 2
    best_score = -1
    
    for k in range(2, min(len(word_list)//2, 10)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = final_kmeans.fit_predict(embeddings)
    
    cluster_groups = defaultdict(list)
    for word, cluster in zip(word_list, clusters):
        cluster_groups[cluster].append(word)
    
    return dict(cluster_groups), best_score

vocabulary = [
    "happy", "joyful", "sad", "depressed", "angry", "furious",
    "car", "bicycle", "train", "airplane", "boat",
    "apple", "banana", "carrot", "broccoli", "pizza", "burger"
]
```

### **Challenge 3: Embedding Arithmetic Explorer**
```python
def embedding_arithmetic_explorer(model, operation_sets):
    results = {}
    
    for operation_name, (word_a, word_b, word_c) in operation_sets.items():
        try:
            vec_a = model.wv[word_a]
            vec_b = model.wv[word_b] 
            vec_c = model.wv[word_c]
            
            result_vector = vec_a - vec_b + vec_c
            similar_words = model.wv.similar_by_vector(result_vector, topn=5)
            
            results[operation_name] = {
                'operation': f"{word_a} - {word_b} + {word_c}",
                'results': similar_words,
                'expected_category': predict_category(word_a, word_b, word_c)
            }
        except KeyError as e:
            results[operation_name] = f"Missing word: {e}"
    
    return results

def predict_category(word_a, word_b, word_c):
    categories = {
        'royalty': ['king', 'queen', 'prince', 'princess'],
        'countries': ['france', 'germany', 'italy', 'spain'],
        'professions': ['doctor', 'teacher', 'engineer', 'lawyer']
    }
    
    for category, words in categories.items():
        if any(word in words for word in [word_a, word_b, word_c]):
            return category
    return 'unknown'

arithmetic_operations = {
    'gender_analogy': ('king', 'man', 'woman'),
    'capital_analogy': ('paris', 'france', 'germany'),
    'profession_analogy': ('doctor', 'hospital', 'school')
}
```

## 🎵 **Real-World Application Projects**

### **Project 1: Document Similarity Engine**
```python
class DocumentSimilarityEngine:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = {}
        self.documents = {}
    
    def add_document(self, doc_id, text, metadata=None):
        embedding = self.model.encode(text)
        self.document_embeddings[doc_id] = embedding
        self.documents[doc_id] = {'text': text, 'metadata': metadata or {}}
    
    def find_similar_documents(self, query, top_k=5, threshold=0.5):
        query_embedding = self.model.encode(query)
        
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0, 0]
            if similarity >= threshold:
                similarities[doc_id] = similarity
        
        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, similarity in sorted_docs:
            results.append({
                'doc_id': doc_id,
                'similarity': similarity,
                'text': self.documents[doc_id]['text'][:200] + '...',
                'metadata': self.documents[doc_id]['metadata']
            })
        
        return results
    
    def cluster_documents(self, n_clusters=None):
        embeddings = list(self.document_embeddings.values())
        doc_ids = list(self.document_embeddings.keys())
        
        if n_clusters is None:
            n_clusters = min(len(embeddings) // 3, 10)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        clustered_docs = defaultdict(list)
        for doc_id, cluster in zip(doc_ids, clusters):
            clustered_docs[cluster].append(doc_id)
        
        return dict(clustered_docs)
```

### **Project 2: Multi-Language Tokenization Analyzer**
```python
class MultiLanguageTokenizationAnalyzer:
    def __init__(self):
        self.tokenizers = {
            'english': AutoTokenizer.from_pretrained('bert-base-uncased'),
            'multilingual': AutoTokenizer.from_pretrained('bert-base-multilingual-cased'),
            'code': AutoTokenizer.from_pretrained('microsoft/codebert-base'),
            'gpt': AutoTokenizer.from_pretrained('gpt2')
        }
    
    def analyze_language_efficiency(self, texts_by_language):
        results = {}
        
        for language, texts in texts_by_language.items():
            lang_results = {}
            
            for tokenizer_name, tokenizer in self.tokenizers.items():
                total_tokens = 0
                total_chars = 0
                unknown_rate = 0
                
                for text in texts:
                    tokens = tokenizer.tokenize(text)
                    total_tokens += len(tokens)
                    total_chars += len(text)
                    
                    unknown_tokens = sum(1 for t in tokens if '[UNK]' in t or '<unk>' in t)
                    unknown_rate += unknown_tokens / len(tokens) if tokens else 0
                
                avg_efficiency = total_chars / total_tokens if total_tokens else 0
                avg_unknown_rate = unknown_rate / len(texts) if texts else 0
                
                lang_results[tokenizer_name] = {
                    'efficiency': avg_efficiency,
                    'unknown_rate': avg_unknown_rate,
                    'score': avg_efficiency * (1 - avg_unknown_rate)
                }
            
            results[language] = lang_results
        
        return results
    
    def recommend_tokenizer(self, text_type, language='english'):
        recommendations = {
            ('english', 'general'): 'english',
            ('multilingual', 'general'): 'multilingual', 
            ('english', 'code'): 'code',
            ('any', 'generation'): 'gpt'
        }
        
        return recommendations.get((language, text_type), 'multilingual')

test_texts = {
    'english': ["Hello world", "The quick brown fox jumps over the lazy dog"],
    'spanish': ["Hola mundo", "El zorro marrón salta sobre el perro perezoso"],
    'chinese': ["你好世界", "人工智能正在改变世界"],
    'japanese': ["こんにちは世界", "人工知能が世界を変えています"],
    'code': ["def hello(): return 'world'", "class AI: def __init__(self): pass"]
}
```

### **Project 3: Advanced Word2Vec Applications**
```python
class AdvancedWord2VecApplications:
    def __init__(self):
        self.models = {}
        self.vocabularies = {}
    
    def train_domain_specific_embeddings(self, domain_name, sequences, **kwargs):
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10,
            'sg': 1  # Skip-gram
        }
        default_params.update(kwargs)
        
        model = Word2Vec(sequences, **default_params)
        self.models[domain_name] = model
        self.vocabularies[domain_name] = set(model.wv.key_to_index.keys())
        
        return model
    
    def create_user_behavior_embeddings(self, user_sessions):
        """
        Create embeddings from user behavior sequences
        user_sessions: [['page1', 'page2', 'page3'], ['page1', 'page4', 'page5']]
        """
        return self.train_domain_specific_embeddings('user_behavior', user_sessions)
    
    def create_product_embeddings(self, purchase_sequences):
        """
        Create product embeddings from purchase history
        purchase_sequences: [['product1', 'product2'], ['product1', 'product3']]
        """
        return self.train_domain_specific_embeddings('products', purchase_sequences)
    
    def analyze_embedding_quality(self, model, test_analogies):
        """
        Test embedding quality using analogy tasks
        test_analogies: [('a', 'b', 'c', 'expected_d')]
        """
        correct = 0
        total = len(test_analogies)
        
        for a, b, c, expected_d in test_analogies:
            try:
                result = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
                predicted_d = result[0][0]
                if predicted_d == expected_d:
                    correct += 1
            except KeyError:
                total -= 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'failed_analogies': total - len(test_analogies)
        }
    
    def cross_domain_similarity(self, domain1, domain2, items1, items2):
        """
        Compare embeddings across different domains
        """
        if domain1 not in self.models or domain2 not in self.models:
            return None
        
        model1 = self.models[domain1]
        model2 = self.models[domain2]
        
        similarities = []
        
        for item1 in items1:
            if item1 in model1.wv:
                vec1 = model1.wv[item1]
                
                for item2 in items2:
                    if item2 in model2.wv:
                        vec2 = model2.wv[item2]
                        
                        # Align embeddings if different dimensions
                        if vec1.shape != vec2.shape:
                            min_dim = min(len(vec1), len(vec2))
                            vec1_aligned = vec1[:min_dim]
                            vec2_aligned = vec2[:min_dim]
                        else:
                            vec1_aligned = vec1
                            vec2_aligned = vec2
                        
                        similarity = cosine_similarity([vec1_aligned], [vec2_aligned])[0, 0]
                        similarities.append({
                            'item1': item1,
                            'item2': item2,
                            'similarity': similarity
                        })
        
        return similarities

# Sample usage examples
sample_user_sessions = [
    ['homepage', 'product_page', 'cart', 'checkout'],
    ['homepage', 'search', 'product_page', 'reviews'],
    ['login', 'dashboard', 'settings', 'logout'],
    ['homepage', 'blog', 'contact', 'about']
]

sample_purchase_sequences = [
    ['laptop', 'mouse', 'keyboard', 'monitor'],
    ['phone', 'case', 'charger', 'earphones'],
    ['book', 'bookmark', 'reading_light'],
    ['laptop', 'case', 'charger', 'mouse']
]
```

## 🧪 **Advanced Research Experiments**

### **Experiment 1: Tokenization Impact on Model Performance**
```python
def tokenization_performance_study(texts, model_checkpoints, tasks):
    """
    Study how different tokenizers affect model performance
    """
    results = {}
    
    for model_name in model_checkpoints:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        task_results = {}
        
        for task_name, task_data in tasks.items():
            if task_name == 'text_classification':
                performance = evaluate_classification_performance(
                    model, tokenizer, task_data['texts'], task_data['labels']
                )
            elif task_name == 'similarity':
                performance = evaluate_similarity_performance(
                    model, tokenizer, task_data['pairs'], task_data['scores']
                )
            elif task_name == 'generation':
                performance = evaluate_generation_performance(
                    model, tokenizer, task_data['prompts'], task_data['targets']
                )
            
            task_results[task_name] = performance
        
        results[model_name] = task_results
    
    return results

def evaluate_classification_performance(model, tokenizer, texts, labels):
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token for classification
            embedding = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings.append(embedding)
    
    # Simple logistic regression for classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    clf = LogisticRegression(random_state=42)
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'embedding_dim': len(embeddings[0]) if embeddings else 0
    }
```

### **Experiment 2: Embedding Stability Analysis**
```python
def embedding_stability_analysis(models, test_words, perturbations):
    """
    Analyze how stable embeddings are to small text perturbations
    """
    stability_results = {}
    
    for model_name, model in models.items():
        word_stabilities = {}
        
        for word in test_words:
            if word not in model.wv:
                continue
                
            original_embedding = model.wv[word]
            perturbed_similarities = []
            
            for perturbation_func in perturbations:
                # Create perturbed contexts
                contexts = [
                    f"The {word} is important.",
                    f"I like {word} very much.",
                    f"This {word} is interesting."
                ]
                
                perturbed_contexts = [perturbation_func(ctx) for ctx in contexts]
                
                # Get embeddings for perturbed contexts (simplified)
                for perturbed_ctx in perturbed_contexts:
                    if word in perturbed_ctx:
                        # Simulate embedding extraction from context
                        # In reality, would need contextual model
                        similarity = np.random.normal(0.8, 0.1)  # Placeholder
                        perturbed_similarities.append(max(0, min(1, similarity)))
            
            word_stabilities[word] = {
                'mean_stability': np.mean(perturbed_similarities),
                'std_stability': np.std(perturbed_similarities),
                'min_stability': np.min(perturbed_similarities)
            }
        
        stability_results[model_name] = word_stabilities
    
    return stability_results

def add_typos(text):
    """Add random typos to text"""
    import random
    words = text.split()
    if words:
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        if len(word) > 2:
            char_idx = random.randint(1, len(word) - 2)
            typo_word = word[:char_idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[char_idx+1:]
            words[word_idx] = typo_word
    return ' '.join(words)

def add_synonyms(text):
    """Replace words with synonyms"""
    synonym_map = {
        'important': 'crucial',
        'like': 'enjoy',
        'interesting': 'fascinating',
        'good': 'excellent',
        'bad': 'terrible'
    }
    
    for original, synonym in synonym_map.items():
        text = text.replace(original, synonym)
    return text

perturbation_functions = [add_typos, add_synonyms]
```

### **Experiment 3: Multilingual Embedding Alignment**
```python
def multilingual_embedding_alignment(models_by_language, translation_pairs):
    """
    Analyze alignment between multilingual embeddings
    """
    alignment_results = {}
    
    languages = list(models_by_language.keys())
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages[i+1:], i+1):
            model1 = models_by_language[lang1]
            model2 = models_by_language[lang2]
            
            pair_key = f"{lang1}_{lang2}"
            similarities = []
            
            for word1, word2 in translation_pairs.get(pair_key, []):
                if word1 in model1.wv and word2 in model2.wv:
                    vec1 = model1.wv[word1]
                    vec2 = model2.wv[word2]
                    
                    # Normalize vectors
                    vec1_norm = vec1 / np.linalg.norm(vec1)
                    vec2_norm = vec2 / np.linalg.norm(vec2)
                    
                    similarity = np.dot(vec1_norm, vec2_norm)
                    similarities.append(similarity)
            
            if similarities:
                alignment_results[pair_key] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'num_pairs': len(similarities),
                    'alignment_quality': 'high' if np.mean(similarities) > 0.6 else 'low'
                }
    
    return alignment_results

# Sample translation pairs
translation_pairs = {
    'english_spanish': [
        ('cat', 'gato'), ('dog', 'perro'), ('house', 'casa'),
        ('water', 'agua'), ('fire', 'fuego'), ('love', 'amor')
    ],
    'english_french': [
        ('cat', 'chat'), ('dog', 'chien'), ('house', 'maison'),
        ('water', 'eau'), ('fire', 'feu'), ('love', 'amour')
    ]
}
```

## 📈 **Performance Benchmarking Suite**

### **Benchmark 1: Tokenization Speed Test**
```python
import time
from collections import defaultdict

def tokenization_speed_benchmark(texts, tokenizers, iterations=100):
    """
    Benchmark tokenization speed across different tokenizers
    """
    results = defaultdict(list)
    
    for _ in range(iterations):
        for tokenizer_name, tokenizer in tokenizers.items():
            start_time = time.time()
            
            for text in texts:
                tokens = tokenizer.tokenize(text)
            
            end_time = time.time()
            elapsed = end_time - start_time
            results[tokenizer_name].append(elapsed)
    
    benchmark_results = {}
    for tokenizer_name, times in results.items():
        benchmark_results[tokenizer_name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'texts_per_second': len(texts) / np.mean(times)
        }
    
    return benchmark_results

# Speed test with various text lengths
speed_test_texts = [
    "Short text.",
    "Medium length text with several words and some punctuation marks.",
    "Very long text " * 100 + "that tests tokenizer performance on extended content.",
    "Mixed content: code_variable = 'string' + 123 + [list, items] 🚀",
    "Special characters: àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
]
```

### **Benchmark 2: Embedding Quality Metrics**
```python
def embedding_quality_benchmark(model, test_suite):
    """
    Comprehensive embedding quality evaluation
    """
    results = {}
    
    # Analogy task
    if 'analogies' in test_suite:
        analogy_accuracy = evaluate_analogies(model, test_suite['analogies'])
        results['analogy_accuracy'] = analogy_accuracy
    
    # Similarity task
    if 'similarity_pairs' in test_suite:
        similarity_correlation = evaluate_similarity_correlation(
            model, test_suite['similarity_pairs']
        )
        results['similarity_correlation'] = similarity_correlation
    
    # Clustering task
    if 'word_categories' in test_suite:
        clustering_score = evaluate_clustering_quality(
            model, test_suite['word_categories']
        )
        results['clustering_score'] = clustering_score
    
    # Odd-one-out task
    if 'odd_one_out' in test_suite:
        odd_one_out_accuracy = evaluate_odd_one_out(
            model, test_suite['odd_one_out']
        )
        results['odd_one_out_accuracy'] = odd_one_out_accuracy
    
    return results

def evaluate_analogies(model, analogies):
    """
    Evaluate model on analogy tasks: a:b :: c:?
    """
    correct = 0
    total = 0
    
    for a, b, c, expected_d in analogies:
        try:
            candidates = model.wv.most_similar(positive=[b, c], negative=[a], topn=10)
            predicted_words = [word for word, score in candidates]
            
            if expected_d in predicted_words:
                correct += 1
            total += 1
        except KeyError:
            pass
    
    return correct / total if total > 0 else 0

def evaluate_similarity_correlation(model, similarity_pairs):
    """
    Evaluate correlation between model similarities and human judgments
    """
    model_similarities = []
    human_similarities = []
    
    for word1, word2, human_score in similarity_pairs:
        try:
            model_sim = model.wv.similarity(word1, word2)
            model_similarities.append(model_sim)
            human_similarities.append(human_score)
        except KeyError:
            pass
    
    if len(model_similarities) > 1:
        correlation = np.corrcoef(model_similarities, human_similarities)[0, 1]
        return correlation
    return 0

# Comprehensive test suite
embedding_test_suite = {
    'analogies': [
        ('king', 'man', 'woman', 'queen'),
        ('paris', 'france', 'italy', 'rome'),
        ('good', 'better', 'bad', 'worse'),
        ('walk', 'walking', 'swim', 'swimming')
    ],
    'similarity_pairs': [
        ('cat', 'dog', 0.8),
        ('car', 'automobile', 0.9),
        ('happy', 'sad', 0.1),
        ('big', 'large', 0.9),
        ('computer', 'banana', 0.0)
    ],
    'word_categories': {
        'animals': ['cat', 'dog', 'bird', 'fish'],
        'colors': ['red', 'blue', 'green', 'yellow'],
        'numbers': ['one', 'two', 'three', 'four']
    },
    'odd_one_out': [
        (['cat', 'dog', 'bird', 'car'], 'car'),
        (['red', 'blue', 'green', 'happy'], 'happy'),
        (['apple', 'banana', 'orange', 'table'], 'table')
    ]
}
```

## 🎯 **Quick Daily Practice Routines**

### **5-Minute Lightning Rounds**
```python
def daily_tokenizer_quiz():
    """Quick tokenizer knowledge check"""
    questions = [
        ("Which tokenizer preserves case?", ["bert-base-cased", "bert-base-uncased"], 0),
        ("What does '##' prefix indicate in BERT?", ["start of word", "subword continuation"], 1),
        ("Which uses BPE algorithm?", ["BERT", "GPT-2"], 1),
        ("What's GPT-4's approximate vocab size?", ["30k", "100k"], 1)
    ]
    
    score = 0
    for question, choices, correct_idx in questions:
        print(f"\n{question}")
        for i, choice in enumerate(choices):
            print(f"{i}: {choice}")
        
        # In practice, get user input
        user_choice = correct_idx  # Simulated correct answer
        if user_choice == correct_idx:
            score += 1
            print("✅ Correct!")
        else:
            print(f"❌ Wrong! Correct answer: {choices[correct_idx]}")
    
    print(f"\nFinal Score: {score}/{len(questions)}")
    return score

def daily_embedding_challenge():
    """Quick embedding similarity challenge"""
    word_pairs = [
        ('king', 'queen', 'high'),
        ('happy', 'joyful', 'high'),
        ('cat', 'mathematics', 'low'),
        ('run', 'sprint', 'medium'),
        ('cold', 'hot', 'low')
    ]
    
    print("🧠 Predict similarity levels (high/medium/low):")
    for word1, word2, expected in word_pairs:
        print(f"'{word1}' vs '{word2}': Expected = {expected}")
```

### **10-Minute Deep Dives**
```python
def mini_tokenization_experiment():
    """10-minute hands-on tokenization experiment"""
    test_text = input("Enter text to analyze: ") or "The AI revolution is transforming everything!"
    
    tokenizers = {
        'BERT': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'GPT-2': AutoTokenizer.from_pretrained('gpt2')
    }
    
    print(f"\n🔍 Analyzing: '{test_text}'")
    print("-" * 50)
    
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(test_text)
        print(f"\n{name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {len(tokens)}")
        print(f"  Efficiency: {len(test_text)/len(tokens):.2f} chars/token")

def mini_embedding_exploration():
    """10-minute embedding similarity exploration"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sentences = [
        "I love machine learning",
        "Artificial intelligence is fascinating", 
        "The weather is nice today",
        "Deep learning models are powerful"
    ]
    
    embeddings = model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)
    
    print("📊 Sentence Similarity Matrix:")
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i < j:
                sim = similarity_matrix[i, j]
                print(f"'{sent1[:20]}...' ↔ '{sent2[:20]}...': {sim:.3f}")
```

---

*Practice these exercises daily to build intuitive understanding of tokenization patterns and embedding behaviors. Start with the lightning rounds, then progress to the deeper experiments as your confidence grows!*

---

## 🎯 **Quick Start Recommendations**

### **Immediate Actions:**
1. **Run the enhanced explorer** - Shows tokenization differences across 6+ models with visual comparisons
2. **Try the tokenizer detective exercise** - Builds pattern recognition for different architectures  
3. **Experiment with embedding arithmetic** - king - man + woman = queen intuition building

### **Key Chapter 2 Insights:**
- **Tokenization determines everything** - Efficiency, cost, multilingual support, code handling
- **Context changes embeddings** - Same word = different vectors based on surrounding text
- **Subword is the sweet spot** - Balance between word-level vocabulary explosion and character-level sequence length
- **Domain matters** - Specialized tokenizers (code, science, music) outperform general ones

### **Memory Anchors:**
- **"Tokens are model currency"** - Everything costs tokens in API calls
- **"Static vs Dynamic"** - Word2Vec gives same embedding everywhere, BERT changes by context
- **"Subword efficiency"** - Handles unknown words by breaking into known parts
- **"BPE builds vocabulary"** - Byte Pair Encoding merges frequent character pairs

### **Immediate Experiments:**
```python
# Quick tokenizer comparison
show_tokens("🤖 AI/ML = amazing!", "bert-base-uncased")
show_tokens("🤖 AI/ML = amazing!", "gpt2") 
show_tokens("🤖 AI/ML = amazing!", "microsoft/Phi-3-mini-4k-instruct")

# Context sensitivity test
analyze_contextualized("bank", [
    "I deposited money at the bank.",
    "We sat by the river bank.",
    "The plane had to bank left."
])

# Embedding arithmetic
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
```

The enhanced code transforms the basic tokenization examples into a comprehensive analysis toolkit that reveals how tokenization decisions impact model behavior, efficiency, and cross-language performance. Each section connects directly to real-world applications like document search, recommendation systems, and multilingual processing.
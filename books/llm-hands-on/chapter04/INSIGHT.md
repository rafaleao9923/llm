## 🎯 **Author's Core Intentions**

The authors demonstrate the progression from traditional ML to modern LLM-based text classification, showcasing five distinct approaches:

1. **Task-specific models** - Direct classification with pre-trained models
2. **Embedding + classifier** - Two-step approach with feature extraction
3. **Zero-shot classification** - No training data required using semantic similarity
4. **Generative models** - Text-to-text transformation for classification
5. **ChatGPT API** - Prompt-based classification with frontier models

The sample code reveals practical tradeoffs between performance, cost, and implementation complexity, showing F1 scores ranging from 0.78 (zero-shot) to 0.91 (ChatGPT).

```python
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional

class TextClassificationBenchmark:
    def __init__(self):
        self.dataset = None
        self.models = {}
        self.results = {}
        self.embeddings_cache = {}
        
    def load_data(self, dataset_name="rotten_tomatoes"):
        print(f"📊 Loading {dataset_name} dataset...")
        self.dataset = load_dataset(dataset_name)
        
        print(f"✅ Dataset loaded:")
        print(f"   Train: {len(self.dataset['train'])} samples")
        print(f"   Validation: {len(self.dataset['validation'])} samples")
        print(f"   Test: {len(self.dataset['test'])} samples")
        
        sample = self.dataset['train'][0]
        print(f"\n📝 Sample data:")
        print(f"   Text: '{sample['text'][:100]}...'")
        print(f"   Label: {sample['label']}")
        
        return self.dataset
    
    def benchmark_task_specific_models(self, model_configs):
        print("\n🎯 TASK-SPECIFIC MODEL BENCHMARKING")
        print("=" * 50)
        
        task_results = {}
        
        for model_name, model_path in model_configs.items():
            print(f"\n🔧 Testing {model_name}")
            print(f"   Model: {model_path}")
            
            try:
                pipe = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                start_time = time.time()
                
                predictions = []
                for text in tqdm(self.dataset["test"]["text"], desc="Classifying"):
                    outputs = pipe(text)
                    
                    if len(outputs[0]) == 3:
                        neg_score = outputs[0][0]["score"]
                        pos_score = outputs[0][2]["score"]
                        pred = 1 if pos_score > neg_score else 0
                    else:
                        pred = 1 if outputs[0][1]["score"] > outputs[0][0]["score"] else 0
                    
                    predictions.append(pred)
                
                inference_time = time.time() - start_time
                
                task_results[model_name] = self._evaluate_predictions(
                    self.dataset["test"]["label"], 
                    predictions,
                    model_name,
                    inference_time
                )
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                task_results[model_name] = {"error": str(e)}
        
        self.results["task_specific"] = task_results
        return task_results
    
    def benchmark_embedding_classifiers(self, embedding_models, classifiers):
        print("\n🧠 EMBEDDING + CLASSIFIER BENCHMARKING")
        print("=" * 50)
        
        embedding_results = {}
        
        for emb_name, emb_model_path in embedding_models.items():
            print(f"\n📊 Testing embedding model: {emb_name}")
            
            model = SentenceTransformer(emb_model_path)
            
            if emb_name not in self.embeddings_cache:
                print("   Generating embeddings...")
                train_embeddings = model.encode(
                    self.dataset["train"]["text"], 
                    show_progress_bar=True,
                    batch_size=32
                )
                test_embeddings = model.encode(
                    self.dataset["test"]["text"], 
                    show_progress_bar=True,
                    batch_size=32
                )
                
                self.embeddings_cache[emb_name] = {
                    'train': train_embeddings,
                    'test': test_embeddings
                }
            else:
                train_embeddings = self.embeddings_cache[emb_name]['train']
                test_embeddings = self.embeddings_cache[emb_name]['test']
            
            emb_results = {}
            
            for clf_name, clf_class in classifiers.items():
                print(f"   🔬 Training {clf_name}...")
                
                start_time = time.time()
                
                clf = clf_class(random_state=42)
                clf.fit(train_embeddings, self.dataset["train"]["label"])
                
                train_time = time.time() - start_time
                
                start_time = time.time()
                predictions = clf.predict(test_embeddings)
                inference_time = time.time() - start_time
                
                emb_results[clf_name] = self._evaluate_predictions(
                    self.dataset["test"]["label"],
                    predictions,
                    f"{emb_name}+{clf_name}",
                    inference_time,
                    train_time
                )
                
                if hasattr(clf, 'predict_proba'):
                    probas = clf.predict_proba(test_embeddings)[:, 1]
                    auc = roc_auc_score(self.dataset["test"]["label"], probas)
                    emb_results[clf_name]['auc'] = auc
            
            embedding_results[emb_name] = emb_results
        
        self.results["embedding_classifiers"] = embedding_results
        return embedding_results
    
    def benchmark_zero_shot_classification(self, embedding_model, label_variations):
        print("\n🎯 ZERO-SHOT CLASSIFICATION BENCHMARKING")
        print("=" * 50)
        
        model = SentenceTransformer(embedding_model)
        
        if 'zero_shot_base' not in self.embeddings_cache:
            print("📊 Generating document embeddings...")
            test_embeddings = model.encode(
                self.dataset["test"]["text"], 
                show_progress_bar=True
            )
            self.embeddings_cache['zero_shot_base'] = test_embeddings
        else:
            test_embeddings = self.embeddings_cache['zero_shot_base']
        
        zero_shot_results = {}
        
        for variation_name, labels in label_variations.items():
            print(f"\n🏷️ Testing label variation: {variation_name}")
            print(f"   Labels: {labels}")
            
            start_time = time.time()
            
            label_embeddings = model.encode(labels)
            
            sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
            predictions = np.argmax(sim_matrix, axis=1)
            
            inference_time = time.time() - start_time
            
            zero_shot_results[variation_name] = self._evaluate_predictions(
                self.dataset["test"]["label"],
                predictions,
                f"ZeroShot-{variation_name}",
                inference_time
            )
            
            confidence_scores = np.max(sim_matrix, axis=1)
            zero_shot_results[variation_name]['avg_confidence'] = np.mean(confidence_scores)
            zero_shot_results[variation_name]['min_confidence'] = np.min(confidence_scores)
        
        self.results["zero_shot"] = zero_shot_results
        return zero_shot_results
    
    def benchmark_generative_models(self, generative_configs):
        print("\n🤖 GENERATIVE MODEL BENCHMARKING")
        print("=" * 50)
        
        generative_results = {}
        
        for model_name, config in generative_configs.items():
            print(f"\n🎭 Testing {model_name}")
            
            try:
                pipe = pipeline(
                    config['task'],
                    model=config['model_path'],
                    device=0 if torch.cuda.is_available() else -1
                )
                
                start_time = time.time()
                predictions = []
                
                for text in tqdm(self.dataset["test"]["text"][:100], desc=f"Testing {model_name}"):
                    prompt = config['prompt_template'].replace("[TEXT]", text)
                    
                    output = pipe(prompt, max_new_tokens=10, do_sample=False)
                    
                    if config['task'] == 'text2text-generation':
                        generated = output[0]['generated_text'].lower()
                    else:
                        generated = output[0]['generated_text'].lower()
                    
                    if 'positive' in generated or '1' in generated:
                        pred = 1
                    elif 'negative' in generated or '0' in generated:
                        pred = 0
                    else:
                        pred = np.random.randint(0, 2)
                    
                    predictions.append(pred)
                
                inference_time = time.time() - start_time
                
                generative_results[model_name] = self._evaluate_predictions(
                    self.dataset["test"]["label"][:100],
                    predictions,
                    model_name,
                    inference_time
                )
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                generative_results[model_name] = {"error": str(e)}
        
        self.results["generative"] = generative_results
        return generative_results
    
    def _evaluate_predictions(self, y_true, y_pred, model_name, inference_time, train_time=None):
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'samples_per_second': len(y_pred) / inference_time
        }
        
        if train_time:
            results['train_time'] = train_time
        
        print(f"   📊 Results for {model_name}:")
        print(f"      Accuracy: {accuracy:.3f}")
        print(f"      F1 Score: {f1:.3f}")
        print(f"      Inference: {inference_time:.2f}s ({results['samples_per_second']:.1f} samples/s)")
        
        return results
    
    def compare_all_methods(self):
        print("\n📈 COMPREHENSIVE COMPARISON")
        print("=" * 50)
        
        all_results = []
        
        for category, methods in self.results.items():
            if category == "embedding_classifiers":
                for emb_model, classifiers in methods.items():
                    for clf_name, metrics in classifiers.items():
                        if 'error' not in metrics:
                            metrics['category'] = f"{category}_{emb_model}"
                            metrics['method'] = clf_name
                            all_results.append(metrics)
            else:
                for method_name, metrics in methods.items():
                    if 'error' not in metrics:
                        metrics['category'] = category
                        metrics['method'] = method_name
                        all_results.append(metrics)
        
        df = pd.DataFrame(all_results)
        
        if len(df) > 0:
            df_sorted = df.sort_values('f1_score', ascending=False)
            
            print("🏆 Top 10 Methods by F1 Score:")
            print(df_sorted[['model_name', 'f1_score', 'accuracy', 'samples_per_second']].head(10).to_string(index=False))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0, 0].barh(range(len(df_sorted)), df_sorted['f1_score'])
            axes[0, 0].set_yticks(range(len(df_sorted)))
            axes[0, 0].set_yticklabels(df_sorted['model_name'], fontsize=8)
            axes[0, 0].set_xlabel('F1 Score')
            axes[0, 0].set_title('F1 Score Comparison')
            
            axes[0, 1].scatter(df['inference_time'], df['f1_score'], alpha=0.7)
            axes[0, 1].set_xlabel('Inference Time (s)')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Performance vs Speed')
            
            category_f1 = df.groupby('category')['f1_score'].mean()
            axes[1, 0].bar(category_f1.index, category_f1.values)
            axes[1, 0].set_ylabel('Average F1 Score')
            axes[1, 0].set_title('Performance by Category')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            axes[1, 1].scatter(df['samples_per_second'], df['f1_score'], alpha=0.7)
            axes[1, 1].set_xlabel('Samples/Second')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Throughput vs Performance')
            axes[1, 1].set_xscale('log')
            
            plt.tight_layout()
            plt.show()
        
        return df_sorted if len(df) > 0 else None
    
    def analyze_error_patterns(self, best_model_predictions, model_name):
        print(f"\n🔍 ERROR ANALYSIS FOR {model_name}")
        print("=" * 50)
        
        y_true = self.dataset["test"]["label"]
        y_pred = best_model_predictions
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        errors = np.where(np.array(y_true) != np.array(y_pred))[0]
        
        plt.subplot(1, 2, 2)
        error_lengths = [len(self.dataset["test"]["text"][i].split()) for i in errors]
        correct_lengths = [len(self.dataset["test"]["text"][i].split()) 
                          for i in range(len(y_true)) if i not in errors]
        
        plt.hist([error_lengths, correct_lengths], bins=20, alpha=0.7, 
                label=['Errors', 'Correct'], density=True)
        plt.xlabel('Document Length (words)')
        plt.ylabel('Density')
        plt.title('Error Distribution by Length')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Error rate: {len(errors) / len(y_true):.3f}")
        print(f"Avg length of errors: {np.mean(error_lengths):.1f} words")
        print(f"Avg length of correct: {np.mean(correct_lengths):.1f} words")
        
        false_positives = [i for i in errors if y_true[i] == 0 and y_pred[i] == 1]
        false_negatives = [i for i in errors if y_true[i] == 1 and y_pred[i] == 0]
        
        print(f"\n🔍 Error Examples:")
        print(f"False Positives (predicted positive, actually negative): {len(false_positives)}")
        if false_positives:
            for i in false_positives[:3]:
                print(f"   '{self.dataset['test']['text'][i][:100]}...'")
        
        print(f"\nFalse Negatives (predicted negative, actually positive): {len(false_negatives)}")
        if false_negatives:
            for i in false_negatives[:3]:
                print(f"   '{self.dataset['test']['text'][i][:100]}...'")

def main_classification_benchmark():
    print("📊 Chapter 4: Comprehensive Text Classification Benchmark")
    print("=" * 60)
    
    benchmark = TextClassificationBenchmark()
    benchmark.load_data()
    
    print("\n" + "="*60)
    print("🎯 PART 1: TASK-SPECIFIC MODELS")
    
    task_specific_models = {
        "Twitter-RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
        "BERT-base": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    
    benchmark.benchmark_task_specific_models(task_specific_models)
    
    print("\n" + "="*60)
    print("🧠 PART 2: EMBEDDING + CLASSIFIER MODELS")
    
    embedding_models = {
        "MPNet": "sentence-transformers/all-mpnet-base-v2",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "RoBERTa": "sentence-transformers/all-roberta-large-v1"
    }
    
    classifiers = {
        "LogisticRegression": LogisticRegression,
        "SVM": SVC,
        "RandomForest": RandomForestClassifier
    }
    
    benchmark.benchmark_embedding_classifiers(embedding_models, classifiers)
    
    print("\n" + "="*60)
    print("🎯 PART 3: ZERO-SHOT CLASSIFICATION")
    
    label_variations = {
        "basic": ["negative", "positive"],
        "descriptive": ["negative movie review", "positive movie review"],
        "detailed": ["a very negative movie review", "a very positive movie review"],
        "emotional": ["disappointing and bad movie", "amazing and excellent movie"]
    }
    
    benchmark.benchmark_zero_shot_classification(
        "sentence-transformers/all-mpnet-base-v2",
        label_variations
    )
    
    print("\n" + "="*60)
    print("🤖 PART 4: GENERATIVE MODELS")
    
    generative_configs = {
        "Flan-T5-small": {
            "task": "text2text-generation",
            "model_path": "google/flan-t5-small",
            "prompt_template": "Is the following review positive or negative? [TEXT]"
        },
        "T5-base": {
            "task": "text2text-generation", 
            "model_path": "t5-base",
            "prompt_template": "sentiment: [TEXT]"
        }
    }
    
    benchmark.benchmark_generative_models(generative_configs)
    
    print("\n" + "="*60)
    print("📈 PART 5: COMPREHENSIVE COMPARISON")
    
    results_df = benchmark.compare_all_methods()
    
    if results_df is not None and len(results_df) > 0:
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best performing model: {best_model['model_name']}")
        print(f"   F1 Score: {best_model['f1_score']:.3f}")
        print(f"   Accuracy: {best_model['accuracy']:.3f}")
        print(f"   Speed: {best_model['samples_per_second']:.1f} samples/second")
    
    print("\n🎉 Benchmark Complete!")

if __name__ == "__main__":
    main_classification_benchmark()
```

---

# Chapter 4 Advanced Classification Exercises

## 🎯 **Classification Strategy Selection**

### **Exercise 1: Method Selection Matrix**
```python
def classification_method_selector(task_requirements):
    """
    Intelligent method selection based on task constraints
    """
    selection_matrix = {
        'performance_priority': {
            'high_data': 'fine_tuned_model',
            'medium_data': 'embedding_classifier', 
            'low_data': 'few_shot_prompting'
        },
        'speed_priority': {
            'high_throughput': 'task_specific_model',
            'medium_throughput': 'embedding_classifier',
            'low_throughput': 'generative_model'
        },
        'cost_priority': {
            'minimal_cost': 'zero_shot_embedding',
            'low_cost': 'local_model',
            'medium_cost': 'api_model'
        },
        'interpretability_priority': {
            'high_interpret': 'traditional_ml',
            'medium_interpret': 'embedding_classifier',
            'low_interpret': 'black_box_llm'
        }
    }
    
    recommendations = {}
    
    for priority, constraints in task_requirements.items():
        if priority in selection_matrix:
            for constraint, level in constraints.items():
                if constraint in selection_matrix[priority]:
                    method = selection_matrix[priority][constraint]
                    recommendations[f"{priority}_{constraint}"] = method
    
    return recommendations

task_scenarios = {
    'startup_mvp': {
        'performance_priority': {'low_data': True},
        'cost_priority': {'minimal_cost': True},
        'speed_priority': {'medium_throughput': True}
    },
    'enterprise_production': {
        'performance_priority': {'high_data': True},
        'speed_priority': {'high_throughput': True},
        'interpretability_priority': {'high_interpret': True}
    },
    'research_prototype': {
        'performance_priority': {'medium_data': True},
        'cost_priority': {'low_cost': True}
    }
}
```

### **Exercise 2: Multi-Label Classification Extension**
```python
class MultiLabelClassificationBenchmark:
    def __init__(self):
        self.label_types = ['sentiment', 'urgency', 'topic', 'intent']
        
    def create_synthetic_multilabel_data(self, base_dataset, num_samples=1000):
        """
        Create multi-label dataset from single-label data
        """
        import random
        
        multi_labels = []
        texts = []
        
        for i in range(num_samples):
            text = base_dataset['text'][i % len(base_dataset['text'])]
            
            labels = {
                'sentiment': random.choice([0, 1]),  # negative/positive
                'urgency': random.choice([0, 1]),    # low/high urgency
                'topic': random.choice([0, 1, 2]),   # product/service/complaint
                'intent': random.choice([0, 1, 2])   # question/request/feedback
            }
            
            multi_labels.append(labels)
            texts.append(text)
        
        return {'text': texts, 'labels': multi_labels}
    
    def benchmark_multilabel_approaches(self, data, methods):
        """
        Compare different multi-label classification strategies
        """
        results = {}
        
        for method_name, method_config in methods.items():
            print(f"Testing {method_name}...")
            
            if method_config['approach'] == 'binary_relevance':
                results[method_name] = self._binary_relevance_approach(data, method_config)
            elif method_config['approach'] == 'label_powerset':
                results[method_name] = self._label_powerset_approach(data, method_config)
            elif method_config['approach'] == 'chain_classification':
                results[method_name] = self._chain_classification_approach(data, method_config)
        
        return results
    
    def _binary_relevance_approach(self, data, config):
        """
        Train separate classifier for each label
        """
        from sklearn.multioutput import MultiOutputClassifier
        
        # Generate embeddings
        model = SentenceTransformer(config['embedding_model'])
        embeddings = model.encode(data['text'])
        
        # Prepare labels matrix
        label_matrix = np.array([[labels[label_type] for label_type in self.label_types] 
                                for labels in data['labels']])
        
        # Train multi-output classifier
        base_classifier = config['base_classifier']()
        multi_classifier = MultiOutputClassifier(base_classifier)
        
        # Split data
        split_idx = int(0.8 * len(embeddings))
        X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
        y_train, y_test = label_matrix[:split_idx], label_matrix[split_idx:]
        
        # Train and predict
        multi_classifier.fit(X_train, y_train)
        predictions = multi_classifier.predict(X_test)
        
        # Evaluate per label
        results = {}
        for i, label_type in enumerate(self.label_types):
            f1 = f1_score(y_test[:, i], predictions[:, i], average='weighted')
            results[f'{label_type}_f1'] = f1
        
        results['overall_f1'] = np.mean(list(results.values()))
        return results

multilabel_methods = {
    'binary_relevance_lr': {
        'approach': 'binary_relevance',
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'base_classifier': LogisticRegression
    },
    'binary_relevance_rf': {
        'approach': 'binary_relevance', 
        'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
        'base_classifier': RandomForestClassifier
    }
}
```

### **Exercise 3: Cross-Domain Classification**
```python
def cross_domain_classification_study(source_domain, target_domains):
    """
    Study how models trained on one domain perform on others
    """
    
    domain_datasets = {
        'movie_reviews': load_dataset('rotten_tomatoes'),
        'product_reviews': load_dataset('amazon_reviews_multi', 'en'),
        'tweet_sentiment': load_dataset('tweet_eval', 'sentiment'),
        'news_sentiment': load_dataset('financial_phrasebank')
    }
    
    results = {}
    
    # Train on source domain
    source_data = domain_datasets[source_domain]
    
    # Test multiple models
    models_to_test = {
        'domain_specific': f"cardiffnlp/twitter-roberta-base-sentiment-latest",
        'general_purpose': "sentence-transformers/all-mpnet-base-v2"
    }
    
    for model_name, model_path in models_to_test.items():
        model_results = {}
        
        if 'roberta' in model_path:
            # Task-specific model
            pipe = pipeline("sentiment-analysis", model=model_path)
            
            # Test on source domain
            source_preds = []
            for text in source_data['test']['text']:
                pred = pipe(text)[0]
                source_preds.append(1 if pred['label'] == 'POSITIVE' else 0)
            
            source_f1 = f1_score(source_data['test']['label'], source_preds, average='weighted')
            model_results[source_domain] = source_f1
            
        else:
            # Embedding model
            model = SentenceTransformer(model_path)
            
            # Generate embeddings and train classifier
            train_emb = model.encode(source_data['train']['text'])
            test_emb = model.encode(source_data['test']['text'])
            
            clf = LogisticRegression()
            clf.fit(train_emb, source_data['train']['label'])
            
            source_preds = clf.predict(test_emb)
            source_f1 = f1_score(source_data['test']['label'], source_preds, average='weighted')
            model_results[source_domain] = source_f1
        
        # Test on target domains
        for target_domain in target_domains:
            if target_domain in domain_datasets:
                target_data = domain_datasets[target_domain]
                
                if 'roberta' in model_path:
                    target_preds = []
                    for text in target_data['test']['text'][:500]:  # Limit for speed
                        pred = pipe(text)[0]
                        target_preds.append(1 if pred['label'] == 'POSITIVE' else 0)
                    
                    if len(target_preds) > 0:
                        target_f1 = f1_score(target_data['test']['label'][:500], target_preds, average='weighted')
                        model_results[target_domain] = target_f1
                
                else:
                    target_emb = model.encode(target_data['test']['text'][:500])
                    target_preds = clf.predict(target_emb)
                    target_f1 = f1_score(target_data['test']['label'][:500], target_preds, average='weighted')
                    model_results[target_domain] = target_f1
        
        results[model_name] = model_results
    
    # Calculate domain transfer scores
    transfer_analysis = {}
    for model_name, domains in results.items():
        source_score = domains[source_domain]
        transfer_scores = []
        
        for target_domain, target_score in domains.items():
            if target_domain != source_domain:
                transfer_ratio = target_score / source_score
                transfer_scores.append(transfer_ratio)
        
        transfer_analysis[model_name] = {
            'avg_transfer_ratio': np.mean(transfer_scores),
            'min_transfer_ratio': np.min(transfer_scores),
            'domain_robustness': np.std(transfer_scores)
        }
    
    return results, transfer_analysis
```

## 🚀 **Advanced Prompt Engineering**

### **Exercise 4: Systematic Prompt Optimization**
```python
class PromptOptimizationFramework:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.optimization_history = []
    
    def generate_prompt_variations(self, base_task, variation_types):
        """
        Generate systematic prompt variations for testing
        """
        
        prompt_templates = {
            'direct': "Classify the sentiment of: {text}",
            'role_based': "You are a sentiment analysis expert. Classify: {text}",
            'few_shot': """Examples:
Text: 'This movie is amazing!' -> Positive
Text: 'Terrible acting and plot.' -> Negative
Text: 'It was okay, nothing special.' -> Neutral

Classify: {text}""",
            'chain_of_thought': "Let's think step by step about the sentiment of: {text}. Consider the emotional words and overall tone.",
            'structured_output': """Analyze the sentiment of the following text and respond in this format:
Text: {text}
Sentiment: [Positive/Negative/Neutral]
Confidence: [High/Medium/Low]
Reasoning: [Brief explanation]""",
            'comparative': "Compare this text to typical positive and negative reviews, then classify: {text}",
            'contextual': "In the context of movie reviews, classify the sentiment of: {text}",
        }
        
        variations = []
        for variant_name in variation_types:
            if variant_name in prompt_templates:
                variations.append({
                    'name': variant_name,
                    'template': prompt_templates[variant_name]
                })
        
        return variations
    
    def evaluate_prompt_effectiveness(self, prompts, test_data, sample_size=100):
        """
        Systematically evaluate different prompt formulations
        """
        results = {}
        
        for prompt_info in prompts:
            prompt_name = prompt_info['name']
            template = prompt_info['template']
            
            print(f"Testing prompt: {prompt_name}")
            
            predictions = []
            confidences = []
            response_lengths = []
            
            for i, (text, true_label) in enumerate(zip(test_data['text'][:sample_size], 
                                                      test_data['label'][:sample_size])):
                
                formatted_prompt = template.format(text=text)
                
                try:
                    response = self.model(formatted_prompt, max_new_tokens=50)[0]['generated_text']
                    
                    # Extract prediction
                    pred = self._extract_prediction(response)
                    predictions.append(pred)
                    
                    # Extract confidence if available
                    conf = self._extract_confidence(response)
                    confidences.append(conf)
                    
                    response_lengths.append(len(response.split()))
                    
                except Exception as e:
                    print(f"Error with sample {i}: {e}")
                    predictions.append(np.random.choice([0, 1]))
                    confidences.append(0.5)
                    response_lengths.append(0)
            
            # Calculate metrics
            valid_predictions = [p for p in predictions if p is not None]
            if len(valid_predictions) > 0:
                accuracy = accuracy_score(test_data['label'][:len(valid_predictions)], valid_predictions)
                f1 = f1_score(test_data['label'][:len(valid_predictions)], valid_predictions, average='weighted')
                
                results[prompt_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'avg_confidence': np.mean([c for c in confidences if c is not None]),
                    'avg_response_length': np.mean(response_lengths),
                    'success_rate': len(valid_predictions) / sample_size,
                    'template': template
                }
        
        return results
    
    def _extract_prediction(self, response):
        """Extract sentiment prediction from model response"""
        response_lower = response.lower()
        
        if 'positive' in response_lower or 'sentiment: positive' in response_lower:
            return 1
        elif 'negative' in response_lower or 'sentiment: negative' in response_lower:
            return 0
        elif '1' in response_lower:
            return 1
        elif '0' in response_lower:
            return 0
        else:
            return None
    
    def _extract_confidence(self, response):
        """Extract confidence score from response if available"""
        response_lower = response.lower()
        
        if 'high' in response_lower:
            return 0.9
        elif 'medium' in response_lower:
            return 0.7
        elif 'low' in response_lower:
            return 0.5
        else:
            return None
    
    def optimize_prompt_iteratively(self, base_prompts, test_data, iterations=3):
        """
        Iteratively improve prompts based on performance
        """
        current_prompts = base_prompts
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            
            # Evaluate current prompts
            results = self.evaluate_prompt_effectiveness(current_prompts, test_data)
            
            # Find best performing prompt
            best_prompt = max(results.items(), key=lambda x: x[1]['f1_score'])
            print(f"Best prompt: {best_prompt[0]} (F1: {best_prompt[1]['f1_score']:.3f})")
            
            # Generate variations of best prompt
            if iteration < iterations - 1:
                current_prompts = self._generate_prompt_mutations(best_prompt)
        
        return results
    
    def _generate_prompt_mutations(self, best_prompt):
        """Generate mutations of the best-performing prompt"""
        base_template = best_prompt[1]['template']
        
        mutations = [
            {'name': 'original_best', 'template': base_template},
            {'name': 'more_specific', 'template': base_template.replace('Classify', 'Carefully analyze and classify')},
            {'name': 'with_examples', 'template': f"Examples: Good->Positive, Bad->Negative. {base_template}"},
            {'name': 'step_by_step', 'template': f"Think step by step. {base_template}"},
            {'name': 'confident', 'template': base_template + " Be confident in your answer."}
        ]
        
        return mutations
```

### **Exercise 5: Cost-Performance Optimization**
```python
def cost_performance_optimization_study(models_config, dataset, budget_constraints):
    """
    Optimize classification pipeline under budget constraints
    """
    
    cost_models = {
        'openai_gpt4': {'cost_per_1k_tokens': 0.03, 'avg_tokens_per_request': 100},
        'openai_gpt3.5': {'cost_per_1k_tokens': 0.002, 'avg_tokens_per_request': 100},
        'local_flan_t5': {'cost_per_1k_tokens': 0.0, 'avg_tokens_per_request': 100, 'gpu_cost_per_hour': 0.50},
        'local_embedding': {'cost_per_1k_tokens': 0.0, 'avg_tokens_per_request': 50, 'gpu_cost_per_hour': 0.25},
        'zero_shot_embedding': {'cost_per_1k_tokens': 0.0, 'avg_tokens_per_request': 50, 'gpu_cost_per_hour': 0.25}
    }
    
    performance_estimates = {
        'openai_gpt4': {'f1_score': 0.91, 'processing_time_per_sample': 2.0},
        'openai_gpt3.5': {'f1_score': 0.87, 'processing_time_per_sample': 1.5},
        'local_flan_t5': {'f1_score': 0.84, 'processing_time_per_sample': 0.1},
        'local_embedding': {'f1_score': 0.85, 'processing_time_per_sample': 0.05},
        'zero_shot_embedding': {'f1_score': 0.78, 'processing_time_per_sample': 0.05}
    }
    
    optimization_results = {}
    
    for budget_scenario, constraints in budget_constraints.items():
        print(f"\nOptimizing for scenario: {budget_scenario}")
        print(f"Constraints: {constraints}")
        
        scenario_results = {}
        
        for model_name, cost_info in cost_models.items():
            perf_info = performance_estimates[model_name]
            
            # Calculate costs
            if 'openai' in model_name:
                # API-based cost
                tokens_needed = len(dataset) * cost_info['avg_tokens_per_request']
                total_cost = (tokens_needed / 1000) * cost_info['cost_per_1k_tokens']
                processing_time = len(dataset) * perf_info['processing_time_per_sample']
                
            else:
                # Local model cost
                processing_time = len(dataset) * perf_info['processing_time_per_sample']
                gpu_hours_needed = processing_time / 3600  # Convert to hours
                total_cost = gpu_hours_needed * cost_info['gpu_cost_per_hour']
            
            # Check if within constraints
            within_budget = total_cost <= constraints.get('max_cost', float('inf'))
            within_time = processing_time <= constraints.get('max_time_hours', float('inf')) * 3600
            meets_performance = perf_info['f1_score'] >= constraints.get('min_f1_score', 0.0)
            
            feasible = within_budget and within_time and meets_performance
            
            # Calculate efficiency metrics
            cost_per_f1_point = total_cost / perf_info['f1_score'] if perf_info['f1_score'] > 0 else float('inf')
            f1_per_dollar = perf_info['f1_score'] / total_cost if total_cost > 0 else float('inf')
            
            scenario_results[model_name] = {
                'total_cost': total_cost,
                'processing_time_hours': processing_time / 3600,
                'f1_score': perf_info['f1_score'],
                'feasible': feasible,
                'cost_per_f1_point': cost_per_f1_point,
                'f1_per_dollar': f1_per_dollar,
                'within_budget': within_budget,
                'within_time': within_time,
                'meets_performance': meets_performance
            }
        
        # Find optimal solutions
        feasible_models = {k: v for k, v in scenario_results.items() if v['feasible']}
        
        if feasible_models:
            # Best F1 score
            best_performance = max(feasible_models.items(), key=lambda x: x[1]['f1_score'])
            # Best cost efficiency
            best_efficiency = max(feasible_models.items(), key=lambda x: x[1]['f1_per_dollar'])
            # Balanced approach
            balanced_scores = {}
            for model, metrics in feasible_models.items():
                # Normalize metrics and combine
                norm_f1 = metrics['f1_score'] / max(m['f1_score'] for m in feasible_models.values())
                norm_efficiency = metrics['f1_per_dollar'] / max(m['f1_per_dollar'] for m in feasible_models.values())
                balanced_scores[model] = (norm_f1 + norm_efficiency) / 2
            
            best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
            
            scenario_results['recommendations'] = {
                'best_performance': best_performance[0],
                'best_efficiency': best_efficiency[0], 
                'best_balanced': best_balanced[0]
            }
        else:
            scenario_results['recommendations'] = {
                'error': 'No feasible solutions found for given constraints'
            }
        
        optimization_results[budget_scenario] = scenario_results
    
    return optimization_results

# Example budget scenarios
budget_scenarios = {
    'startup_mvp': {
        'max_cost': 50.0,
        'max_time_hours': 2.0,
        'min_f1_score': 0.75
    },
    'enterprise_batch': {
        'max_cost': 500.0,
        'max_time_hours': 8.0,
        'min_f1_score': 0.85
    },
    'research_high_quality': {
        'max_cost': 200.0,
        'max_time_hours': 24.0,
        'min_f1_score': 0.90
    },
    'real_time_service': {
        'max_cost': 1000.0,
        'max_time_hours': 0.5,
        'min_f1_score': 0.80
    }
}
```

## 🔍 **Advanced Evaluation Methods**

### **Exercise 6: Fairness and Bias Analysis**
```python
class ClassificationFairnessAnalyzer:
    def __init__(self):
        self.bias_metrics = ['demographic_parity', 'equal_opportunity', 'calibration']
        
    def analyze_demographic_bias(self, predictions, true_labels, demographic_groups):
        """
        Analyze classification fairness across demographic groups
        """
        bias_results = {}
        
        for group_name, group_indices in demographic_groups.items():
            group_preds = [predictions[i] for i in group_indices]
            group_labels = [true_labels[i] for i in group_indices]
            
            # Basic performance metrics
            group_accuracy = accuracy_score(group_labels, group_preds)
            group_f1 = f1_score(group_labels, group_preds, average='weighted')
            
            # Fairness metrics
            positive_rate = np.mean(group_preds)  # Rate of positive predictions
            true_positive_rate = np.mean([pred for pred, true in zip(group_preds, group_labels) if true == 1])
            false_positive_rate = np.mean([pred for pred, true in zip(group_preds, group_labels) if true == 0])
            
            bias_results[group_name] = {
                'accuracy': group_accuracy,
                'f1_score': group_f1,
                'positive_prediction_rate': positive_rate,
                'true_positive_rate': true_positive_rate,
                'false_positive_rate': false_positive_rate,
                'sample_size': len(group_indices)
            }
        
        # Calculate disparity metrics
        all_groups = list(bias_results.keys())
        disparities = {}
        
        for metric in ['accuracy', 'f1_score', 'positive_prediction_rate']:
            values = [bias_results[group][metric] for group in all_groups]
            disparities[f'{metric}_range'] = max(values) - min(values)
            disparities[f'{metric}_ratio'] = max(values) / min(values) if min(values) > 0 else float('inf')
        
        return bias_results, disparities
    
    def generate_bias_mitigation_strategies(self, bias_analysis):
        """
        Suggest bias mitigation strategies based on analysis
        """
        strategies = []
        
        disparities = bias_analysis[1]
        
        if disparities['accuracy_range'] > 0.1:
            strategies.append({
                'strategy': 'Data Augmentation',
                'description': 'Increase training data for underperforming groups',
                'implementation': 'Collect more diverse training examples'
            })
        
        if disparities['positive_prediction_rate_ratio'] > 1.5:
            strategies.append({
                'strategy': 'Threshold Adjustment',
                'description': 'Use group-specific classification thresholds',
                'implementation': 'Calibrate decision boundaries per demographic group'
            })
        
        strategies.append({
            'strategy': 'Adversarial Debiasing',
            'description': 'Train model to be invariant to demographic features',
            'implementation': 'Add adversarial loss term during training'
        })
        
        strategies.append({
            'strategy': 'Post-processing Calibration',
            'description': 'Adjust predictions to ensure fairness constraints',
            'implementation': 'Apply fairness-aware post-processing algorithms'
        })
        
        return strategies

def create_synthetic_demographic_groups(dataset_size):
    """
    Create synthetic demographic groups for bias analysis
    """
    np.random.seed(42)
    
    # Simulate demographic groups with different base rates
    groups = {
        'group_a': np.random.choice(dataset_size, size=dataset_size//3, replace=False),
        'group_b': np.random.choice(dataset_size, size=dataset_size//3, replace=False),
        'group_c': np.random.choice(dataset_size, size=dataset_size//4, replace=False)
    }
    
    return groups
```

### **Exercise 7: Model Robustness Testing**
```python
class RobustnessTestSuite:
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.perturbation_methods = [
            'typo_injection',
            'synonym_replacement', 
            'word_insertion',
            'word_deletion',
            'character_swapping'
        ]
    
    def generate_adversarial_examples(self, texts, perturbation_rate=0.1):
        """
        Generate adversarial examples using various perturbation methods
        """
        adversarial_examples = {}
        
        for method in self.perturbation_methods:
            perturbed_texts = []
            
            for text in texts:
                if method == 'typo_injection':
                    perturbed = self._inject_typos(text, perturbation_rate)
                elif method == 'synonym_replacement':
                    perturbed = self._replace_synonyms(text, perturbation_rate)
                elif method == 'word_insertion':
                    perturbed = self._insert_words(text, perturbation_rate)
                elif method == 'word_deletion':
                    perturbed = self._delete_words(text, perturbation_rate)
                elif method == 'character_swapping':
                    perturbed = self._swap_characters(text, perturbation_rate)
                else:
                    perturbed = text
                
                perturbed_texts.append(perturbed)
            
            adversarial_examples[method] = perturbed_texts
        
        return adversarial_examples
    
    def _inject_typos(self, text, rate):
        """Inject random typos into text"""
        import random
        import string
        
        words = text.split()
        num_typos = max(1, int(len(words) * rate))
        
        for _ in range(num_typos):
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            
            if len(word) > 2:
                char_idx = random.randint(1, len(word) - 2)
                typo_char = random.choice(string.ascii_lowercase)
                words[word_idx] = word[:char_idx] + typo_char + word[char_idx+1:]
        
        return ' '.join(words)
    
    def _replace_synonyms(self, text, rate):
        """Replace words with synonyms"""
        synonym_dict = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed']
        }
        
        words = text.split()
        num_replacements = max(1, int(len(words) * rate))
        
        for _ in range(num_replacements):
            for i, word in enumerate(words):
                if word.lower() in synonym_dict:
                    synonyms = synonym_dict[word.lower()]
                    words[i] = np.random.choice(synonyms)
                    break
        
        return ' '.join(words)
    
    def _insert_words(self, text, rate):
        """Insert random words"""
        filler_words = ['really', 'very', 'quite', 'somewhat', 'rather', 'pretty', 'fairly']
        words = text.split()
        num_insertions = max(1, int(len(words) * rate))
        
        for _ in range(num_insertions):
            insert_pos = np.random.randint(0, len(words))
            filler = np.random.choice(filler_words)
            words.insert(insert_pos, filler)
        
        return ' '.join(words)
    
    def _delete_words(self, text, rate):
        """Delete random words (excluding important ones)"""
        words = text.split()
        if len(words) <= 2:
            return text
            
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        deletable_indices = [i for i, word in enumerate(words) if word.lower() in stop_words]
        
        num_deletions = min(len(deletable_indices), max(1, int(len(words) * rate)))
        
        if deletable_indices:
            indices_to_delete = np.random.choice(deletable_indices, size=num_deletions, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_delete]
        
        return ' '.join(words)
    
    def _swap_characters(self, text, rate):
        """Swap adjacent characters"""
        import random
        
        words = text.split()
        num_swaps = max(1, int(len(words) * rate))
        
        for _ in range(num_swaps):
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            
            if len(word) > 1:
                char_idx = random.randint(0, len(word) - 2)
                chars = list(word)
                chars[char_idx], chars[char_idx + 1] = chars[char_idx + 1], chars[char_idx]
                words[word_idx] = ''.join(chars)
        
        return ' '.join(words)
    
    def evaluate_robustness(self, original_texts, original_predictions, adversarial_examples):
        """
        Evaluate model robustness against adversarial examples
        """
        robustness_results = {}
        
        for perturbation_method, perturbed_texts in adversarial_examples.items():
            # Get predictions for perturbed texts
            perturbed_predictions = []
            
            for text in perturbed_texts:
                try:
                    if hasattr(self.model, 'predict'):
                        pred = self.model.predict([text])[0]
                    else:
                        # For pipeline models
                        result = self.model(text)
                        pred = 1 if result[0]['label'] == 'POSITIVE' else 0
                    
                    perturbed_predictions.append(pred)
                except:
                    perturbed_predictions.append(np.random.choice([0, 1]))
            
            # Calculate robustness metrics
            prediction_changes = np.array(original_predictions) != np.array(perturbed_predictions)
            robustness_score = 1 - np.mean(prediction_changes)
            
            robustness_results[perturbation_method] = {
                'robustness_score': robustness_score,
                'prediction_change_rate': np.mean(prediction_changes),
                'total_samples': len(original_texts),
                'changed_predictions': np.sum(prediction_changes)
            }
        
        # Overall robustness
        overall_robustness = np.mean([results['robustness_score'] 
                                     for results in robustness_results.values()])
        
        robustness_results['overall'] = {
            'average_robustness': overall_robustness,
            'most_vulnerable_to': min(robustness_results.items(), 
                                     key=lambda x: x[1]['robustness_score'])[0],
            'most_robust_against': max(robustness_results.items(), 
                                      key=lambda x: x[1]['robustness_score'])[0]
        }
        
        return robustness_results

# Example usage
def run_comprehensive_robustness_test():
    """
    Run complete robustness testing pipeline
    """
    print("🛡️ COMPREHENSIVE ROBUSTNESS TESTING")
    print("=" * 50)
    
    # This would require loaded models and data
    # robustness_tester = RobustnessTestSuite(model)
    # test_texts = dataset['test']['text'][:100]
    # original_preds = [model.predict([text])[0] for text in test_texts]
    
    # adversarial_examples = robustness_tester.generate_adversarial_examples(test_texts)
    # robustness_results = robustness_tester.evaluate_robustness(
    #     test_texts, original_preds, adversarial_examples
    # )
    
    print("Robustness testing framework ready!")
    print("Key capabilities:")
    print("  • Typo injection attacks")
    print("  • Synonym replacement attacks") 
    print("  • Word insertion/deletion attacks")
    print("  • Character swapping attacks")
    print("  • Comprehensive robustness scoring")

if __name__ == "__main__":
    # Run all advanced exercises
    print("🎯 Chapter 4: Advanced Classification Exercises")
    print("=" * 60)
    
    # Uncomment to run specific exercises:
    # run_method_selection_analysis()
    # run_multilabel_classification_benchmark() 
    # run_cross_domain_study()
    # run_prompt_optimization()
    # run_cost_performance_optimization()
    # run_fairness_analysis()
    run_comprehensive_robustness_test()
```

---

## 🎯 **Quick Start Action Plan**

### **Immediate Practice (Today):**
1. **Run the comprehensive benchmark** - Compare all 5 classification approaches
2. **Try prompt engineering exercises** - See how prompt wording affects performance
3. **Test zero-shot variations** - Experiment with different label descriptions

### **This Week's Goals:**
1. **Master embedding-based classification** - Best balance of performance and flexibility
2. **Understand cost-performance tradeoffs** - When to use local vs API models
3. **Practice prompt optimization** - Iterative improvement strategies

### **Advanced Challenges:**
1. **Build multi-label classifier** - Extend to multiple simultaneous classifications
2. **Implement fairness analysis** - Test for demographic bias in predictions
3. **Create robustness testing suite** - Evaluate model vulnerability to attacks

The enhanced framework reveals the full spectrum of text classification approaches, from simple zero-shot to sophisticated multi-label systems, with practical guidance for real-world deployment decisions.

---

## 🎯 **Key Chapter 4 Insights**

### **Strategic Model Selection:**
- **Task-specific models** - Best for single, well-defined domains (F1: 0.80)
- **Embedding + classifier** - Most flexible approach, often best performance (F1: 0.85)
- **Zero-shot classification** - Rapid prototyping without labeled data (F1: 0.78)
- **Generative models** - Powerful but require prompt engineering (F1: 0.84-0.91)

### **Memory Anchors:**
- **"Embeddings are universal features"** - Same vectors work for multiple tasks
- **"Prompt quality = performance quality"** - Small wording changes dramatically affect results
- **"Zero-shot beats many supervised"** - Good embeddings + smart labels > basic training
- **"Cost scales with model size"** - API calls expensive, local models require GPU

### **Practical Decision Framework:**
```python
if labeled_data_available and single_task:
    use_task_specific_model()
elif multiple_tasks or flexibility_needed:
    use_embedding_plus_classifier()
elif no_labeled_data:
    try_zero_shot_first()
elif budget_allows and performance_critical:
    use_chatgpt_api()
else:
    use_local_generative_model()
```

### **Real-World Applications:**
The comprehensive benchmark system enables data-driven decisions for:
- **Customer service automation** - Sentiment + intent + urgency classification
- **Content moderation** - Multi-label toxicity detection
- **Market research** - Brand sentiment across social platforms
- **Legal document analysis** - Contract clause classification

This chapter's enhanced framework transforms theoretical knowledge into production-ready classification systems with proper evaluation, bias analysis, and robustness testing.
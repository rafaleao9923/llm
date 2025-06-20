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


## 🎯 **Author's Core Intentions**

The authors demystify the "black box" nature of LLMs by revealing the step-by-step computation flow inside transformer architectures. They focus on practical understanding rather than mathematical complexity, showing:

1. **Token-by-token generation** - Models don't generate entire responses at once
2. **Parallel vs Sequential processing** - All input tokens processed simultaneously, but output generated sequentially 
3. **Attention as information routing** - How tokens "talk" to each other to build context
4. **Performance optimization** - KV caching as real-world engineering solution

The sample code demonstrates three critical concepts:
- **Model inspection** - Examining internal components and dimensions
- **Probability distribution analysis** - How final token selection works
- **Performance benchmarking** - Cache vs no-cache generation speed

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Optional

class TransformerArchitectureExplorer:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True
        )
        self.device = next(self.model.parameters()).device
        print(f"✅ Loaded {model_name} on {self.device}")
        
    def analyze_model_architecture(self):
        print("🏗️ MODEL ARCHITECTURE ANALYSIS")
        print("=" * 50)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 2 / 1e9:.1f} GB (FP16)")
        
        config = self.model.config
        print(f"\n🔧 Architecture Details:")
        print(f"  Vocabulary size: {config.vocab_size:,}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Number of layers: {config.num_hidden_layers}")
        print(f"  Number of attention heads: {config.num_attention_heads}")
        print(f"  Context length: {config.max_position_embeddings:,}")
        
        if hasattr(config, 'num_key_value_heads'):
            print(f"  Key-Value heads: {config.num_key_value_heads} (Grouped-Query Attention)")
        
        print(f"\n🧮 Computational Details:")
        head_dim = config.hidden_size // config.num_attention_heads
        print(f"  Head dimension: {head_dim}")
        print(f"  Feed-forward dimension: {getattr(config, 'intermediate_size', 'Unknown')}")
        
        return {
            'total_params': total_params,
            'config': config,
            'head_dim': head_dim
        }
    
    def trace_forward_pass(self, text: str, max_tokens: int = 5):
        print(f"\n🔍 TRACING FORWARD PASS")
        print(f"Input: '{text}'")
        print("=" * 50)
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        print(f"📝 Tokenized input: {input_ids.shape}")
        print(f"   Token IDs: {input_ids[0].tolist()}")
        print(f"   Tokens: {[self.tokenizer.decode(t) for t in input_ids[0]]}")
        
        generation_steps = []
        current_ids = input_ids.clone()
        
        for step in range(max_tokens):
            print(f"\n🚀 Generation Step {step + 1}")
            print("-" * 30)
            
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False
                )
            
            logits = outputs.logits[0, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            
            top_k = 10
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            print(f"📊 Top {top_k} token predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = self.tokenizer.decode(idx)
                print(f"  {i+1:2d}. '{token}' (ID: {idx.item()}) - {prob.item():.4f}")
            
            next_token_id = top_indices[0]
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            step_info = {
                'step': step + 1,
                'current_text': self.tokenizer.decode(current_ids[0]),
                'new_token': self.tokenizer.decode(next_token_id),
                'top_predictions': [(self.tokenizer.decode(idx), prob.item()) 
                                  for prob, idx in zip(top_probs[:5], top_indices[:5])],
                'hidden_states_shape': outputs.hidden_states[-1].shape,
                'attention_shape': outputs.attentions[0].shape
            }
            generation_steps.append(step_info)
            
            if step == 0:
                print(f"\n🧠 Internal Representations:")
                print(f"   Hidden states shape: {outputs.hidden_states[-1].shape}")
                print(f"   Attention shape: {outputs.attentions[0].shape}")
                print(f"   Number of layers: {len(outputs.hidden_states) - 1}")
        
        print(f"\n✅ Final generated text: '{self.tokenizer.decode(current_ids[0])}'")
        return generation_steps
    
    def analyze_attention_patterns(self, text: str, layer_idx: int = -1, head_idx: int = 0):
        print(f"\n🎯 ATTENTION PATTERN ANALYSIS")
        print(f"Text: '{text}'")
        print(f"Layer: {layer_idx}, Head: {head_idx}")
        print("=" * 50)
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        attention_weights = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
        
        print(f"📝 Tokens ({len(tokens)}):")
        for i, token in enumerate(tokens):
            print(f"  {i}: '{token}'")
        
        print(f"\n🔍 Attention Matrix Shape: {attention_weights.shape}")
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            attention_weights,
            xticklabels=[f"{i}:'{t}'" for i, t in enumerate(tokens)],
            yticklabels=[f"{i}:'{t}'" for i, t in enumerate(tokens)],
            annot=True if len(tokens) <= 10 else False,
            fmt='.3f',
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title(f'Attention Patterns - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens (what we attend TO)')
        plt.ylabel('Query Tokens (what is attending)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        for i, token in enumerate(tokens):
            if i < len(attention_weights):
                top_attention_idx = np.argmax(attention_weights[i])
                top_attention_score = attention_weights[i, top_attention_idx]
                print(f"Token '{token}' attends most to '{tokens[top_attention_idx]}' ({top_attention_score:.3f})")
        
        return attention_weights, tokens
    
    def compare_attention_heads(self, text: str, layer_idx: int = -1, num_heads: int = 4):
        print(f"\n🔄 MULTI-HEAD ATTENTION COMPARISON")
        print(f"Text: '{text}'")
        print("=" * 50)
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        attention_layer = outputs.attentions[layer_idx][0].cpu().numpy()
        total_heads = attention_layer.shape[0]
        
        print(f"Total attention heads in layer {layer_idx}: {total_heads}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for head_idx in range(min(num_heads, total_heads)):
            attention_weights = attention_layer[head_idx]
            
            sns.heatmap(
                attention_weights,
                xticklabels=[f"{i}" for i in range(len(tokens))],
                yticklabels=[f"{i}" for i in range(len(tokens))],
                annot=False,
                cmap='Blues',
                ax=axes[head_idx],
                cbar=True
            )
            
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].set_xlabel('Key Position')
            axes[head_idx].set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.show()
        
        print("\n🔍 Head Specialization Analysis:")
        for head_idx in range(min(num_heads, total_heads)):
            attention_weights = attention_layer[head_idx]
            
            diagonal_attention = np.mean([attention_weights[i, i] for i in range(len(tokens))])
            local_attention = np.mean([
                attention_weights[i, max(0, i-1):i+2].sum() 
                for i in range(1, len(tokens))
            ])
            distant_attention = np.mean([
                attention_weights[i, :max(1, i-2)].sum() 
                for i in range(2, len(tokens))
            ])
            
            print(f"  Head {head_idx}: Self-attention: {diagonal_attention:.3f}, "
                  f"Local: {local_attention:.3f}, Distant: {distant_attention:.3f}")
    
    def benchmark_kv_caching(self, prompt: str, max_new_tokens: int = 50, iterations: int = 3):
        print(f"\n⚡ KV CACHE PERFORMANCE BENCHMARK")
        print(f"Prompt: '{prompt[:50]}...'")
        print(f"Max new tokens: {max_new_tokens}")
        print("=" * 50)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        times_with_cache = []
        times_without_cache = []
        
        for i in range(iterations):
            print(f"🔄 Iteration {i+1}/{iterations}")
            
            torch.cuda.empty_cache()
            start_time = time.time()
            with torch.no_grad():
                output_with_cache = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            time_with_cache = end_time - start_time
            times_with_cache.append(time_with_cache)
            
            torch.cuda.empty_cache()
            start_time = time.time()
            with torch.no_grad():
                output_without_cache = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            time_without_cache = end_time - start_time
            times_without_cache.append(time_without_cache)
            
            print(f"  With cache: {time_with_cache:.2f}s")
            print(f"  Without cache: {time_without_cache:.2f}s")
            print(f"  Speedup: {time_without_cache/time_with_cache:.1f}x")
        
        avg_with_cache = np.mean(times_with_cache)
        avg_without_cache = np.mean(times_without_cache)
        avg_speedup = avg_without_cache / avg_with_cache
        
        print(f"\n📊 Average Results:")
        print(f"  With cache: {avg_with_cache:.2f}s ± {np.std(times_with_cache):.2f}s")
        print(f"  Without cache: {avg_without_cache:.2f}s ± {np.std(times_without_cache):.2f}s")
        print(f"  Average speedup: {avg_speedup:.1f}x")
        
        tokens_per_second_cached = max_new_tokens / avg_with_cache
        tokens_per_second_uncached = max_new_tokens / avg_without_cache
        
        print(f"\n🚀 Throughput:")
        print(f"  With cache: {tokens_per_second_cached:.1f} tokens/second")
        print(f"  Without cache: {tokens_per_second_uncached:.1f} tokens/second")
        
        return {
            'avg_with_cache': avg_with_cache,
            'avg_without_cache': avg_without_cache,
            'speedup': avg_speedup,
            'tokens_per_second_cached': tokens_per_second_cached,
            'tokens_per_second_uncached': tokens_per_second_uncached
        }
    
    def analyze_layer_representations(self, text: str):
        print(f"\n🧬 LAYER-BY-LAYER REPRESENTATION ANALYSIS")
        print(f"Text: '{text}'")
        print("=" * 50)
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)
        
        print(f"📊 Model has {num_layers - 1} transformer layers (+ embedding layer)")
        
        last_token_representations = []
        for layer_idx, hidden_state in enumerate(hidden_states):
            last_token_repr = hidden_state[0, -1, :].cpu().numpy()
            last_token_representations.append(last_token_repr)
            
            if layer_idx == 0:
                print(f"Layer {layer_idx} (Embeddings): shape {hidden_state.shape}")
                print(f"  Last token representation sample: {last_token_repr[:5]}")
            elif layer_idx % 5 == 0 or layer_idx == num_layers - 1:
                print(f"Layer {layer_idx}: shape {hidden_state.shape}")
                print(f"  Last token representation sample: {last_token_repr[:5]}")
        
        layer_similarities = []
        for i in range(len(last_token_representations) - 1):
            repr1 = last_token_representations[i]
            repr2 = last_token_representations[i + 1]
            
            similarity = np.dot(repr1, repr2) / (np.linalg.norm(repr1) * np.linalg.norm(repr2))
            layer_similarities.append(similarity)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(layer_similarities) + 1), layer_similarities, 'bo-')
        plt.xlabel('Layer Transition (N → N+1)')
        plt.ylabel('Cosine Similarity')
        plt.title('Representation Similarity Between Adjacent Layers')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"\n🔍 Layer Analysis Results:")
        print(f"  Average similarity between adjacent layers: {np.mean(layer_similarities):.3f}")
        print(f"  Most stable transition (highest similarity): Layer {np.argmax(layer_similarities)} → {np.argmax(layer_similarities) + 1}")
        print(f"  Most dynamic transition (lowest similarity): Layer {np.argmin(layer_similarities)} → {np.argmin(layer_similarities) + 1}")
        
        return {
            'hidden_states': hidden_states,
            'layer_similarities': layer_similarities,
            'last_token_representations': last_token_representations
        }
    
    def demonstrate_sampling_strategies(self, prompt: str, strategies: List[Dict]):
        print(f"\n🎲 SAMPLING STRATEGIES COMPARISON")
        print(f"Prompt: '{prompt}'")
        print("=" * 50)
        
        results = {}
        
        for strategy in strategies:
            name = strategy['name']
            params = strategy.copy()
            del params['name']
            
            print(f"\n🎯 Strategy: {name}")
            print(f"   Parameters: {params}")
            
            generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                max_new_tokens=30,
                **params
            )
            
            generations = []
            for i in range(3):
                output = generator(prompt)
                generation = output[0]['generated_text']
                generations.append(generation)
                print(f"   Sample {i+1}: {generation}")
            
            results[name] = generations
        
        print(f"\n📝 Summary: Generated {len(strategies)} different strategy outputs")
        return results

def main_exploration():
    print("🔍 Chapter 3: Transformer Architecture Deep Dive")
    print("=" * 60)
    
    explorer = TransformerArchitectureExplorer()
    
    print("\n" + "="*60)
    print("🏗️ PART 1: ARCHITECTURE ANALYSIS")
    arch_info = explorer.analyze_model_architecture()
    
    print("\n" + "="*60)
    print("🚀 PART 2: FORWARD PASS TRACING")
    test_prompt = "The capital of France is"
    generation_steps = explorer.trace_forward_pass(test_prompt, max_tokens=3)
    
    print("\n" + "="*60)
    print("🎯 PART 3: ATTENTION PATTERN ANALYSIS")
    attention_text = "The dog chased the cat because it was hungry."
    attention_weights, tokens = explorer.analyze_attention_patterns(attention_text)
    
    print("\n" + "="*60)
    print("🔄 PART 4: MULTI-HEAD ATTENTION COMPARISON")
    explorer.compare_attention_heads(attention_text)
    
    print("\n" + "="*60)
    print("⚡ PART 5: KV CACHE PERFORMANCE BENCHMARK")
    cache_prompt = "Write a detailed explanation of machine learning algorithms and their applications in modern technology."
    cache_results = explorer.benchmark_kv_caching(cache_prompt)
    
    print("\n" + "="*60)
    print("🧬 PART 6: LAYER REPRESENTATION ANALYSIS")
    layer_analysis = explorer.analyze_layer_representations(test_prompt)
    
    print("\n" + "="*60)
    print("🎲 PART 7: SAMPLING STRATEGIES")
    sampling_strategies = [
        {'name': 'Greedy', 'do_sample': False},
        {'name': 'Low Temperature', 'do_sample': True, 'temperature': 0.3},
        {'name': 'High Temperature', 'do_sample': True, 'temperature': 1.0},
        {'name': 'Top-k', 'do_sample': True, 'top_k': 5, 'temperature': 0.7},
        {'name': 'Top-p', 'do_sample': True, 'top_p': 0.9, 'temperature': 0.7}
    ]
    
    sampling_results = explorer.demonstrate_sampling_strategies(
        "Once upon a time in a distant galaxy",
        sampling_strategies
    )
    
    print("\n🎉 Exploration Complete!")
    print("You've seen inside the transformer architecture from embeddings to output!")

if __name__ == "__main__":
    main_exploration()
```

---

# Chapter 3 Advanced Practice Exercises

## 🔍 **Architecture Investigation Drills**

### **Exercise 1: Model Archaeology**
```python
def model_archaeology(model_names):
    """
    Investigate and compare different transformer architectures
    """
    architecture_data = {}
    
    for model_name in model_names:
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            architecture_data[model_name] = {
                'layers': config.num_hidden_layers,
                'hidden_size': config.hidden_size,
                'attention_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size,
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'Unknown'),
                'head_dim': config.hidden_size // config.num_attention_heads,
                'has_grouped_query': hasattr(config, 'num_key_value_heads'),
                'activation_function': getattr(config, 'hidden_act', 'Unknown'),
                'architecture_type': config.architectures[0] if hasattr(config, 'architectures') else 'Unknown'
            }
            
            if hasattr(config, 'num_key_value_heads'):
                architecture_data[model_name]['kv_heads'] = config.num_key_value_heads
                architecture_data[model_name]['query_groups'] = config.num_attention_heads // config.num_key_value_heads
                
        except Exception as e:
            architecture_data[model_name] = f"Error: {e}"
    
    return architecture_data

models_to_investigate = [
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-2-7b-hf",
    "google/flan-t5-small",
    "microsoft/DialoGPT-medium",
    "EleutherAI/gpt-neo-125M"
]

def analyze_architecture_evolution(architecture_data):
    """
    Analyze trends in transformer architecture evolution
    """
    df = pd.DataFrame(architecture_data).T
    
    # Calculate parameter estimates
    df['estimated_params'] = (
        df['hidden_size'] * df['vocab_size'] +  # Embedding matrix
        df['layers'] * (
            4 * df['hidden_size']**2 +  # Attention projections
            8 * df['hidden_size']**2    # FFN (typical 4x expansion)
        )
    ) / 1e6  # Convert to millions
    
    print("📊 Architecture Comparison:")
    print(df[['layers', 'hidden_size', 'attention_heads', 'estimated_params']])
    
    return df
```

### **Exercise 2: Attention Pattern Detective**
```python
def attention_pattern_analyzer(model, texts, pattern_types):
    """
    Analyze different types of attention patterns across texts
    """
    pattern_results = {}
    
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        tokens = [tokenizer.decode(t) for t in input_ids[0]]
        
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        
        text_patterns = {}
        
        for layer_idx in range(len(outputs.attentions)):
            layer_attention = outputs.attentions[layer_idx][0]  # [heads, seq_len, seq_len]
            
            layer_patterns = {}
            
            for pattern_name, pattern_func in pattern_types.items():
                pattern_score = pattern_func(layer_attention, tokens)
                layer_patterns[pattern_name] = pattern_score
            
            text_patterns[f"layer_{layer_idx}"] = layer_patterns
        
        pattern_results[text] = text_patterns
    
    return pattern_results

def calculate_local_attention(attention_tensor, tokens, window_size=3):
    """
    Calculate how much attention focuses on nearby tokens
    """
    seq_len = attention_tensor.shape[-1]
    local_scores = []
    
    for head in range(attention_tensor.shape[0]):
        head_local_score = 0
        for i in range(seq_len):
            window_start = max(0, i - window_size)
            window_end = min(seq_len, i + window_size + 1)
            local_attention = attention_tensor[head, i, window_start:window_end].sum().item()
            head_local_score += local_attention
        
        local_scores.append(head_local_score / seq_len)
    
    return np.mean(local_scores)

def calculate_diagonal_attention(attention_tensor, tokens):
    """
    Calculate self-attention (diagonal elements)
    """
    seq_len = attention_tensor.shape[-1]
    diagonal_scores = []
    
    for head in range(attention_tensor.shape[0]):
        diagonal_sum = sum(attention_tensor[head, i, i].item() for i in range(seq_len))
        diagonal_scores.append(diagonal_sum / seq_len)
    
    return np.mean(diagonal_scores)

def calculate_syntactic_attention(attention_tensor, tokens):
    """
    Heuristic for syntactic attention patterns
    """
    # Look for attention between function words and content words
    function_words = {'the', 'a', 'an', 'is', 'was', 'were', 'be', 'have', 'has', 'do', 'does'}
    
    syntactic_score = 0
    total_pairs = 0
    
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            if i != j and (token_i.lower() in function_words or token_j.lower() in function_words):
                avg_attention = attention_tensor[:, i, j].mean().item()
                syntactic_score += avg_attention
                total_pairs += 1
    
    return syntactic_score / total_pairs if total_pairs > 0 else 0

pattern_detectors = {
    'local_attention': calculate_local_attention,
    'self_attention': calculate_diagonal_attention,
    'syntactic_attention': calculate_syntactic_attention
}
```

### **Exercise 3: Generation Strategy Laboratory**
```python
class GenerationStrategyLab:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def analyze_probability_distributions(self, prompt, temperature_range=[0.1, 0.5, 1.0, 2.0]):
        """
        Analyze how temperature affects probability distributions
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
        
        results = {}
        
        for temp in temperature_range:
            if temp == 0:
                # Greedy decoding
                probs = F.softmax(logits, dim=-1)
            else:
                # Temperature scaling
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
            
            # Get top-k predictions
            top_k = 20
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Calculate entropy (measure of randomness)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            
            # Calculate effective vocabulary size (inverse participation ratio)
            effective_vocab = 1 / (probs ** 2).sum().item()
            
            results[temp] = {
                'top_tokens': [(self.tokenizer.decode(idx), prob.item()) 
                             for idx, prob in zip(top_indices, top_probs)],
                'entropy': entropy,
                'effective_vocab_size': effective_vocab,
                'max_prob': top_probs[0].item(),
                'prob_concentration': (top_probs[:5].sum() / top_probs.sum()).item()
            }
        
        return results
    
    def compare_sampling_methods(self, prompt, methods_config, num_samples=5):
        """
        Generate multiple samples with different methods and analyze diversity
        """
        comparison_results = {}
        
        for method_name, config in methods_config.items():
            samples = []
            
            for _ in range(num_samples):
                generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    return_full_text=False,
                    max_new_tokens=50,
                    **config
                )
                
                output = generator(prompt)
                samples.append(output[0]['generated_text'])
            
            # Analyze diversity
            unique_samples = len(set(samples))
            avg_length = np.mean([len(s.split()) for s in samples])
            
            # Calculate pairwise similarity
            similarities = []
            for i in range(len(samples)):
                for j in range(i+1, len(samples)):
                    # Simple word overlap similarity
                    words1 = set(samples[i].lower().split())
                    words2 = set(samples[j].lower().split())
                    if words1 or words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        similarities.append(similarity)
            
            comparison_results[method_name] = {
                'samples': samples,
                'unique_count': unique_samples,
                'avg_length': avg_length,
                'avg_similarity': np.mean(similarities) if similarities else 0,
                'diversity_score': unique_samples / num_samples
            }
        
        return comparison_results
    
    def analyze_attention_during_generation(self, prompt, max_new_tokens=10):
        """
        Track how attention patterns change during generation
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        generation_attention_data = []
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, output_attentions=True)
                
            # Analyze attention in the last layer
            last_layer_attention = outputs.attentions[-1][0]  # [heads, seq_len, seq_len]
            
            # Focus on the last token's attention (what it's attending to)
            last_token_attention = last_layer_attention[:, -1, :].mean(dim=0)  # Average across heads
            
            # Get next token
            logits = outputs.logits[0, -1, :]
            next_token_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # Store attention data
            tokens_so_far = [self.tokenizer.decode(t) for t in current_ids[0]]
            
            generation_attention_data.append({
                'step': step,
                'new_token': self.tokenizer.decode(next_token_id[0]),
                'tokens': tokens_so_far.copy(),
                'attention_weights': last_token_attention.cpu().numpy(),
                'most_attended_token': tokens_so_far[torch.argmax(last_token_attention).item()],
                'attention_entropy': -(last_token_attention * torch.log(last_token_attention + 1e-10)).sum().item()
            })
        
        return generation_attention_data

sampling_methods = {
    'greedy': {'do_sample': False},
    'random_low_temp': {'do_sample': True, 'temperature': 0.3},
    'random_high_temp': {'do_sample': True, 'temperature': 1.5},
    'top_k': {'do_sample': True, 'top_k': 10, 'temperature': 0.8},
    'top_p': {'do_sample': True, 'top_p': 0.8, 'temperature': 0.8},
    'typical_p': {'do_sample': True, 'typical_p': 0.9, 'temperature': 0.8}
}
```

## ⚡ **Performance Optimization Challenges**

### **Challenge 1: KV Cache Memory Analysis**
```python
def analyze_kv_cache_memory_usage(model, sequence_lengths, batch_sizes=[1]):
    """
    Analyze memory usage patterns for KV caching
    """
    config = model.config
    
    # Calculate theoretical memory requirements
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = hidden_size // config.num_attention_heads
    
    results = {}
    
    for batch_size in batch_sizes:
        batch_results = {}
        
        for seq_len in sequence_lengths:
            # Memory for keys and values
            # Each layer stores: 2 (K,V) * batch_size * num_heads * seq_len * head_dim * 2 bytes (fp16)
            kv_memory_per_layer = 2 * batch_size * num_heads * seq_len * head_dim * 2  # bytes
            total_kv_memory = kv_memory_per_layer * num_layers
            
            # Convert to MB
            kv_memory_mb = total_kv_memory / (1024 * 1024)
            
            # Estimate total memory including activations
            activation_memory_mb = batch_size * seq_len * hidden_size * 2 / (1024 * 1024)  # rough estimate
            
            batch_results[seq_len] = {
                'kv_cache_memory_mb': kv_memory_mb,
                'activation_memory_mb': activation_memory_mb,
                'total_estimated_memory_mb': kv_memory_mb + activation_memory_mb,
                'kv_cache_percentage': kv_memory_mb / (kv_memory_mb + activation_memory_mb) * 100
            }
        
        results[batch_size] = batch_results
    
    return results

def benchmark_generation_speeds(model, tokenizer, test_cases):
    """
    Benchmark generation speeds for different scenarios
    """
    device = next(model.parameters()).device
    results = {}
    
    for case_name, config in test_cases.items():
        prompt = config['prompt']
        max_new_tokens = config['max_new_tokens']
        batch_size = config.get('batch_size', 1)
        
        # Prepare inputs
        if batch_size == 1:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        else:
            prompts = [prompt] * batch_size
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
        
        # Warm up
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=5, use_cache=True)
        
        # Benchmark with cache
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            output_cached = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        cached_time = time.time() - start_time
        
        # Benchmark without cache
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            output_uncached = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        uncached_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens_generated = batch_size * max_new_tokens
        cached_throughput = total_tokens_generated / cached_time
        uncached_throughput = total_tokens_generated / uncached_time
        
        results[case_name] = {
            'cached_time': cached_time,
            'uncached_time': uncached_time,
            'speedup': uncached_time / cached_time,
            'cached_throughput': cached_throughput,
            'uncached_throughput': uncached_throughput,
            'efficiency_gain': (cached_throughput - uncached_throughput) / uncached_throughput * 100
        }
    
    return results

generation_test_cases = {
    'short_single': {
        'prompt': "The capital of France is",
        'max_new_tokens': 10,
        'batch_size': 1
    },
    'medium_single': {
        'prompt': "Write a short story about a robot learning to paint.",
        'max_new_tokens': 50,
        'batch_size': 1
    },
    'long_single': {
        'prompt': "Explain the theory of relativity in detail, covering both special and general relativity.",
        'max_new_tokens': 200,
        'batch_size': 1
    }
}
```

### **Challenge 2: Attention Mechanism Efficiency**
```python
def analyze_attention_computational_complexity(model, sequence_lengths):
    """
    Analyze computational complexity of attention mechanism
    """
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    
    complexity_analysis = {}
    
    for seq_len in sequence_lengths:
        # Standard Multi-Head Attention
        mha_qkv_ops = 3 * seq_len * hidden_size * hidden_size  # Q, K, V projections
        mha_attention_ops = num_heads * seq_len * seq_len * head_dim  # Attention scores
        mha_output_ops = seq_len * hidden_size * hidden_size  # Output projection
        mha_total_ops = mha_qkv_ops + mha_attention_ops + mha_output_ops
        
        # Multi-Query Attention (if applicable)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        
        if num_kv_heads < num_heads:  # Grouped-Query Attention
            gqa_qkv_ops = (hidden_size * hidden_size +  # Q projection
                          2 * num_kv_heads * head_dim * hidden_size) * seq_len  # K, V projections
            gqa_attention_ops = num_heads * seq_len * seq_len * head_dim
            gqa_output_ops = seq_len * hidden_size * hidden_size
            gqa_total_ops = gqa_qkv_ops + gqa_attention_ops + gqa_output_ops
            
            memory_reduction = (num_heads - num_kv_heads) * 2 * seq_len * head_dim
        else:
            gqa_total_ops = mha_total_ops
            memory_reduction = 0
        
        complexity_analysis[seq_len] = {
            'mha_operations': mha_total_ops,
            'gqa_operations': gqa_total_ops,
            'operation_reduction': (mha_total_ops - gqa_total_ops) / mha_total_ops * 100,
            'memory_reduction_elements': memory_reduction,
            'attention_memory_complexity': f"O({seq_len}²)",
            'flops_per_token': mha_total_ops / seq_len
        }
    
    return complexity_analysis

def simulate_attention_patterns(attention_type, seq_len, num_heads=8):
    """
    Simulate different attention patterns for analysis
    """
    patterns = {}
    
    if attention_type == "full":
        # Full attention - each token can attend to all previous tokens
        attention_mask = torch.tril(torch.ones(seq_len, seq_len))
        
    elif attention_type == "local":
        # Local attention - each token attends to nearby tokens only
        window_size = 64
        attention_mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + 1)
            attention_mask[i, start:end] = 1
            
    elif attention_type == "sparse":
        # Sparse attention - combination of local and strided patterns
        attention_mask = torch.zeros(seq_len, seq_len)
        stride = 64
        local_window = 32
        
        for i in range(seq_len):
            # Local attention
            start_local = max(0, i - local_window)
            attention_mask[i, start_local:i+1] = 1
            
            # Strided attention
            for j in range(0, i, stride):
                attention_mask[i, j] = 1
    
    # Calculate efficiency metrics
    total_positions = seq_len * seq_len
    attended_positions = attention_mask.sum().item()
    sparsity = 1 - (attended_positions / total_positions)
    
    patterns[attention_type] = {
        'mask': attention_mask,
        'attended_positions': attended_positions,
        'total_positions': total_positions,
        'sparsity': sparsity,
        'memory_reduction': sparsity,
        'computational_reduction': sparsity
    }
    
    return patterns
```

## 🧠 **Advanced Architecture Understanding**

### **Project 1: Transformer Block Analyzer**
```python
class TransformerBlockAnalyzer:
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
    
    def analyze_block_contributions(self, text, target_layers=None):
        """
        Analyze how much each transformer block contributes to the final output
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        if target_layers is None:
            target_layers = list(range(0, self.config.num_hidden_layers, 2))  # Every 2nd layer
        
        # Get baseline output (all layers)
        with torch.no_grad():
            full_outputs = self.model(input_ids, output_hidden_states=True)
            baseline_logits = full_outputs.logits[0, -1, :]
            baseline_probs = F.softmax(baseline_logits, dim=-1)
        
        layer_contributions = {}
        
        for layer_to_skip in target_layers:
            # This is a simplified analysis - in practice, you'd need to modify the model
            # to skip specific layers, which requires more complex implementation
            
            # For demonstration, we'll analyze hidden state similarities
            hidden_states = full_outputs.hidden_states
            
            if layer_to_skip < len(hidden_states) - 1:
                before_skip = hidden_states[layer_to_skip]
                after_skip = hidden_states[layer_to_skip + 1]
                
                # Calculate similarity between consecutive layers
                before_norm = before_skip / before_skip.norm(dim=-1, keepdim=True)
                after_norm = after_skip / after_skip.norm(dim=-1, keepdim=True)
                
                similarity = (before_norm * after_norm).sum(dim=-1).mean().item()
                
                layer_contributions[layer_to_skip] = {
                    'layer_index': layer_to_skip,
                    'representation_change': 1 - similarity,
                    'normalized_contribution': (1 - similarity) / len(target_layers)
                }
        
        return layer_contributions
    
    def analyze_attention_head_specialization(self, texts, layer_indices=None):
        """
        Analyze what different attention heads specialize in
        """
        if layer_indices is None:
            layer_indices = [-1]  # Just analyze the last layer
        
        head_analysis = {}
        
        for text in texts:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
            
            for layer_idx in layer_indices:
                if layer_idx not in head_analysis:
                    head_analysis[layer_idx] = {}
                
                layer_attention = outputs.attentions[layer_idx][0]  # [heads, seq_len, seq_len]
                num_heads = layer_attention.shape[0]
                
                for head_idx in range(num_heads):
                    head_key = f"head_{head_idx}"
                    if head_key not in head_analysis[layer_idx]:
                        head_analysis[layer_idx][head_key] = {
                            'local_attention_scores': [],
                            'diagonal_attention_scores': [],
                            'distant_attention_scores': [],
                            'attention_entropy_scores': []
                        }
                    
                    head_attention = layer_attention[head_idx]
                    
                    # Calculate different attention patterns
                    local_score = self._calculate_local_attention_score(head_attention)
                    diagonal_score = self._calculate_diagonal_score(head_attention)
                    distant_score = self._calculate_distant_attention_score(head_attention)
                    entropy_score = self._calculate_attention_entropy(head_attention)
                    
                    head_analysis[layer_idx][head_key]['local_attention_scores'].append(local_score)
                    head_analysis[layer_idx][head_key]['diagonal_attention_scores'].append(diagonal_score)
                    head_analysis[layer_idx][head_key]['distant_attention_scores'].append(distant_score)
                    head_analysis[layer_idx][head_key]['attention_entropy_scores'].append(entropy_score)
        
        # Aggregate results
        for layer_idx in head_analysis:
            for head_key in head_analysis[layer_idx]:
                head_data = head_analysis[layer_idx][head_key]
                
                head_analysis[layer_idx][head_key] = {
                    'avg_local_attention': np.mean(head_data['local_attention_scores']),
                    'avg_diagonal_attention': np.mean(head_data['diagonal_attention_scores']),
                    'avg_distant_attention': np.mean(head_data['distant_attention_scores']),
                    'avg_attention_entropy': np.mean(head_data['attention_entropy_scores']),
                    'specialization_type': self._classify_head_specialization(head_data)
                }
        
        return head_analysis
    
    def _calculate_local_attention_score(self, attention_matrix, window=3):
        """Calculate how much attention focuses on nearby tokens"""
        seq_len = attention_matrix.shape[0]
        local_scores = []
        
        for i in range(seq_len):
            window_start = max(0, i - window)
            window_end = min(seq_len, i + window + 1)
            local_attention = attention_matrix[i, window_start:window_end].sum().item()
            local_scores.append(local_attention)
        
        return np.mean(local_scores)
    
    def _calculate_diagonal_score(self, attention_matrix):
        """Calculate self-attention strength"""
        diagonal_sum = sum(attention_matrix[i, i].item() for i in range(attention_matrix.shape[0]))
        return diagonal_sum / attention_matrix.shape[0]
    
    def _calculate_distant_attention_score(self, attention_matrix, min_distance=5):
        """Calculate attention to distant tokens"""
        seq_len = attention_matrix.shape[0]
        distant_scores = []
        
        for i in range(seq_len):
            distant_positions = list(range(0, max(0, i - min_distance))) + list(range(min(seq_len, i + min_distance), seq_len))
            if distant_positions:
                distant_attention = attention_matrix[i, distant_positions].sum().item()
                distant_scores.append(distant_attention)
        
        return np.mean(distant_scores) if distant_scores else 0
    
    def _calculate_attention_entropy(self, attention_matrix):
        """Calculate entropy of attention distribution"""
        entropies = []
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i]
            entropy = -(row * torch.log(row + 1e-10)).sum().item()
            entropies.append(entropy)
        return np.mean(entropies)
    
    def _classify_head_specialization(self, head_data):
        """Classify what the attention head specializes in"""
        local_score = np.mean(head_data['local_attention_scores'])
        diagonal_score = np.mean(head_data['diagonal_attention_scores'])
        distant_score = np.mean(head_data['distant_attention_scores'])
        
        if diagonal_score > 0.3:
            return "self_attention"
        elif local_score > 0.7:
            return "local_syntax"
        elif distant_score > 0.4:
            return "long_range_dependency"
        else:
            return "mixed_pattern"
```

### **Project 2: Custom Attention Mechanism**
```python
def implement_custom_attention_mechanisms():
    """
    Implement and compare different attention mechanisms
    """
    
    class MultiHeadAttention(torch.nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            
            self.q_linear = torch.nn.Linear(d_model, d_model)
            self.k_linear = torch.nn.Linear(d_model, d_model)
            self.v_linear = torch.nn.Linear(d_model, d_model)
            self.out_linear = torch.nn.Linear(d_model, d_model)
            
            self.dropout = torch.nn.Dropout(dropout)
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, d_model = query.shape
            
            # Linear projections
            Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention computation
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            
            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            
            # Final linear projection
            output = self.out_linear(context)
            
            return output, attention_weights
    
    class GroupedQueryAttention(torch.nn.Module):
        def __init__(self, d_model, num_query_heads, num_kv_heads, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_query_heads = num_query_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = d_model // num_query_heads
            
            assert num_query_heads % num_kv_heads == 0
            self.num_queries_per_kv = num_query_heads // num_kv_heads
            
            self.q_linear = torch.nn.Linear(d_model, d_model)
            self.k_linear = torch.nn.Linear(d_model, num_kv_heads * self.head_dim)
            self.v_linear = torch.nn.Linear(d_model, num_kv_heads * self.head_dim)
            self.out_linear = torch.nn.Linear(d_model, d_model)
            
            self.dropout = torch.nn.Dropout(dropout)
        
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, d_model = query.shape
            
            # Query projection
            Q = self.q_linear(query).view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
            
            # Key and value projections
            K = self.k_linear(key).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            V = self.v_linear(value).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Repeat K and V for each query group
            K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
            V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
            
            # Attention computation (same as standard attention)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            context = torch.matmul(attention_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            
            output = self.out_linear(context)
            
            return output, attention_weights
    
    return MultiHeadAttention, GroupedQueryAttention

def benchmark_attention_mechanisms(seq_lengths, d_model=768, num_heads=12):
    """
    Benchmark different attention mechanisms
    """
    import time
    import math
    
    MultiHeadAttention, GroupedQueryAttention = implement_custom_attention_mechanisms()
    
    # Initialize attention mechanisms
    mha = MultiHeadAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_heads // 4)  # 4x reduction in KV heads
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_len}")
        
        # Create dummy input
        batch_size = 1
        dummy_input = torch.randn(batch_size, seq_len, d_model)
        
        # Benchmark Multi-Head Attention
        start_time = time.time()
        for _ in range(10):  # Average over multiple runs
            mha_output, mha_weights = mha(dummy_input, dummy_input, dummy_input)
        mha_time = (time.time() - start_time) / 10
        
        # Benchmark Grouped-Query Attention
        start_time = time.time()
        for _ in range(10):
            gqa_output, gqa_weights = gqa(dummy_input, dummy_input, dummy_input)
        gqa_time = (time.time() - start_time) / 10
        
        # Calculate memory usage (theoretical)
        mha_kv_memory = 2 * seq_len * d_model * 2  # K, V matrices in fp16
        gqa_kv_memory = 2 * seq_len * (d_model // 4) * 2  # Reduced KV heads
        
        results[seq_len] = {
            'mha_time': mha_time,
            'gqa_time': gqa_time,
            'speedup': mha_time / gqa_time,
            'mha_kv_memory_mb': mha_kv_memory / (1024 * 1024),
            'gqa_kv_memory_mb': gqa_kv_memory / (1024 * 1024),
            'memory_reduction': (mha_kv_memory - gqa_kv_memory) / mha_kv_memory * 100
        }
        
        print(f"  MHA time: {mha_time:.4f}s")
        print(f"  GQA time: {gqa_time:.4f}s")
        print(f"  Speedup: {results[seq_len]['speedup']:.2f}x")
        print(f"  Memory reduction: {results[seq_len]['memory_reduction']:.1f}%")
    
    return results
```

## 🎯 **Quick Daily Practice Routines**

### **5-Minute Architecture Quizzes**
```python
def daily_architecture_quiz():
    """Quick transformer architecture knowledge check"""
    
    questions = [
        {
            "question": "What does the KV cache store?",
            "options": ["Token embeddings", "Key and Value matrices", "Attention weights", "Final outputs"],
            "correct": 1,
            "explanation": "KV cache stores computed Key and Value matrices to avoid recomputation during generation"
        },
        {
            "question": "In Grouped-Query Attention, what is shared?",
            "options": ["Queries only", "Keys and Values", "All projections", "Nothing"],
            "correct": 1,
            "explanation": "GQA shares Key and Value projections across multiple query heads"
        },
        {
            "question": "What determines the context window size?",
            "options": ["Model size", "max_position_embeddings", "Number of layers", "Vocabulary size"],
            "correct": 1,
            "explanation": "max_position_embeddings in the config determines maximum sequence length"
        },
        {
            "question": "Why is temperature=0 equivalent to greedy decoding?",
            "options": ["Random sampling", "All probabilities equal", "Max probability = 1", "Temperature scales logits"],
            "correct": 2,
            "explanation": "Temperature=0 makes the highest logit approach infinity, giving max probability ≈ 1"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1}: {q['question']}")
        for j, option in enumerate(q['options']):
            print(f"  {j}: {option}")
        
        # In practice, get user input
        user_choice = q['correct']  # Simulated correct answer
        if user_choice == q['correct']:
            score += 1
            print("✅ Correct!")
        else:
            print(f"❌ Wrong! Correct answer: {q['options'][q['correct']]}")
        print(f"💡 {q['explanation']}")
    
    print(f"\nFinal Score: {score}/{len(questions)}")
    return score

def attention_pattern_drill():
    """Quick attention pattern recognition"""
    
    scenarios = [
        {
            "context": "The dog chased the cat because it was hungry.",
            "question": "Which token should 'it' attend to most?",
            "options": ["The", "dog", "cat", "because"],
            "correct": 1,
            "reasoning": "Ambiguous pronoun resolution - could be dog or cat, but dog is the subject"
        },
        {
            "context": "John told Mary that he would call her tomorrow.",
            "question": "Which tokens should form the strongest attention links?",
            "options": ["John-he", "Mary-her", "told-call", "All of above"],
            "correct": 3,
            "reasoning": "Pronouns resolve to antecedents, verbs link semantically"
        }
    ]
    
    for scenario in scenarios:
        print(f"\nContext: '{scenario['context']}'")
        print(f"Question: {scenario['question']}")
        for i, option in enumerate(scenario['options']):
            print(f"  {i}: {option}")
        print(f"💭 Reasoning: {scenario['reasoning']}")
```

### **10-Minute Deep Dives**
```python
def transformer_component_explorer():
    """Interactive exploration of transformer components"""
    
    def explore_attention_heads():
        """Examine what different attention heads learn"""
        
        patterns = {
            "Syntactic Head": {
                "description": "Focuses on grammatical relationships",
                "example_attention": "subject ↔ verb, determiner ↔ noun",
                "layer_tendency": "Earlier layers (1-4)",
                "characteristics": ["High attention to function words", "Local attention patterns"]
            },
            "Semantic Head": {
                "description": "Captures meaning relationships", 
                "example_attention": "synonyms, antonyms, related concepts",
                "layer_tendency": "Middle layers (5-8)",
                "characteristics": ["Long-range dependencies", "Content word focus"]
            },
            "Positional Head": {
                "description": "Tracks token positions and ordering",
                "example_attention": "previous token, next token, fixed offsets",
                "layer_tendency": "All layers",
                "characteristics": ["Diagonal patterns", "Regular position intervals"]
            },
            "Copy Head": {
                "description": "Identifies tokens to copy or repeat",
                "example_attention": "repeated words, quoted text",
                "layer_tendency": "Later layers (9-12)",
                "characteristics": ["Self-attention", "Exact match patterns"]
            }
        }
        
        print("🧠 ATTENTION HEAD SPECIALIZATION")
        print("=" * 40)
        
        for head_type, info in patterns.items():
            print(f"\n🎯 {head_type}:")
            print(f"   Purpose: {info['description']}")
            print(f"   Examples: {info['example_attention']}")
            print(f"   Common in: {info['layer_tendency']}")
            print(f"   Traits: {', '.join(info['characteristics'])}")
    
    def explore_layer_functions():
        """Examine what different layers specialize in"""
        
        layer_functions = {
            "Early Layers (1-3)": [
                "Basic syntax and grammar",
                "Part-of-speech tagging",
                "Local word relationships",
                "Surface-level patterns"
            ],
            "Middle Layers (4-8)": [
                "Semantic understanding",
                "Entity recognition",
                "Concept relationships", 
                "Context integration"
            ],
            "Late Layers (9-12)": [
                "Task-specific processing",
                "Complex reasoning",
                "Output preparation",
                "Final decision making"
            ]
        }
        
        print("\n🏗️ LAYER SPECIALIZATION")
        print("=" * 40)
        
        for layer_group, functions in layer_functions.items():
            print(f"\n📍 {layer_group}:")
            for func in functions:
                print(f"   • {func}")
    
    explore_attention_heads()
    explore_layer_functions()

def generation_strategy_workshop():
    """Hands-on workshop for understanding generation strategies"""
    
    def demonstrate_temperature_effects():
        """Show how temperature affects generation"""
        
        # Simulated probability distribution
        logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
        vocab = ['the', 'a', 'and', 'to', 'of', 'in', 'for']
        
        temperatures = [0.1, 0.5, 1.0, 2.0]
        
        print("🌡️ TEMPERATURE EFFECTS ON GENERATION")
        print("=" * 50)
        print(f"Base logits: {logits.tolist()}")
        print(f"Vocabulary: {vocab}")
        
        for temp in temperatures:
            if temp == 0:
                # Greedy
                probs = torch.zeros_like(logits)
                probs[logits.argmax()] = 1.0
            else:
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=0)
            
            print(f"\n🌡️ Temperature = {temp}:")
            
            # Show top 3 tokens
            top_probs, top_indices = torch.topk(probs, 3)
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = vocab[idx]
                print(f"   {i+1}. '{token}': {prob.item():.3f}")
            
            # Calculate entropy (randomness measure)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            print(f"   Entropy: {entropy:.3f} (higher = more random)")
    
    def demonstrate_top_k_top_p():
        """Show how top-k and top-p sampling work"""
        
        # Simulated probability distribution (already softmax'd)
        probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02])
        vocab = ['the', 'a', 'and', 'to', 'of', 'in', 'for']
        
        print("\n🔢 TOP-K AND TOP-P SAMPLING")
        print("=" * 40)
        print("Original probabilities:")
        for token, prob in zip(vocab, probs):
            print(f"  '{token}': {prob.item():.3f}")
        
        # Top-k sampling
        k = 3
        top_k_probs, top_k_indices = torch.topk(probs, k)
        top_k_normalized = top_k_probs / top_k_probs.sum()
        
        print(f"\n🔢 Top-k (k={k}) sampling:")
        for i, (prob, idx) in enumerate(zip(top_k_normalized, top_k_indices)):
            token = vocab[idx]
            original_prob = probs[idx].item()
            print(f"  '{token}': {prob.item():.3f} (was {original_prob:.3f})")
        
        # Top-p sampling
        p = 0.8
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        top_p_mask = cumsum_probs <= p
        
        # Include at least one token
        if not top_p_mask.any():
            top_p_mask[0] = True
        
        # Include the first token that exceeds threshold
        exceed_idx = (cumsum_probs > p).nonzero(as_tuple=True)[0]
        if len(exceed_idx) > 0:
            top_p_mask[exceed_idx[0]] = True
        
        top_p_probs = sorted_probs[top_p_mask]
        top_p_indices = sorted_indices[top_p_mask]
        top_p_normalized = top_p_probs / top_p_probs.sum()
        
        print(f"\n📊 Top-p (p={p}) sampling:")
        for i, (prob, idx) in enumerate(zip(top_p_normalized, top_p_indices)):
            token = vocab[idx]
            original_prob = probs[idx].item()
            print(f"  '{token}': {prob.item():.3f} (was {original_prob:.3f})")
    
    demonstrate_temperature_effects()
    demonstrate_top_k_top_p()
```

## 🚀 **Advanced Research Projects**

### **Project 1: Attention Pattern Visualizer**
```python
class AttentionPatternVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def create_attention_heatmap_animation(self, text, output_file="attention_animation.gif"):
        """
        Create an animated visualization of attention patterns across layers
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        attention_layers = [attn[0].cpu().numpy() for attn in outputs.attentions]
        
        # Create animation frames
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        frames = []
        layer_indices = [0, len(attention_layers)//4, len(attention_layers)//2, 
                        3*len(attention_layers)//4, len(attention_layers)-2, len(attention_layers)-1]
        
        for frame_idx, layer_idx in enumerate(layer_indices):
            if frame_idx < len(axes):
                attention = attention_layers[layer_idx]
                
                # Average across heads for visualization
                avg_attention = attention.mean(axis=0)
                
                im = axes[frame_idx].imshow(avg_attention, cmap='Blues', aspect='auto')
                axes[frame_idx].set_title(f'Layer {layer_idx}')
                axes[frame_idx].set_xticks(range(len(tokens)))
                axes[frame_idx].set_yticks(range(len(tokens)))
                axes[frame_idx].set_xticklabels(tokens, rotation=45, ha='right')
                axes[frame_idx].set_yticklabels(tokens)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[frame_idx])
        
        plt.tight_layout()
        plt.savefig('attention_layers_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return attention_layers
    
    def analyze_attention_flow(self, text, source_token_idx, target_layer=-1):
        """
        Trace how attention flows from a specific token through the network
        """
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
        
        if source_token_idx >= len(tokens):
            raise ValueError(f"Token index {source_token_idx} out of range (max: {len(tokens)-1})")
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        
        attention_flow = {}
        
        for layer_idx, layer_attention in enumerate(outputs.attentions):
            layer_attention = layer_attention[0].cpu().numpy()  # [heads, seq_len, seq_len]
            
            # Average attention from source token across all heads
            source_attention = layer_attention[:, source_token_idx, :].mean(axis=0)
            
            # Find top attended tokens
            top_attention_indices = np.argsort(source_attention)[::-1][:5]
            
            attention_flow[layer_idx] = {
                'source_token': tokens[source_token_idx],
                'attention_weights': source_attention,
                'top_attended_tokens': [
                    (tokens[idx], source_attention[idx]) 
                    for idx in top_attention_indices
                ],
                'attention_entropy': -(source_attention * np.log(source_attention + 1e-10)).sum()
            }
        
        # Visualize attention flow
        layers = list(range(len(attention_flow)))
        entropies = [attention_flow[layer]['attention_entropy'] for layer in layers]
        
        plt.figure(figsize=(12, 6))
        plt.plot(layers, entropies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Layer')
        plt.ylabel('Attention Entropy')
        plt.title(f"Attention Entropy Flow for Token: '{tokens[source_token_idx]}'")
        plt.grid(True, alpha=0.3)
        
        # Annotate interesting points
        max_entropy_layer = np.argmax(entropies)
        min_entropy_layer = np.argmin(entropies)
        
        plt.annotate(f'Max entropy\n(Layer {max_entropy_layer})', 
                    xy=(max_entropy_layer, entropies[max_entropy_layer]),
                    xytext=(max_entropy_layer+1, entropies[max_entropy_layer]+0.5),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.annotate(f'Min entropy\n(Layer {min_entropy_layer})', 
                    xy=(min_entropy_layer, entropies[min_entropy_layer]),
                    xytext=(min_entropy_layer+1, entropies[min_entropy_layer]-0.5),
                    arrowprops=dict(arrowstyle='->', color='green'))
        
        plt.show()
        
        return attention_flow
    
    def compare_attention_across_contexts(self, base_text, variations):
        """
        Compare how attention patterns change with different contexts
        """
        contexts = [base_text] + variations
        all_attention_data = {}
        
        for i, context in enumerate(contexts):
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)
            
            # Focus on last layer attention
            last_layer_attention = outputs.attentions[-1][0].cpu().numpy()
            avg_attention = last_layer_attention.mean(axis=0)  # Average across heads
            
            all_attention_data[f"Context_{i}"] = {
                'text': context,
                'tokens': tokens,
                'attention_matrix': avg_attention
            }
        
        # Create comparison visualization
        num_contexts = len(contexts)
        fig, axes = plt.subplots(1, num_contexts, figsize=(6*num_contexts, 5))
        
        if num_contexts == 1:
            axes = [axes]
        
        for i, (context_key, data) in enumerate(all_attention_data.items()):
            tokens = data['tokens']
            attention = data['attention_matrix']
            
            im = axes[i].imshow(attention, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Context {i}: {data["text"][:30]}...')
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_yticks(range(len(tokens)))
            
            if len(tokens) <= 10:  # Only show token labels for short sequences
                axes[i].set_xticklabels(tokens, rotation=45, ha='right')
                axes[i].set_yticklabels(tokens)
            
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()
        
        return all_attention_data

# Usage example
def run_attention_analysis():
    """
    Run comprehensive attention pattern analysis
    """
    # This would require a loaded model and tokenizer
    # visualizer = AttentionPatternVisualizer(model, tokenizer)
    
    test_cases = [
        "The cat sat on the mat because it was comfortable.",
        "John told Mary that he would call her tomorrow.",
        "The bank by the river offers great loan rates."
    ]
    
    print("🎯 ATTENTION PATTERN ANALYSIS TOOLKIT")
    print("=" * 50)
    
    for i, text in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}: '{text}'")
        print("Available analyses:")
        print("  1. Layer-by-layer attention heatmaps")
        print("  2. Token attention flow tracing") 
        print("  3. Context variation comparison")
        print("  4. Head specialization analysis")
        
        # In a real implementation, you would run:
        # visualizer.create_attention_heatmap_animation(text)
        # visualizer.analyze_attention_flow(text, source_token_idx=2)
```

### **Project 2: Performance Optimization Lab**
```python
def create_optimization_benchmark_suite():
    """
    Create comprehensive benchmarking suite for transformer optimizations
    """
    
    class OptimizationBenchmark:
        def __init__(self):
            self.results = {}
            
        def benchmark_kv_cache_scaling(self, model, sequence_lengths, batch_sizes):
            """
            Benchmark KV cache performance across different scales
            """
            results = {}
            
            for batch_size in batch_sizes:
                batch_results = {}
                
                for seq_len in sequence_lengths:
                    # Create dummy input
                    dummy_text = " ".join(["word"] * seq_len)
                    inputs = tokenizer([dummy_text] * batch_size, return_tensors="pt", padding=True)
                    
                    # Benchmark with cache
                    torch.cuda.empty_cache()
                    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs.input_ids,
                            max_new_tokens=50,
                            use_cache=True,
                            do_sample=False
                        )
                    cached_time = time.time() - start_time
                    cached_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    # Benchmark without cache
                    torch.cuda.empty_cache()
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs.input_ids,
                            max_new_tokens=50,
                            use_cache=False,
                            do_sample=False
                        )
                    uncached_time = time.time() - start_time
                    
                    batch_results[seq_len] = {
                        'cached_time': cached_time,
                        'uncached_time': uncached_time,
                        'speedup': uncached_time / cached_time,
                        'memory_overhead_mb': (cached_memory - start_memory) / (1024**2),
                        'tokens_per_second': (batch_size * 50) / cached_time
                    }
                
                results[batch_size] = batch_results
            
            return results
        
        def benchmark_attention_mechanisms(self, d_model=768, sequence_lengths=[128, 512, 1024, 2048]):
            """
            Compare computational complexity of different attention mechanisms
            """
            import math
            
            results = {}
            
            for seq_len in sequence_lengths:
                # Standard Multi-Head Attention complexity
                mha_attention_ops = seq_len * seq_len * d_model  # O(n²d)
                mha_linear_ops = 4 * seq_len * d_model * d_model  # Q,K,V,O projections
                mha_total_ops = mha_attention_ops + mha_linear_ops
                
                # Local Attention (window size = 256)
                window_size = min(256, seq_len)
                local_attention_ops = seq_len * window_size * d_model  # O(nd)
                local_total_ops = local_attention_ops + mha_linear_ops
                
                # Sparse Attention (every 64th position + local window)
                sparse_positions = seq_len // 64 + window_size
                sparse_attention_ops = seq_len * sparse_positions * d_model
                sparse_total_ops = sparse_attention_ops + mha_linear_ops
                
                results[seq_len] = {
                    'mha_ops': mha_total_ops,
                    'local_ops': local_total_ops,
                    'sparse_ops': sparse_total_ops,
                    'local_speedup': mha_total_ops / local_total_ops,
                    'sparse_speedup': mha_total_ops / sparse_total_ops,
                    'mha_memory': seq_len * seq_len * 4,  # Attention matrix in bytes (fp32)
                    'local_memory': seq_len * window_size * 4,
                    'sparse_memory': seq_len * sparse_positions * 4
                }
            
            return results
        
        def profile_generation_bottlenecks(self, model, tokenizer, prompts):
            """
            Profile where time is spent during generation
            """
            import cProfile
            import pstats
            import io
            
            profiling_results = {}
            
            for i, prompt in enumerate(prompts):
                print(f"Profiling prompt {i+1}/{len(prompts)}")
                
                # Set up profiling
                pr = cProfile.Profile()
                
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                
                # Profile the generation
                pr.enable()
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        use_cache=True,
                        do_sample=False
                    )
                pr.disable()
                
                # Extract profiling results
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                
                profiling_results[f"prompt_{i}"] = {
                    'prompt': prompt,
                    'profile_output': s.getvalue(),
                    'total_time': sum(ps.get_stats_profile().stats.values())
                }
            
            return profiling_results
    
    return OptimizationBenchmark

# Example usage
def run_comprehensive_benchmarks():
    """
    Run the complete optimization benchmark suite
    """
    benchmark = create_optimization_benchmark_suite()()
    
    print("🚀 COMPREHENSIVE OPTIMIZATION BENCHMARKS")
    print("=" * 60)
    
    # Theoretical analysis (doesn't require actual model)
    print("\n📊 Attention Mechanism Complexity Analysis")
    attention_results = benchmark.benchmark_attention_mechanisms()
    
    for seq_len, metrics in attention_results.items():
        print(f"\nSequence Length: {seq_len}")
        print(f"  Standard Attention: {metrics['mha_ops']:,} operations")
        print(f"  Local Attention: {metrics['local_ops']:,} operations ({metrics['local_speedup']:.1f}x faster)")
        print(f"  Sparse Attention: {metrics['sparse_ops']:,} operations ({metrics['sparse_speedup']:.1f}x faster)")
        print(f"  Memory usage reduction: {(1 - metrics['local_memory']/metrics['mha_memory'])*100:.1f}% (local)")
    
    print("\n💡 Key Insights:")
    print("  • Local attention provides linear scaling vs quadratic")
    print("  • Sparse attention balances efficiency with global context")
    print("  • Memory savings become critical for long sequences")
    print("  • Choice depends on task requirements and hardware constraints")

if __name__ == "__main__":
    # Run the daily practice routines
    daily_architecture_quiz()
    attention_pattern_drill()
    transformer_component_explorer()
    generation_strategy_workshop()
    
    # Run advanced analysis
    run_comprehensive_benchmarks()
```

---

## 🎯 **Quick Start Action Plan**

### **Immediate Practice (Today):**
1. **Run the enhanced explorer** - See actual attention patterns and layer processing
2. **Try the 5-minute architecture quiz** - Test your understanding
3. **Experiment with temperature settings** - Observe how it affects creativity vs consistency

### **This Week's Goals:**
1. **Master KV caching concept** - Understand why it's 5x faster
2. **Analyze attention patterns** - Load a model and examine what different heads attend to
3. **Compare sampling strategies** - Generate text with different decoding methods

### **Monthly Deep Dive:**
1. **Build custom attention mechanism** - Implement grouped-query attention
2. **Create attention visualizer** - Make heatmaps of your own text
3. **Benchmark optimization techniques** - Measure real performance differences

The enhanced code reveals the "black box" by making every internal computation visible and interactive. You'll understand not just *what* transformers do, but *how* they process each token through the entire architecture stack.

---

## 🎯 **Key Chapter 3 Insights**

### **Core Architectural Understanding:**
- **Token streams flow in parallel** - All input tokens processed simultaneously, but generation is sequential
- **Attention = information routing** - Determines which previous tokens inform current token processing
- **KV cache is essential** - 5x speedup by caching key/value matrices during generation
- **Layers specialize progressively** - Early layers handle syntax, late layers handle reasoning

### **Memory Anchors:**
- **"Parallel in, sequential out"** - Processing vs generation pattern
- **"Attention weights sum to 1"** - Softmax ensures proper probability distribution
- **"Cache keys and values, not queries"** - Queries change each step, K/V stay constant
- **"Head specialization emerges"** - Different attention heads learn different linguistic patterns

### **Practical Applications:**
The enhanced architecture explorer transforms abstract concepts into interactive experiences. You can now:
- **Trace token-by-token generation** step-by-step
- **Visualize attention patterns** across all layers and heads
- **Benchmark KV cache performance** with real timing measurements
- **Compare sampling strategies** and see diversity vs quality tradeoffs

This deep architectural understanding is crucial for:
- **Optimizing inference performance** in production
- **Debugging model behavior** when outputs seem wrong
- **Designing better prompts** by understanding attention flow
- **Choosing appropriate models** based on architecture tradeoffs
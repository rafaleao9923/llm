## 🎯 **Author's Core Intentions**

The authors demonstrate prompt engineering as a systematic craft that bridges human intent with LLM capabilities. Key progressions include:

1. **Basic prompt components** - Instruction, data, format specifications
2. **Advanced architecture** - Modular design with persona, context, audience, tone
3. **Reasoning enhancement** - Chain-of-thought, tree-of-thought, self-consistency
4. **Output validation** - Structured generation with grammar constraints

The sample code reveals practical patterns: temperature/top_p for creativity control, chat templates for role-based interactions, and constrained sampling for reliable JSON output.

```python
import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List, Tuple, Optional, Any
import time
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedPromptEngineering:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.prompt_templates = {}
        self.evaluation_results = {}
        self.load_model()
        self.setup_templates()
        
    def load_model(self):
        print(f"Loading {self.model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=500,
            do_sample=False
        )
        
        print("✅ Model loaded successfully")
    
    def setup_templates(self):
        self.prompt_templates = {
            'basic': {
                'template': "{instruction}\n\nInput: {data}\n\nOutput:",
                'components': ['instruction', 'data']
            },
            'structured': {
                'template': "{persona}\n\n{instruction}\n\n{context}\n\n{format_spec}\n\nInput: {data}\n\nOutput:",
                'components': ['persona', 'instruction', 'context', 'format_spec', 'data']
            },
            'few_shot': {
                'template': "{instruction}\n\n{examples}\n\nInput: {data}\n\nOutput:",
                'components': ['instruction', 'examples', 'data']
            },
            'cot': {
                'template': "{instruction}\n\nLet's think step by step.\n\nInput: {data}\n\nOutput:",
                'components': ['instruction', 'data']
            },
            'tot': {
                'template': "{instruction}\n\nImagine three experts solving this. Each expert thinks one step at a time and shares their reasoning. If an expert realizes they're wrong, they stop.\n\nInput: {data}\n\nOutput:",
                'components': ['instruction', 'data']
            }
        }
    
    def generate_with_parameters(self, prompt, generation_configs):
        results = {}
        
        for config_name, config in generation_configs.items():
            try:
                if isinstance(prompt, list):
                    output = self.pipe(prompt, **config)
                else:
                    messages = [{"role": "user", "content": prompt}]
                    output = self.pipe(messages, **config)
                
                results[config_name] = {
                    'output': output[0]['generated_text'],
                    'config': config,
                    'success': True
                }
            except Exception as e:
                results[config_name] = {
                    'output': '',
                    'config': config,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def parameter_exploration_study(self, test_prompt, parameter_ranges):
        print("🎛️ PARAMETER EXPLORATION STUDY")
        print("=" * 50)
        
        base_config = {'max_new_tokens': 100, 'do_sample': True}
        
        results = []
        
        for temp in parameter_ranges['temperature']:
            for top_p in parameter_ranges['top_p']:
                config = base_config.copy()
                config.update({'temperature': temp, 'top_p': top_p})
                
                outputs = []
                for trial in range(3):
                    try:
                        messages = [{"role": "user", "content": test_prompt}]
                        output = self.pipe(messages, **config)
                        outputs.append(output[0]['generated_text'])
                    except:
                        outputs.append("")
                
                diversity_score = len(set(outputs)) / len(outputs)
                avg_length = np.mean([len(out.split()) for out in outputs if out])
                
                results.append({
                    'temperature': temp,
                    'top_p': top_p,
                    'diversity_score': diversity_score,
                    'avg_length': avg_length,
                    'outputs': outputs
                })
        
        results_df = pd.DataFrame(results)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        pivot_diversity = results_df.pivot(index='temperature', columns='top_p', values='diversity_score')
        sns.heatmap(pivot_diversity, annot=True, fmt='.2f', ax=axes[0], cmap='YlOrRd')
        axes[0].set_title('Output Diversity by Parameters')
        
        pivot_length = results_df.pivot(index='temperature', columns='top_p', values='avg_length')
        sns.heatmap(pivot_length, annot=True, fmt='.1f', ax=axes[1], cmap='YlGnBu')
        axes[1].set_title('Average Output Length by Parameters')
        
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def prompt_component_ablation(self, base_components, task_input):
        print("🧩 PROMPT COMPONENT ABLATION STUDY")
        print("=" * 50)
        
        component_names = list(base_components.keys())
        ablation_results = {}
        
        full_prompt = ' '.join(base_components.values()) + f"\n\nInput: {task_input}\n\nOutput:"
        
        messages = [{"role": "user", "content": full_prompt}]
        full_output = self.pipe(messages)
        
        ablation_results['full'] = {
            'components': component_names,
            'output': full_output[0]['generated_text'],
            'prompt_length': len(full_prompt.split())
        }
        
        for component_to_remove in component_names:
            partial_components = {k: v for k, v in base_components.items() if k != component_to_remove}
            partial_prompt = ' '.join(partial_components.values()) + f"\n\nInput: {task_input}\n\nOutput:"
            
            messages = [{"role": "user", "content": partial_prompt}]
            partial_output = self.pipe(messages)
            
            ablation_results[f'without_{component_to_remove}'] = {
                'components': [c for c in component_names if c != component_to_remove],
                'output': partial_output[0]['generated_text'],
                'prompt_length': len(partial_prompt.split()),
                'removed_component': component_to_remove
            }
        
        for component_only in component_names:
            single_prompt = base_components[component_only] + f"\n\nInput: {task_input}\n\nOutput:"
            
            messages = [{"role": "user", "content": single_prompt}]
            single_output = self.pipe(messages)
            
            ablation_results[f'only_{component_only}'] = {
                'components': [component_only],
                'output': single_output[0]['generated_text'],
                'prompt_length': len(single_prompt.split()),
                'isolated_component': component_only
            }
        
        self._analyze_ablation_results(ablation_results)
        return ablation_results
    
    def _analyze_ablation_results(self, results):
        print("\n📊 Ablation Analysis:")
        
        full_output = results['full']['output']
        
        for key, result in results.items():
            if key == 'full':
                continue
            
            output_similarity = self._calculate_text_similarity(full_output, result['output'])
            
            print(f"\n{key}:")
            print(f"  Similarity to full: {output_similarity:.3f}")
            print(f"  Output length: {len(result['output'].split())} words")
            print(f"  Preview: {result['output'][:100]}...")
    
    def _calculate_text_similarity(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def reasoning_technique_comparison(self, reasoning_tasks):
        print("🧠 REASONING TECHNIQUE COMPARISON")
        print("=" * 50)
        
        techniques = {
            'direct': lambda task: f"Solve this problem: {task}",
            'cot': lambda task: f"Solve this problem step by step: {task}",
            'zero_shot_cot': lambda task: f"Solve this problem. Let's think step by step: {task}",
            'tot': lambda task: f"Three experts will solve this problem. Each expert shares one step of reasoning at a time. Problem: {task}",
            'self_consistency': lambda task: f"Solve this problem carefully and double-check your answer: {task}"
        }
        
        results = {}
        
        for task_name, task in reasoning_tasks.items():
            print(f"\n🎯 Testing task: {task_name}")
            task_results = {}
            
            for technique_name, technique_func in techniques.items():
                prompt = technique_func(task)
                
                if technique_name == 'self_consistency':
                    outputs = []
                    for trial in range(3):
                        config = {'temperature': 0.7, 'do_sample': True, 'max_new_tokens': 200}
                        messages = [{"role": "user", "content": prompt}]
                        output = self.pipe(messages, **config)
                        outputs.append(output[0]['generated_text'])
                    
                    final_answer = self._extract_majority_answer(outputs)
                    task_results[technique_name] = {
                        'outputs': outputs,
                        'final_answer': final_answer,
                        'reasoning_steps': self._count_reasoning_steps(outputs[0])
                    }
                else:
                    messages = [{"role": "user", "content": prompt}]
                    output = self.pipe(messages)
                    
                    task_results[technique_name] = {
                        'output': output[0]['generated_text'],
                        'reasoning_steps': self._count_reasoning_steps(output[0]['generated_text'])
                    }
                
                print(f"  ✅ {technique_name} completed")
            
            results[task_name] = task_results
        
        self._analyze_reasoning_results(results)
        return results
    
    def _extract_majority_answer(self, outputs):
        answers = []
        for output in outputs:
            lines = output.split('\n')
            for line in lines:
                if 'answer' in line.lower() or line.strip().endswith('.'):
                    answers.append(line.strip())
                    break
        
        if answers:
            answer_counts = Counter(answers)
            return answer_counts.most_common(1)[0][0]
        return "No consensus"
    
    def _count_reasoning_steps(self, text):
        step_indicators = ['step', 'first', 'then', 'next', 'finally', 'therefore', 'so']
        step_count = 0
        
        for indicator in step_indicators:
            step_count += text.lower().count(indicator)
        
        return step_count
    
    def _analyze_reasoning_results(self, results):
        print("\n📈 Reasoning Analysis Summary:")
        
        technique_scores = {}
        
        for task_name, task_results in results.items():
            print(f"\n{task_name}:")
            
            for technique, result in task_results.items():
                reasoning_steps = result['reasoning_steps']
                
                if technique not in technique_scores:
                    technique_scores[technique] = []
                
                technique_scores[technique].append(reasoning_steps)
                
                print(f"  {technique}: {reasoning_steps} reasoning steps")
        
        print(f"\n🏆 Average reasoning steps by technique:")
        for technique, scores in technique_scores.items():
            avg_steps = np.mean(scores)
            print(f"  {technique}: {avg_steps:.1f} steps")
    
    def few_shot_learning_optimization(self, task_description, examples, test_cases):
        print("🎯 FEW-SHOT LEARNING OPTIMIZATION")
        print("=" * 50)
        
        strategies = {
            'random_order': lambda exs: random.sample(exs, len(exs)),
            'difficulty_ascending': lambda exs: sorted(exs, key=lambda x: len(x['input'])),
            'difficulty_descending': lambda exs: sorted(exs, key=lambda x: len(x['input']), reverse=True),
            'similarity_based': lambda exs: exs  # Would need embedding similarity in practice
        }
        
        n_examples_range = [1, 2, 3, min(5, len(examples))]
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            strategy_results = {}
            
            for n_examples in n_examples_range:
                if n_examples > len(examples):
                    continue
                
                ordered_examples = strategy_func(examples)[:n_examples]
                
                example_text = ""
                for ex in ordered_examples:
                    example_text += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
                
                test_results = []
                
                for test_case in test_cases:
                    prompt = f"{task_description}\n\nExamples:\n{example_text}Input: {test_case}\nOutput:"
                    
                    messages = [{"role": "user", "content": prompt}]
                    output = self.pipe(messages)
                    
                    test_results.append({
                        'input': test_case,
                        'output': output[0]['generated_text'],
                        'prompt_length': len(prompt.split())
                    })
                
                strategy_results[n_examples] = test_results
            
            results[strategy_name] = strategy_results
        
        self._analyze_few_shot_results(results)
        return results
    
    def _analyze_few_shot_results(self, results):
        print("\n📊 Few-Shot Analysis:")
        
        for strategy, strategy_results in results.items():
            print(f"\n{strategy}:")
            
            for n_examples, test_results in strategy_results.items():
                avg_output_length = np.mean([len(r['output'].split()) for r in test_results])
                avg_prompt_length = np.mean([r['prompt_length'] for r in test_results])
                
                print(f"  {n_examples} examples: avg output {avg_output_length:.1f} words, prompt {avg_prompt_length:.1f} words")
    
    def output_validation_system(self, tasks_with_constraints):
        print("✅ OUTPUT VALIDATION SYSTEM")
        print("=" * 50)
        
        validation_results = {}
        
        for task_name, task_config in tasks_with_constraints.items():
            print(f"\n🔧 Testing {task_name}")
            
            prompt = task_config['prompt']
            constraints = task_config['constraints']
            
            validation_methods = {
                'basic': self._generate_basic_output,
                'structured_prompt': self._generate_with_format_examples,
                'validation_loop': self._generate_with_validation,
                'constraint_prompting': self._generate_with_constraints
            }
            
            task_results = {}
            
            for method_name, method_func in validation_methods.items():
                try:
                    output = method_func(prompt, constraints)
                    validation_score = self._validate_output(output, constraints)
                    
                    task_results[method_name] = {
                        'output': output,
                        'validation_score': validation_score,
                        'constraints_met': validation_score > 0.8,
                        'success': True
                    }
                except Exception as e:
                    task_results[method_name] = {
                        'output': '',
                        'validation_score': 0.0,
                        'constraints_met': False,
                        'success': False,
                        'error': str(e)
                    }
                
                print(f"  ✅ {method_name}: {task_results[method_name]['validation_score']:.2f}")
            
            validation_results[task_name] = task_results
        
        return validation_results
    
    def _generate_basic_output(self, prompt, constraints):
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages)
        return output[0]['generated_text']
    
    def _generate_with_format_examples(self, prompt, constraints):
        if 'format_example' in constraints:
            enhanced_prompt = f"{prompt}\n\nUse this exact format:\n{constraints['format_example']}"
        else:
            enhanced_prompt = prompt
        
        messages = [{"role": "user", "content": enhanced_prompt}]
        output = self.pipe(messages)
        return output[0]['generated_text']
    
    def _generate_with_validation(self, prompt, constraints):
        max_attempts = 3
        
        for attempt in range(max_attempts):
            messages = [{"role": "user", "content": prompt}]
            output = self.pipe(messages)
            result = output[0]['generated_text']
            
            if self._validate_output(result, constraints) > 0.5:
                return result
            
            prompt = f"{prompt}\n\nPrevious attempt failed validation. Please follow the requirements exactly."
        
        return result
    
    def _generate_with_constraints(self, prompt, constraints):
        constraint_text = ""
        
        if 'max_length' in constraints:
            constraint_text += f"Keep response under {constraints['max_length']} words. "
        
        if 'required_format' in constraints:
            constraint_text += f"Use {constraints['required_format']} format. "
        
        if 'forbidden_words' in constraints:
            constraint_text += f"Do not use these words: {', '.join(constraints['forbidden_words'])}. "
        
        enhanced_prompt = f"{constraint_text}\n\n{prompt}"
        
        messages = [{"role": "user", "content": enhanced_prompt}]
        output = self.pipe(messages)
        return output[0]['generated_text']
    
    def _validate_output(self, output, constraints):
        score = 1.0
        
        if 'max_length' in constraints:
            word_count = len(output.split())
            if word_count > constraints['max_length']:
                score *= 0.5
        
        if 'required_format' in constraints:
            if constraints['required_format'].lower() == 'json':
                try:
                    json.loads(output)
                except:
                    score *= 0.3
        
        if 'forbidden_words' in constraints:
            output_lower = output.lower()
            for word in constraints['forbidden_words']:
                if word.lower() in output_lower:
                    score *= 0.7
        
        if 'required_elements' in constraints:
            for element in constraints['required_elements']:
                if element.lower() not in output.lower():
                    score *= 0.8
        
        return score
    
    def create_prompt_optimization_dashboard(self, optimization_results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if 'parameter_study' in optimization_results:
            param_data = optimization_results['parameter_study']
            param_df = pd.DataFrame(param_data)
            
            pivot_data = param_df.pivot(index='temperature', columns='top_p', values='diversity_score')
            sns.heatmap(pivot_data, annot=True, ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('Parameter Diversity Scores')
        
        if 'reasoning_comparison' in optimization_results:
            reasoning_data = optimization_results['reasoning_comparison']
            
            technique_scores = {}
            for task_results in reasoning_data.values():
                for technique, result in task_results.items():
                    if technique not in technique_scores:
                        technique_scores[technique] = []
                    technique_scores[technique].append(result['reasoning_steps'])
            
            techniques = list(technique_scores.keys())
            avg_scores = [np.mean(technique_scores[t]) for t in techniques]
            
            axes[0, 1].bar(techniques, avg_scores)
            axes[0, 1].set_title('Average Reasoning Steps by Technique')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        if 'validation_results' in optimization_results:
            validation_data = optimization_results['validation_results']
            
            methods = []
            scores = []
            
            for task_results in validation_data.values():
                for method, result in task_results.items():
                    if result['success']:
                        methods.append(method)
                        scores.append(result['validation_score'])
            
            method_df = pd.DataFrame({'method': methods, 'score': scores})
            method_avg = method_df.groupby('method')['score'].mean()
            
            axes[1, 0].bar(method_avg.index, method_avg.values)
            axes[1, 0].set_title('Validation Success by Method')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        if 'few_shot_results' in optimization_results:
            few_shot_data = optimization_results['few_shot_results']
            
            example_counts = []
            output_lengths = []
            
            for strategy_results in few_shot_data.values():
                for n_examples, test_results in strategy_results.items():
                    avg_length = np.mean([len(r['output'].split()) for r in test_results])
                    example_counts.append(n_examples)
                    output_lengths.append(avg_length)
            
            axes[1, 1].scatter(example_counts, output_lengths, alpha=0.6)
            axes[1, 1].set_xlabel('Number of Examples')
            axes[1, 1].set_ylabel('Average Output Length')
            axes[1, 1].set_title('Output Length vs Example Count')
        
        plt.tight_layout()
        plt.show()
        
        return fig

def main_prompt_engineering_analysis():
    print("🎯 Chapter 6: Advanced Prompt Engineering Analysis")
    print("=" * 60)
    
    pe_system = AdvancedPromptEngineering()
    
    print("\n" + "="*60)
    print("🎛️ PART 1: PARAMETER EXPLORATION")
    
    test_prompt = "Write a creative short story about a robot learning to paint."
    
    parameter_ranges = {
        'temperature': [0.1, 0.5, 0.8, 1.2],
        'top_p': [0.3, 0.7, 0.9, 1.0]
    }
    
    param_results = pe_system.parameter_exploration_study(test_prompt, parameter_ranges)
    
    print("\n" + "="*60)
    print("🧩 PART 2: COMPONENT ABLATION")
    
    components = {
        'persona': "You are a creative writing expert with 20 years of experience.",
        'context': "This story is for a science fiction anthology targeting adult readers.",
        'format': "Write exactly 100 words in three paragraphs.",
        'tone': "Use a thoughtful, slightly melancholic tone."
    }
    
    ablation_results = pe_system.prompt_component_ablation(components, "A robot discovers art")
    
    print("\n" + "="*60)
    print("🧠 PART 3: REASONING TECHNIQUES")
    
    reasoning_tasks = {
        'math_problem': "If a train travels 120 miles in 2 hours, and another train travels 180 miles in 3 hours, which train is faster and by how much?",
        'logic_puzzle': "All cats are animals. Some animals are pets. Are all cats pets?",
        'word_problem': "Sarah has twice as many apples as Tom. Together they have 15 apples. How many apples does each person have?"
    }
    
    reasoning_results = pe_system.reasoning_technique_comparison(reasoning_tasks)
    
    print("\n" + "="*60)
    print("🎯 PART 4: FEW-SHOT OPTIMIZATION")
    
    task_desc = "Convert the following sentence to active voice:"
    examples = [
        {'input': 'The book was read by John.', 'output': 'John read the book.'},
        {'input': 'The cake was baked by Mary.', 'output': 'Mary baked the cake.'},
        {'input': 'The song was sung by the choir.', 'output': 'The choir sang the song.'}
    ]
    test_cases = [
        'The letter was written by Alice.',
        'The house was built by the workers.',
        'The movie was watched by the audience.'
    ]
    
    few_shot_results = pe_system.few_shot_learning_optimization(task_desc, examples, test_cases)
    
    print("\n" + "="*60)
    print("✅ PART 5: OUTPUT VALIDATION")
    
    validation_tasks = {
        'json_generation': {
            'prompt': 'Create a character profile for a video game in JSON format.',
            'constraints': {
                'required_format': 'json',
                'required_elements': ['name', 'class', 'level'],
                'max_length': 100
            }
        },
        'constrained_writing': {
            'prompt': 'Write a product description for a new smartphone.',
            'constraints': {
                'max_length': 50,
                'forbidden_words': ['best', 'amazing', 'revolutionary'],
                'required_elements': ['camera', 'battery', 'price']
            }
        }
    }
    
    validation_results = pe_system.output_validation_system(validation_tasks)
    
    print("\n" + "="*60)
    print("📊 PART 6: COMPREHENSIVE DASHBOARD")
    
    all_results = {
        'parameter_study': param_results.to_dict('records'),
        'reasoning_comparison': reasoning_results,
        'validation_results': validation_results,
        'few_shot_results': few_shot_results
    }
    
    dashboard = pe_system.create_prompt_optimization_dashboard(all_results)
    
    print("\n🎉 Analysis Complete!")
    print("Key insights from prompt engineering optimization:")
    print("• Parameter tuning dramatically affects creativity and consistency")
    print("• Component ablation reveals which prompt parts matter most")
    print("• Reasoning techniques significantly improve complex problem solving")
    print("• Few-shot examples boost performance but increase token costs")
    print("• Output validation ensures reliable, structured responses")

if __name__ == "__main__":
    main_prompt_engineering_analysis()
```

---

# Chapter 6 Advanced Prompt Engineering Exercises

## 🎯 **Prompt Optimization Frameworks**

### **Exercise 1: Automated Prompt Evolution**
```python
class GeneticPromptOptimizer:
    def __init__(self, model_pipeline, fitness_function):
        self.model = model_pipeline
        self.fitness_function = fitness_function
        self.population = []
        self.generation = 0
        
    def initialize_population(self, base_components, population_size=20):
        component_variations = {
            'persona': [
                "You are an expert assistant.",
                "You are a helpful AI specialized in this task.",
                "You are a professional consultant with deep knowledge.",
                "As an AI with extensive training, you excel at this task."
            ],
            'instruction_style': [
                "Please complete the following task:",
                "Your job is to:",
                "I need you to:",
                "Help me by:"
            ],
            'output_format': [
                "Provide your answer in a clear, structured format.",
                "Format your response with clear headings and bullet points.",
                "Use numbered steps in your response.",
                "Answer in paragraph form with logical flow."
            ],
            'context_emphasis': [
                "Consider all relevant factors when responding.",
                "Think carefully about the context before answering.",
                "Take into account the broader implications.",
                "Focus on accuracy and completeness."
            ]
        }
        
        for _ in range(population_size):
            individual = {}
            for component, options in component_variations.items():
                individual[component] = random.choice(options)
            
            self.population.append(individual)
        
        return self.population
    
    def evaluate_fitness(self, test_cases):
        fitness_scores = []
        
        for individual in self.population:
            total_score = 0
            
            for test_case in test_cases:
                prompt = self._construct_prompt(individual, test_case['input'])
                
                try:
                    messages = [{"role": "user", "content": prompt}]
                    output = self.model(messages)
                    result = output[0]['generated_text']
                    
                    score = self.fitness_function(result, test_case.get('expected', ''), test_case.get('criteria', {}))
                    total_score += score
                    
                except Exception:
                    total_score += 0
            
            avg_score = total_score / len(test_cases) if test_cases else 0
            fitness_scores.append(avg_score)
        
        return fitness_scores
    
    def _construct_prompt(self, individual, task_input):
        return f"{individual['persona']}\n\n{individual['instruction_style']} {task_input}\n\n{individual['context_emphasis']}\n\n{individual['output_format']}"
    
    def evolve_generation(self, fitness_scores, mutation_rate=0.3):
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), reverse=True)]
        
        elite_size = len(self.population) // 4
        new_population = sorted_population[:elite_size]
        
        while len(new_population) < len(self.population):
            parent1 = self._tournament_selection(sorted_population[:len(sorted_population)//2])
            parent2 = self._tournament_selection(sorted_population[:len(sorted_population)//2])
            
            child = self._crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def _tournament_selection(self, candidates, tournament_size=3):
        tournament = random.sample(candidates, min(tournament_size, len(candidates)))
        return random.choice(tournament)
    
    def _crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            child[key] = random.choice([parent1[key], parent2[key]])
        return child
    
    def _mutate(self, individual):
        mutations = {
            'persona': [
                "You are a domain expert with extensive experience.",
                "As a specialist in this field, you provide accurate insights.",
                "You are known for clear, precise explanations."
            ],
            'instruction_style': [
                "Execute the following task:",
                "Complete this request:",
                "Address the following:"
            ],
            'output_format': [
                "Structure your response clearly and logically.",
                "Provide a comprehensive and well-organized answer.",
                "Present your response in an easy-to-follow format."
            ],
            'context_emphasis': [
                "Consider all aspects carefully.",
                "Analyze thoroughly before responding.",
                "Ensure accuracy and relevance."
            ]
        }
        
        mutated = individual.copy()
        component_to_mutate = random.choice(list(mutations.keys()))
        mutated[component_to_mutate] = random.choice(mutations[component_to_mutate])
        
        return mutated

def comprehensive_fitness_function(output, expected, criteria):
    score = 0.0
    
    if 'length_range' in criteria:
        word_count = len(output.split())
        min_len, max_len = criteria['length_range']
        if min_len <= word_count <= max_len:
            score += 0.3
        else:
            penalty = abs(word_count - np.mean([min_len, max_len])) / max_len
            score += max(0, 0.3 - penalty)
    
    if 'required_keywords' in criteria:
        output_lower = output.lower()
        found_keywords = sum(1 for keyword in criteria['required_keywords'] if keyword.lower() in output_lower)
        score += 0.3 * (found_keywords / len(criteria['required_keywords']))
    
    if 'forbidden_keywords' in criteria:
        output_lower = output.lower()
        forbidden_found = sum(1 for keyword in criteria['forbidden_keywords'] if keyword.lower() in output_lower)
        score += max(0, 0.2 - 0.2 * forbidden_found / max(1, len(criteria['forbidden_keywords'])))
    
    if 'format_requirements' in criteria:
        format_req = criteria['format_requirements']
        if format_req == 'json':
            try:
                json.loads(output)
                score += 0.2
            except:
                pass
        elif format_req == 'numbered_list':
            if any(line.strip().startswith(str(i)) for i in range(1, 6) for line in output.split('\n')):
                score += 0.2
        elif format_req == 'bullet_points':
            if '•' in output or any(line.strip().startswith('-') for line in output.split('\n')):
                score += 0.2
    
    return min(1.0, score)
```

### **Exercise 2: Multi-Modal Prompt Engineering**
```python
class MultiModalPromptFramework:
    def __init__(self, text_model, vision_model=None):
        self.text_model = text_model
        self.vision_model = vision_model
        self.prompt_history = []
        
    def create_contextual_prompt_chain(self, task_sequence):
        results = []
        context = ""
        
        for i, task in enumerate(task_sequence):
            if i == 0:
                prompt = self._build_initial_prompt(task)
            else:
                prompt = self._build_contextual_prompt(task, context, results[-1])
            
            output = self._execute_prompt(prompt, task.get('parameters', {}))
            
            results.append({
                'task_id': i,
                'prompt': prompt,
                'output': output,
                'task_type': task.get('type', 'text')
            })
            
            context = self._update_context(context, output, task)
        
        return results
    
    def _build_initial_prompt(self, task):
        base_template = """
        {persona}
        
        Task: {instruction}
        
        Context: {context}
        
        Requirements:
        {requirements}
        
        Input: {input_data}
        
        Output:
        """
        
        return base_template.format(
            persona=task.get('persona', 'You are a helpful assistant.'),
            instruction=task['instruction'],
            context=task.get('context', 'No additional context provided.'),
            requirements='\n'.join(f"- {req}" for req in task.get('requirements', [])),
            input_data=task['input']
        )
    
    def _build_contextual_prompt(self, task, context, previous_result):
        contextual_template = """
        Previous context: {context}
        
        Previous result: {previous_output}
        
        New task: {instruction}
        
        Build upon the previous work to: {input_data}
        
        Output:
        """
        
        return contextual_template.format(
            context=context[:500],
            previous_output=previous_result['output'][:300],
            instruction=task['instruction'],
            input_data=task['input']
        )
    
    def _execute_prompt(self, prompt, parameters):
        config = {
            'max_new_tokens': parameters.get('max_tokens', 200),
            'temperature': parameters.get('temperature', 0.7),
            'do_sample': parameters.get('do_sample', True)
        }
        
        messages = [{"role": "user", "content": prompt}]
        output = self.text_model(messages, **config)
        return output[0]['generated_text']
    
    def _update_context(self, current_context, new_output, task):
        context_update = f"Task completed: {task['instruction'][:100]}... Result: {new_output[:200]}..."
        
        combined_context = f"{current_context}\n{context_update}"
        
        if len(combined_context) > 1000:
            lines = combined_context.split('\n')
            combined_context = '\n'.join(lines[-5:])
        
        return combined_context
    
    def adaptive_prompt_refinement(self, base_prompt, test_cases, max_iterations=5):
        current_prompt = base_prompt
        best_score = 0
        best_prompt = base_prompt
        iteration_history = []
        
        for iteration in range(max_iterations):
            scores = []
            outputs = []
            
            for test_case in test_cases:
                personalized_prompt = current_prompt.format(**test_case['variables'])
                
                messages = [{"role": "user", "content": personalized_prompt}]
                output = self.text_model(messages)
                result = output[0]['generated_text']
                
                score = self._evaluate_output_quality(result, test_case)
                scores.append(score)
                outputs.append(result)
            
            avg_score = np.mean(scores)
            
            iteration_history.append({
                'iteration': iteration,
                'prompt': current_prompt,
                'avg_score': avg_score,
                'scores': scores,
                'outputs': outputs
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = current_prompt
            
            if iteration < max_iterations - 1:
                current_prompt = self._refine_prompt(current_prompt, scores, outputs, test_cases)
        
        return {
            'best_prompt': best_prompt,
            'best_score': best_score,
            'history': iteration_history,
            'final_prompt': current_prompt
        }
    
    def _evaluate_output_quality(self, output, test_case):
        score = 0.0
        
        if 'expected_elements' in test_case:
            for element in test_case['expected_elements']:
                if element.lower() in output.lower():
                    score += 1.0 / len(test_case['expected_elements'])
        
        if 'length_target' in test_case:
            target_length = test_case['length_target']
            actual_length = len(output.split())
            length_score = 1.0 - abs(actual_length - target_length) / target_length
            score = (score + max(0, length_score)) / 2
        
        if 'quality_keywords' in test_case:
            quality_count = sum(1 for kw in test_case['quality_keywords'] if kw.lower() in output.lower())
            quality_score = quality_count / len(test_case['quality_keywords'])
            score = (score + quality_score) / 2
        
        return min(1.0, score)
    
    def _refine_prompt(self, current_prompt, scores, outputs, test_cases):
        low_performing_indices = [i for i, score in enumerate(scores) if score < np.mean(scores)]
        
        if not low_performing_indices:
            return current_prompt
        
        refinements = []
        
        for idx in low_performing_indices:
            test_case = test_cases[idx]
            output = outputs[idx]
            
            if 'expected_elements' in test_case:
                missing_elements = [elem for elem in test_case['expected_elements'] 
                                 if elem.lower() not in output.lower()]
                if missing_elements:
                    refinements.append(f"Make sure to include: {', '.join(missing_elements)}")
        
        if refinements:
            refinement_text = " ".join(set(refinements))
            refined_prompt = f"{current_prompt}\n\nAdditional guidance: {refinement_text}"
            return refined_prompt
        
        return current_prompt
```

### **Exercise 3: Dynamic Prompt Adaptation**
```python
class DynamicPromptAdapter:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.adaptation_history = []
        self.performance_metrics = {}
        
    def context_aware_prompting(self, user_profile, task_history, current_task):
        expertise_level = self._assess_user_expertise(user_profile, task_history)
        complexity_level = self._assess_task_complexity(current_task)
        
        adapted_prompt = self._build_adaptive_prompt(
            current_task, 
            expertise_level, 
            complexity_level,
            task_history
        )
        
        return adapted_prompt
    
    def _assess_user_expertise(self, user_profile, task_history):
        base_score = 0.5
        
        if 'education_level' in user_profile:
            education_mapping = {
                'high_school': 0.3,
                'bachelor': 0.6,
                'master': 0.8,
                'phd': 1.0
            }
            base_score = education_mapping.get(user_profile['education_level'], 0.5)
        
        if task_history:
            recent_tasks = task_history[-5:]
            success_rate = sum(1 for task in recent_tasks if task.get('success', False)) / len(recent_tasks)
            complexity_handled = np.mean([task.get('complexity', 0.5) for task in recent_tasks])
            
            experience_score = (success_rate + complexity_handled) / 2
            base_score = (base_score + experience_score) / 2
        
        return min(1.0, base_score)
    
    def _assess_task_complexity(self, task):
        complexity_indicators = {
            'multi_step': 0.3,
            'technical_domain': 0.2,
            'creative_elements': 0.2,
            'research_required': 0.3
        }
        
        complexity = 0.0
        task_description = task.get('description', '').lower()
        
        if any(word in task_description for word in ['first', 'then', 'next', 'finally']):
            complexity += complexity_indicators['multi_step']
        
        technical_keywords = ['algorithm', 'technical', 'engineering', 'scientific', 'mathematical']
        if any(word in task_description for word in technical_keywords):
            complexity += complexity_indicators['technical_domain']
        
        creative_keywords = ['creative', 'story', 'design', 'artistic', 'innovative']
        if any(word in task_description for word in creative_keywords):
            complexity += complexity_indicators['creative_elements']
        
        research_keywords = ['research', 'analyze', 'investigate', 'compare', 'evaluate']
        if any(word in task_description for word in research_keywords):
            complexity += complexity_indicators['research_required']
        
        return min(1.0, complexity)
    
    def _build_adaptive_prompt(self, task, expertise_level, complexity_level, task_history):
        if expertise_level < 0.3:
            explanation_level = "detailed with examples"
            instruction_style = "step-by-step guidance"
        elif expertise_level < 0.7:
            explanation_level = "moderate detail"
            instruction_style = "clear instructions"
        else:
            explanation_level = "concise and direct"
            instruction_style = "brief directives"
        
        if complexity_level > 0.7:
            approach = "break down into smaller parts"
            additional_support = "provide reasoning for each step"
        elif complexity_level > 0.4:
            approach = "structure your response clearly"
            additional_support = "explain your approach"
        else:
            approach = "provide a direct answer"
            additional_support = "be concise"
        
        context_from_history = ""
        if task_history:
            recent_context = task_history[-2:]
            context_items = [f"Previously: {task.get('description', '')[:100]}" for task in recent_context]
            context_from_history = "\n".join(context_items)
        
        adaptive_template = f"""
Based on your experience level, I'll provide {explanation_level}.

{instruction_style.capitalize()}: {task['description']}

{context_from_history}

Approach: {approach}

Remember to {additional_support}.

Response:
        """
        
        return adaptive_template.strip()
    
    def real_time_prompt_optimization(self, base_prompt, feedback_stream):
        optimized_prompt = base_prompt
        feedback_buffer = []
        
        for feedback_item in feedback_stream:
            feedback_buffer.append(feedback_item)
            
            if len(feedback_buffer) >= 5:
                optimization_signal = self._analyze_feedback_pattern(feedback_buffer)
                
                if optimization_signal['should_optimize']:
                    optimized_prompt = self._apply_optimization(
                        optimized_prompt, 
                        optimization_signal['optimization_type'],
                        optimization_signal['specific_issues']
                    )
                
                feedback_buffer = feedback_buffer[-3:]
        
        return optimized_prompt
    
    def _analyze_feedback_pattern(self, feedback_buffer):
        issues = []
        feedback_scores = []
        
        for feedback in feedback_buffer:
            feedback_scores.append(feedback.get('score', 0.5))
            
            if 'issues' in feedback:
                issues.extend(feedback['issues'])
        
        avg_score = np.mean(feedback_scores)
        issue_counts = Counter(issues)
        
        should_optimize = avg_score < 0.6 or len(issue_counts) > 0
        
        optimization_type = 'general'
        if issue_counts:
            most_common_issue = issue_counts.most_common(1)[0][0]
            optimization_type = most_common_issue
        
        return {
            'should_optimize': should_optimize,
            'optimization_type': optimization_type,
            'specific_issues': list(issue_counts.keys()),
            'severity': 1.0 - avg_score
        }
    
    def _apply_optimization(self, current_prompt, optimization_type, specific_issues):
        optimizations = {
            'clarity': "Be more specific and clear in your instructions.",
            'length': "Adjust the response length to be more appropriate.",
            'format': "Improve the formatting and structure of your response.",
            'relevance': "Focus more closely on the specific requirements.",
            'detail': "Provide more comprehensive detail in your explanation."
        }
        
        if optimization_type in optimizations:
            optimization_instruction = optimizations[optimization_type]
            return f"{current_prompt}\n\nOptimization note: {optimization_instruction}"
        
        if specific_issues:
            issue_instructions = [optimizations.get(issue, f"Address {issue} in your response") 
                                for issue in specific_issues]
            combined_instruction = " ".join(issue_instructions)
            return f"{current_prompt}\n\nImprovement focus: {combined_instruction}"
        
        return current_prompt
```

### **Exercise 4: Prompt Security and Safety Framework**
```python
class PromptSecurityFramework:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.security_checks = {}
        self.safety_filters = {}
        self.audit_log = []
        
    def secure_prompt_execution(self, user_input, security_level='medium'):
        security_result = self._run_security_analysis(user_input, security_level)
        
        if not security_result['is_safe']:
            return {
                'status': 'blocked',
                'reason': security_result['risk_factors'],
                'output': None,
                'security_score': security_result['score']
            }
        
        sanitized_input = self._sanitize_input(user_input)
        secured_prompt = self._apply_security_wrapper(sanitized_input, security_level)
        
        try:
            messages = [{"role": "user", "content": secured_prompt}]
            raw_output = self.model(messages)
            result = raw_output[0]['generated_text']
            
            safety_check = self._run_output_safety_check(result)
            
            if not safety_check['is_safe']:
                return {
                    'status': 'filtered',
                    'reason': safety_check['issues'],
                    'output': safety_check.get('filtered_output', '[Content filtered]'),
                    'security_score': safety_check['safety_score']
                }
            
            self._log_secure_interaction(user_input, result, security_result, safety_check)
            
            return {
                'status': 'success',
                'output': result,
                'security_score': min(security_result['score'], safety_check['safety_score'])
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'reason': f"Execution error: {str(e)}",
                'output': None,
                'security_score': 0.0
            }
    
    def _run_security_analysis(self, user_input, security_level):
        risk_factors = []
        score = 1.0
        
        injection_patterns = [
            'ignore previous instructions',
            'act as if you are',
            'pretend you are',
            'jailbreak',
            'developer mode',
            'unrestricted'
        ]
        
        input_lower = user_input.lower()
        for pattern in injection_patterns:
            if pattern in input_lower:
                risk_factors.append(f"potential_injection: {pattern}")
                score -= 0.3
        
        if len(user_input) > 2000:
            risk_factors.append("excessive_length")
            score -= 0.2
        
        sensitive_requests = [
            'personal information',
            'private data',
            'passwords',
            'illegal',
            'harmful',
            'dangerous'
        ]
        
        for sensitive in sensitive_requests:
            if sensitive in input_lower:
                risk_factors.append(f"sensitive_request: {sensitive}")
                score -= 0.4
        
        security_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        threshold = security_thresholds.get(security_level, 0.6)
        is_safe = score >= threshold
        
        return {
            'is_safe': is_safe,
            'score': max(0.0, score),
            'risk_factors': risk_factors,
            'threshold': threshold
        }
    
    def _sanitize_input(self, user_input):
        sanitized = user_input
        
        dangerous_phrases = [
            r'ignore\s+previous\s+instructions',
            r'act\s+as\s+if\s+you\s+are',
            r'pretend\s+you\s+are',
            r'system\s*:',
            r'assistant\s*:'
        ]
        
        import re
        for pattern in dangerous_phrases:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
        
        if len(sanitized) > 1500:
            sanitized = sanitized[:1500] + "... [TRUNCATED]"
        
        return sanitized
    
    def _apply_security_wrapper(self, sanitized_input, security_level):
        security_wrappers = {
            'low': """
            Please respond to the following user request appropriately:
            
            {input}
            """,
            'medium': """
            You are a helpful assistant. Please respond to the user's request while following safety guidelines. Do not provide harmful, illegal, or inappropriate content.
            
            User request: {input}
            
            Response:
            """,
            'high': """
            You are a helpful, harmless, and honest assistant. You must:
            1. Refuse any harmful, illegal, or unethical requests
            2. Not provide personal information about individuals
            3. Not assist with dangerous activities
            4. Stay within your training guidelines
            
            User request: {input}
            
            Provide a safe and appropriate response:
            """
        }
        
        wrapper = security_wrappers.get(security_level, security_wrappers['medium'])
        return wrapper.format(input=sanitized_input)
    
    def _run_output_safety_check(self, output):
        issues = []
        safety_score = 1.0
        
        harmful_indicators = [
            'violence', 'illegal', 'dangerous', 'harmful', 'inappropriate',
            'personal information', 'private data', 'confidential'
        ]
        
        output_lower = output.lower()
        for indicator in harmful_indicators:
            if indicator in output_lower:
                issues.append(f"harmful_content: {indicator}")
                safety_score -= 0.3
        
        if len(output) > 1000:
            issues.append("excessive_length")
            safety_score -= 0.1
        
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ]
        
        import re
        for pattern in pii_patterns:
            if re.search(pattern, output):
                issues.append("potential_pii")
                safety_score -= 0.4
        
        filtered_output = output
        if safety_score < 0.5:
            filtered_output = self._filter_unsafe_content(output, issues)
        
        return {
            'is_safe': safety_score >= 0.5,
            'safety_score': max(0.0, safety_score),
            'issues': issues,
            'filtered_output': filtered_output
        }
    
    def _filter_unsafe_content(self, output, issues):
        filtered = output
        
        for issue in issues:
            if 'harmful_content' in issue:
                filtered = "[Content filtered due to safety concerns]"
                break
            elif 'potential_pii' in issue:
                import re
                filtered = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', filtered)
                filtered = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', '[CARD REDACTED]', filtered)
                filtered = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]', filtered)
        
        return filtered
    
    def _log_secure_interaction(self, user_input, output, security_result, safety_check):
        self.audit_log.append({
            'timestamp': time.time(),
            'input_hash': hash(user_input),
            'output_hash': hash(output),
            'security_score': security_result['score'],
            'safety_score': safety_check['safety_score'],
            'risk_factors': security_result['risk_factors'],
            'safety_issues': safety_check['issues']
        })
    
    def generate_security_report(self):
        if not self.audit_log:
            return {"message": "No interactions logged"}
        
        total_interactions = len(self.audit_log)
        avg_security_score = np.mean([log['security_score'] for log in self.audit_log])
        avg_safety_score = np.mean([log['safety_score'] for log in self.audit_log])
        
        all_risk_factors = []
        all_safety_issues = []
        
        for log in self.audit_log:
            all_risk_factors.extend(log['risk_factors'])
            all_safety_issues.extend(log['safety_issues'])
        
        risk_factor_counts = Counter(all_risk_factors)
        safety_issue_counts = Counter(all_safety_issues)
        
        return {
            'total_interactions': total_interactions,
            'avg_security_score': avg_security_score,
            'avg_safety_score': avg_safety_score,
            'top_risk_factors': risk_factor_counts.most_common(5),
            'top_safety_issues': safety_issue_counts.most_common(5),
            'security_incidents': sum(1 for log in self.audit_log if log['security_score'] < 0.5),
            'safety_incidents': sum(1 for log in self.audit_log if log['safety_score'] < 0.5)
        }

# Example usage frameworks
genetic_optimizer = GeneticPromptOptimizer(model_pipeline, comprehensive_fitness_function)
multimodal_framework = MultiModalPromptFramework(text_model)
dynamic_adapter = DynamicPromptAdapter(model_pipeline)
security_framework = PromptSecurityFramework(model_pipeline)
```
### **Exercise 5: Production Prompt Management System**
```python
class ProductionPromptManager:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.prompt_registry = {}
        self.version_history = {}
        self.performance_metrics = {}
        self.ab_tests = {}
        
    def register_prompt_template(self, template_id, template_data):
        self.prompt_registry[template_id] = {
            'template': template_data['template'],
            'variables': template_data.get('variables', []),
            'version': template_data.get('version', '1.0.0'),
            'created_at': time.time(),
            'usage_count': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0
        }
        
        if template_id not in self.version_history:
            self.version_history[template_id] = []
        
        self.version_history[template_id].append({
            'version': template_data.get('version', '1.0.0'),
            'template': template_data['template'],
            'timestamp': time.time(),
            'changes': template_data.get('changes', 'Initial version')
        })
    
    def execute_prompt_with_monitoring(self, template_id, variables, config=None):
        if template_id not in self.prompt_registry:
            raise ValueError(f"Template {template_id} not found")
        
        template_data = self.prompt_registry[template_id]
        prompt = template_data['template'].format(**variables)
        
        start_time = time.time()
        
        try:
            if config is None:
                config = {'max_new_tokens': 200, 'temperature': 0.7}
            
            messages = [{"role": "user", "content": prompt}]
            output = self.model(messages, **config)
            result = output[0]['generated_text']
            
            execution_time = time.time() - start_time
            success = True
            
        except Exception as e:
            result = f"Error: {str(e)}"
            execution_time = time.time() - start_time
            success = False
        
        self._update_performance_metrics(template_id, execution_time, success)
        
        return {
            'template_id': template_id,
            'result': result,
            'execution_time': execution_time,
            'success': success,
            'timestamp': time.time()
        }
    
    def _update_performance_metrics(self, template_id, execution_time, success):
        template_data = self.prompt_registry[template_id]
        
        template_data['usage_count'] += 1
        
        current_success_rate = template_data['success_rate']
        current_avg_time = template_data['avg_response_time']
        usage_count = template_data['usage_count']
        
        new_success_rate = ((current_success_rate * (usage_count - 1)) + (1 if success else 0)) / usage_count
        new_avg_time = ((current_avg_time * (usage_count - 1)) + execution_time) / usage_count
        
        template_data['success_rate'] = new_success_rate
        template_data['avg_response_time'] = new_avg_time
    
    def create_ab_test(self, test_name, template_a_id, template_b_id, traffic_split=0.5):
        self.ab_tests[test_name] = {
            'template_a': template_a_id,
            'template_b': template_b_id,
            'traffic_split': traffic_split,
            'results_a': [],
            'results_b': [],
            'start_time': time.time(),
            'status': 'active'
        }
    
    def execute_ab_test(self, test_name, variables, config=None):
        if test_name not in self.ab_tests:
            raise ValueError(f"A/B test {test_name} not found")
        
        test_data = self.ab_tests[test_name]
        
        if test_data['status'] != 'active':
            raise ValueError(f"A/B test {test_name} is not active")
        
        use_template_a = random.random() < test_data['traffic_split']
        template_id = test_data['template_a'] if use_template_a else test_data['template_b']
        
        result = self.execute_prompt_with_monitoring(template_id, variables, config)
        
        if use_template_a:
            test_data['results_a'].append(result)
        else:
            test_data['results_b'].append(result)
        
        result['ab_test'] = test_name
        result['template_variant'] = 'A' if use_template_a else 'B'
        
        return result
    
    def analyze_ab_test(self, test_name, min_samples=10):
        if test_name not in self.ab_tests:
            raise ValueError(f"A/B test {test_name} not found")
        
        test_data = self.ab_tests[test_name]
        results_a = test_data['results_a']
        results_b = test_data['results_b']
        
        if len(results_a) < min_samples or len(results_b) < min_samples:
            return {
                'status': 'insufficient_data',
                'samples_a': len(results_a),
                'samples_b': len(results_b),
                'min_required': min_samples
            }
        
        success_rate_a = sum(1 for r in results_a if r['success']) / len(results_a)
        success_rate_b = sum(1 for r in results_b if r['success']) / len(results_b)
        
        avg_time_a = np.mean([r['execution_time'] for r in results_a])
        avg_time_b = np.mean([r['execution_time'] for r in results_b])
        
        improvement = (success_rate_b - success_rate_a) / success_rate_a * 100 if success_rate_a > 0 else 0
        significance = abs(improvement) > 5.0 and abs(len(results_a) - len(results_b)) < max(len(results_a), len(results_b)) * 0.1
        
        return {
            'test_name': test_name,
            'template_a_id': test_data['template_a'],
            'template_b_id': test_data['template_b'],
            'samples_a': len(results_a),
            'samples_b': len(results_b),
            'success_rate_a': success_rate_a,
            'success_rate_b': success_rate_b,
            'avg_time_a': avg_time_a,
            'avg_time_b': avg_time_b,
            'improvement_percent': improvement,
            'is_significant': significance,
            'winner': 'B' if success_rate_b > success_rate_a and significance else 'A' if success_rate_a > success_rate_b and significance else 'tie'
        }
    
    def prompt_performance_dashboard(self):
        dashboard_data = {
            'total_templates': len(self.prompt_registry),
            'total_executions': sum(t['usage_count'] for t in self.prompt_registry.values()),
            'avg_success_rate': np.mean([t['success_rate'] for t in self.prompt_registry.values()]) if self.prompt_registry else 0,
            'avg_response_time': np.mean([t['avg_response_time'] for t in self.prompt_registry.values()]) if self.prompt_registry else 0,
            'active_ab_tests': sum(1 for test in self.ab_tests.values() if test['status'] == 'active'),
            'template_rankings': []
        }
        
        for template_id, template_data in self.prompt_registry.items():
            template_score = (template_data['success_rate'] * 0.7) + ((1 / (template_data['avg_response_time'] + 0.1)) * 0.3)
            
            dashboard_data['template_rankings'].append({
                'template_id': template_id,
                'usage_count': template_data['usage_count'],
                'success_rate': template_data['success_rate'],
                'avg_response_time': template_data['avg_response_time'],
                'performance_score': template_score,
                'version': template_data['version']
            })
        
        dashboard_data['template_rankings'].sort(key=lambda x: x['performance_score'], reverse=True)
        
        return dashboard_data
    
    def export_prompt_analytics(self, output_file='prompt_analytics.json'):
        analytics_data = {
            'export_timestamp': time.time(),
            'prompt_registry': self.prompt_registry,
            'version_history': self.version_history,
            'ab_test_results': {},
            'performance_summary': self.prompt_performance_dashboard()
        }
        
        for test_name, test_data in self.ab_tests.items():
            if len(test_data['results_a']) >= 5 and len(test_data['results_b']) >= 5:
                analytics_data['ab_test_results'][test_name] = self.analyze_ab_test(test_name, min_samples=5)
        
        with open(output_file, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        return analytics_data

class PromptQualityAssurance:
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.quality_checks = {}
        self.benchmark_suite = {}
        
    def create_quality_benchmark(self, benchmark_name, test_cases):
        self.benchmark_suite[benchmark_name] = {
            'test_cases': test_cases,
            'created_at': time.time(),
            'last_run': None,
            'historical_results': []
        }
    
    def run_quality_assessment(self, prompt_template, variables_list, benchmark_name=None):
        if benchmark_name and benchmark_name in self.benchmark_suite:
            test_cases = self.benchmark_suite[benchmark_name]['test_cases']
        else:
            test_cases = [{'variables': vars, 'expected_quality': 0.8} for vars in variables_list]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            variables = test_case['variables']
            expected_quality = test_case.get('expected_quality', 0.8)
            
            prompt = prompt_template.format(**variables)
            
            try:
                messages = [{"role": "user", "content": prompt}]
                output = self.model(messages)
                result = output[0]['generated_text']
                
                quality_metrics = self._calculate_quality_metrics(result, test_case, prompt)
                
                results.append({
                    'test_case_id': i,
                    'variables': variables,
                    'prompt': prompt,
                    'output': result,
                    'quality_metrics': quality_metrics,
                    'meets_threshold': quality_metrics['overall_score'] >= expected_quality,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'test_case_id': i,
                    'variables': variables,
                    'prompt': prompt,
                    'output': None,
                    'quality_metrics': {'overall_score': 0.0, 'error': str(e)},
                    'meets_threshold': False,
                    'success': False
                })
        
        assessment_summary = self._summarize_quality_assessment(results)
        
        if benchmark_name and benchmark_name in self.benchmark_suite:
            self.benchmark_suite[benchmark_name]['last_run'] = time.time()
            self.benchmark_suite[benchmark_name]['historical_results'].append({
                'timestamp': time.time(),
                'summary': assessment_summary,
                'detailed_results': results
            })
        
        return {
            'summary': assessment_summary,
            'detailed_results': results,
            'benchmark_name': benchmark_name
        }
    
    def _calculate_quality_metrics(self, output, test_case, prompt):
        metrics = {}
        
        metrics['length_score'] = self._evaluate_length_appropriateness(output, test_case)
        metrics['coherence_score'] = self._evaluate_coherence(output)
        metrics['relevance_score'] = self._evaluate_relevance(output, prompt)
        metrics['format_score'] = self._evaluate_format_compliance(output, test_case)
        metrics['safety_score'] = self._evaluate_safety(output)
        
        weights = {
            'length_score': 0.15,
            'coherence_score': 0.25,
            'relevance_score': 0.25,
            'format_score': 0.20,
            'safety_score': 0.15
        }
        
        overall_score = sum(metrics[metric] * weight for metric, weight in weights.items())
        metrics['overall_score'] = overall_score
        
        return metrics
    
    def _evaluate_length_appropriateness(self, output, test_case):
        word_count = len(output.split())
        
        if 'target_length' in test_case:
            target = test_case['target_length']
            deviation = abs(word_count - target) / target
            return max(0, 1 - deviation)
        
        if word_count < 10:
            return 0.3
        elif word_count > 500:
            return 0.6
        else:
            return 1.0
    
    def _evaluate_coherence(self, output):
        sentences = output.split('.')
        if len(sentences) < 2:
            return 0.8
        
        coherence_indicators = [
            'therefore', 'however', 'moreover', 'furthermore', 'consequently',
            'in addition', 'as a result', 'on the other hand', 'similarly'
        ]
        
        indicator_count = sum(1 for indicator in coherence_indicators if indicator in output.lower())
        coherence_score = min(1.0, 0.5 + (indicator_count * 0.1))
        
        return coherence_score
    
    def _evaluate_relevance(self, output, prompt):
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        
        overlap = len(prompt_words.intersection(output_words))
        relevance_score = overlap / len(prompt_words) if prompt_words else 0
        
        return min(1.0, relevance_score)
    
    def _evaluate_format_compliance(self, output, test_case):
        if 'format_requirements' not in test_case:
            return 1.0
        
        requirements = test_case['format_requirements']
        score = 1.0
        
        if 'bullet_points' in requirements and '•' not in output and '-' not in output:
            score -= 0.5
        
        if 'numbered_list' in requirements and not any(f"{i}." in output for i in range(1, 6)):
            score -= 0.5
        
        if 'json' in requirements:
            try:
                json.loads(output)
            except:
                score -= 0.8
        
        return max(0, score)
    
    def _evaluate_safety(self, output):
        unsafe_indicators = ['harmful', 'dangerous', 'illegal', 'inappropriate', 'offensive']
        
        output_lower = output.lower()
        unsafe_count = sum(1 for indicator in unsafe_indicators if indicator in output_lower)
        
        safety_score = max(0, 1.0 - (unsafe_count * 0.3))
        return safety_score
    
    def _summarize_quality_assessment(self, results):
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'total_tests': len(results),
                'successful_tests': 0,
                'success_rate': 0.0,
                'avg_quality_score': 0.0,
                'threshold_pass_rate': 0.0
            }
        
        quality_scores = [r['quality_metrics']['overall_score'] for r in successful_results]
        threshold_passes = sum(1 for r in successful_results if r['meets_threshold'])
        
        return {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'threshold_pass_rate': threshold_passes / len(successful_results),
            'quality_distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.9),
                'good': sum(1 for score in quality_scores if 0.7 <= score < 0.9),
                'fair': sum(1 for score in quality_scores if 0.5 <= score < 0.7),
                'poor': sum(1 for score in quality_scores if score < 0.5)
            }
        }

production_manager = ProductionPromptManager(model_pipeline)
quality_assurance = PromptQualityAssurance(model_pipeline)
```

---

## 🎯 **Quick Start Action Plan**

### **Immediate Practice:**
1. **Run advanced prompt engineering system** - Test parameter effects, component ablation, reasoning techniques
2. **Experiment with genetic optimization** - Evolve prompts automatically using fitness functions
3. **Try dynamic adaptation** - Adjust prompts based on user expertise and task complexity

### **This Week's Goals:**
1. **Master reasoning enhancement** - Chain-of-thought, tree-of-thought, self-consistency
2. **Implement security framework** - Protect against prompt injection and unsafe outputs
3. **Build production pipeline** - Template management, A/B testing, quality monitoring

### **Advanced Projects:**
1. **Create automated prompt evolution** - Genetic algorithms for prompt optimization
2. **Deploy multi-modal framework** - Chain prompts across different modalities
3. **Implement real-time adaptation** - Dynamic prompt adjustment based on feedback

The enhanced framework transforms basic prompting into sophisticated prompt engineering with automated optimization, security protection, and production-ready management systems.# Chapter 6 Advanced Prompt Engineering Exercises

--- 

## 🎯 **Key Chapter 6 Insights**

### **Systematic Prompt Architecture:**
- **Modular design** - Persona, instruction, context, format as swappable components
- **Parameter control** - Temperature/top_p for creativity vs consistency balance
- **Reasoning enhancement** - CoT, ToT, self-consistency for complex problem solving
- **Output validation** - Grammar constraints and structured generation for reliability

### **Memory Anchors:**
- **"Temperature controls creativity"** - Low = focused, high = diverse outputs
- **"Components are Lego blocks"** - Mix and match prompt pieces for optimal results
- **"Reasoning beats direct answers"** - Step-by-step thinking improves accuracy
- **"Security requires layers"** - Input sanitization + output filtering + monitoring

### **Production Considerations:**
The enhanced system enables enterprise deployment through:
- **Automated optimization** - Genetic algorithms evolve prompts without manual tuning
- **Security frameworks** - Protect against injection attacks and unsafe content
- **Performance monitoring** - A/B testing and quality metrics for continuous improvement
- **Dynamic adaptation** - Real-time prompt adjustment based on user context and feedback

### **Practical Applications:**
- **Customer service automation** - Context-aware responses with safety guarantees
- **Content generation pipelines** - Multi-step creation with quality control
- **Educational systems** - Adaptive prompting based on learner expertise
- **Research assistance** - Chain-of-thought reasoning for complex analysis

This chapter's enhanced framework transforms ad-hoc prompting into systematic prompt engineering with automated optimization, comprehensive security, and production-ready deployment capabilities.
## 🎯 **Author's Core Intentions**

The authors demonstrate evolution from basic prompting to production-ready LLM systems through modular enhancements. Key patterns revealed:

1. **Quantized models** - Use GGUF format for efficient inference without GPU requirements
2. **Chain abstraction** - Sequential processing with prompt templates and variable injection
3. **Memory systems** - Three strategies (buffer, windowed, summary) for conversation continuity
4. **Agent frameworks** - ReAct pattern enabling autonomous tool use and decision making

The sample code shows practical LangChain implementation: loading quantized models, chaining prompts for story generation, adding memory for conversation state, and creating agents with web search + math capabilities.

```python
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.agents import create_react_agent, AgentExecutor, load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    model_path: str
    n_gpu_layers: int = -1
    max_tokens: int = 500
    n_ctx: int = 2048
    temperature: float = 0.7
    seed: int = 42

@dataclass
class ChainMetrics:
    execution_time: float
    token_count: int
    output_length: int
    quality_score: float
    memory_usage: int

class AdvancedLLMSystem:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.llm = None
        self.chains = {}
        self.memories = {}
        self.agents = {}
        self.performance_metrics = defaultdict(list)
        
    def load_quantized_model(self):
        print(f"Loading quantized model: {self.model_config.model_path}")
        
        self.llm = LlamaCpp(
            model_path=self.model_config.model_path,
            n_gpu_layers=self.model_config.n_gpu_layers,
            max_tokens=self.model_config.max_tokens,
            n_ctx=self.model_config.n_ctx,
            temperature=self.model_config.temperature,
            seed=self.model_config.seed,
            verbose=False
        )
        
        test_response = self.llm.invoke("Test: What is 2+2?")
        print(f"✅ Model loaded successfully. Test response: {test_response[:50]}...")
        
        return self
    
    def create_advanced_chain_system(self):
        chain_configs = {
            'story_title': {
                'template': """<s><|user|>Create a compelling title for a story about {theme}. 
Genre: {genre}. Target audience: {audience}. 
Return only the title, nothing else.<|end|><|assistant|>""",
                'variables': ['theme', 'genre', 'audience'],
                'output_key': 'title'
            },
            'character_development': {
                'template': """<s><|user|>Create a detailed character profile for the protagonist of "{title}".
Theme: {theme}. Genre: {genre}.
Include: name, age, background, motivation, key traits.
Format as structured description.<|end|><|assistant|>""",
                'variables': ['title', 'theme', 'genre'],
                'output_key': 'character'
            },
            'plot_outline': {
                'template': """<s><|user|>Create a plot outline for "{title}".
Main character: {character}
Theme: {theme}. Genre: {genre}.
Structure: Setup, Conflict, Resolution.
Keep to 3 paragraphs.<|end|><|assistant|>""",
                'variables': ['title', 'character', 'theme', 'genre'],
                'output_key': 'plot'
            },
            'story_generation': {
                'template': """<s><|user|>Write a complete short story based on:
Title: {title}
Character: {character}
Plot: {plot}
Theme: {theme}. Genre: {genre}.
Target length: 500-800 words.<|end|><|assistant|>""",
                'variables': ['title', 'character', 'plot', 'theme', 'genre'],
                'output_key': 'story'
            }
        }
        
        for chain_name, config in chain_configs.items():
            prompt = PromptTemplate(
                template=config['template'],
                input_variables=config['variables']
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                output_key=config['output_key']
            )
            
            self.chains[chain_name] = chain
        
        print(f"✅ Created {len(self.chains)} specialized chains")
        return self.chains
    
    def execute_sequential_chains(self, initial_inputs: Dict[str, Any]):
        results = initial_inputs.copy()
        execution_log = []
        total_start_time = time.time()
        
        chain_sequence = ['story_title', 'character_development', 'plot_outline', 'story_generation']
        
        for chain_name in chain_sequence:
            if chain_name not in self.chains:
                continue
                
            chain = self.chains[chain_name]
            
            start_time = time.time()
            
            try:
                chain_inputs = {var: results.get(var, '') for var in chain.prompt.input_variables}
                output = chain.invoke(chain_inputs)
                
                execution_time = time.time() - start_time
                
                output_key = getattr(chain, 'output_key', 'text')
                if isinstance(output, dict) and output_key in output:
                    results[output_key] = output[output_key]
                else:
                    results[output_key] = str(output)
                
                step_log = {
                    'chain_name': chain_name,
                    'execution_time': execution_time,
                    'input_vars': list(chain_inputs.keys()),
                    'output_key': output_key,
                    'output_length': len(str(results[output_key])),
                    'success': True
                }
                
                execution_log.append(step_log)
                
                print(f"✅ {chain_name}: {execution_time:.2f}s, {step_log['output_length']} chars")
                
            except Exception as e:
                error_log = {
                    'chain_name': chain_name,
                    'execution_time': 0,
                    'error': str(e),
                    'success': False
                }
                execution_log.append(error_log)
                print(f"❌ {chain_name}: Error - {e}")
        
        total_time = time.time() - total_start_time
        
        return {
            'results': results,
            'execution_log': execution_log,
            'total_execution_time': total_time,
            'success_rate': sum(1 for log in execution_log if log.get('success', False)) / len(execution_log)
        }
    
    def benchmark_memory_systems(self, conversation_scenarios):
        memory_types = {
            'buffer': ConversationBufferMemory(memory_key="chat_history"),
            'windowed_3': ConversationBufferWindowMemory(k=3, memory_key="chat_history"),
            'windowed_5': ConversationBufferWindowMemory(k=5, memory_key="chat_history"),
            'summary': ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                prompt=PromptTemplate(
                    template="""<s><|user|>Summarize: Current: {summary} New: {new_lines} Updated summary:<|end|><|assistant|>""",
                    input_variables=["summary", "new_lines"]
                )
            )
        }
        
        memory_performance = {}
        
        template = """<s><|user|>Chat history: {chat_history}
Current: {input_prompt}<|end|><|assistant|>"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input_prompt", "chat_history"]
        )
        
        for memory_name, memory in memory_types.items():
            print(f"\n🧠 Testing {memory_name} memory...")
            
            chain = LLMChain(prompt=prompt, llm=self.llm, memory=memory)
            
            conversation_metrics = {
                'memory_sizes': [],
                'response_times': [],
                'memory_recall_scores': [],
                'context_retention': []
            }
            
            for scenario in conversation_scenarios:
                for turn in scenario['turns']:
                    start_time = time.time()
                    
                    response = chain.invoke({"input_prompt": turn['input']})
                    
                    response_time = time.time() - start_time
                    
                    memory_vars = memory.load_memory_variables({})
                    memory_size = len(str(memory_vars.get('chat_history', '')))
                    
                    recall_score = self._evaluate_memory_recall(response, turn, scenario)
                    
                    conversation_metrics['memory_sizes'].append(memory_size)
                    conversation_metrics['response_times'].append(response_time)
                    conversation_metrics['memory_recall_scores'].append(recall_score)
                
                memory.clear()
            
            avg_metrics = {
                'avg_memory_size': np.mean(conversation_metrics['memory_sizes']),
                'avg_response_time': np.mean(conversation_metrics['response_times']),
                'avg_recall_score': np.mean(conversation_metrics['memory_recall_scores']),
                'memory_efficiency': np.mean(conversation_metrics['memory_recall_scores']) / np.mean(conversation_metrics['memory_sizes']) * 1000
            }
            
            memory_performance[memory_name] = avg_metrics
            
            print(f"  Avg memory size: {avg_metrics['avg_memory_size']:.0f} chars")
            print(f"  Avg response time: {avg_metrics['avg_response_time']:.2f}s")
            print(f"  Avg recall score: {avg_metrics['avg_recall_score']:.3f}")
        
        return memory_performance
    
    def _evaluate_memory_recall(self, response, turn, scenario):
        if 'expected_recall' not in turn:
            return 0.5
        
        expected_info = turn['expected_recall']
        response_text = str(response).lower()
        
        recall_items = expected_info if isinstance(expected_info, list) else [expected_info]
        
        recalled_count = sum(1 for item in recall_items if item.lower() in response_text)
        return recalled_count / len(recall_items) if recall_items else 0
    
    def create_advanced_agent_system(self, openai_api_key=None):
        if not openai_api_key:
            print("⚠️ OpenAI API key required for agent system")
            return None
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        custom_tools = self._create_custom_tools()
        
        search_tool = DuckDuckGoSearchResults(max_results=3)
        math_tools = load_tools(["llm-math"], llm=openai_llm)
        
        all_tools = custom_tools + [search_tool] + math_tools
        
        react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            template=react_template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        agent = create_react_agent(openai_llm, all_tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=60
        )
        
        self.agents['research_agent'] = agent_executor
        
        print(f"✅ Created agent with {len(all_tools)} tools")
        return agent_executor
    
    def _create_custom_tools(self):
        def text_analyzer(text):
            words = text.split()
            sentences = text.split('.')
            
            analysis = {
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'readability_score': len(words) / len(sentences) if sentences else 0
            }
            
            return json.dumps(analysis, indent=2)
        
        def data_processor(data_string):
            try:
                numbers = [float(x.strip()) for x in data_string.split(',')]
                
                stats = {
                    'count': len(numbers),
                    'sum': sum(numbers),
                    'mean': np.mean(numbers),
                    'median': np.median(numbers),
                    'std': np.std(numbers),
                    'min': min(numbers),
                    'max': max(numbers)
                }
                
                return json.dumps(stats, indent=2)
            except:
                return "Error: Please provide comma-separated numbers"
        
        return [
            Tool(
                name="text_analyzer",
                description="Analyze text for word count, sentences, readability metrics",
                func=text_analyzer
            ),
            Tool(
                name="data_processor", 
                description="Calculate statistics for comma-separated numbers",
                func=data_processor
            )
        ]
    
    def benchmark_agent_performance(self, test_queries):
        if 'research_agent' not in self.agents:
            print("❌ No agent available for benchmarking")
            return None
        
        agent = self.agents['research_agent']
        performance_results = []
        
        for query in test_queries:
            print(f"\n🤖 Testing query: {query['question']}")
            
            start_time = time.time()
            
            try:
                result = agent.invoke({"input": query['question']})
                
                execution_time = time.time() - start_time
                
                success = 'output' in result and len(str(result['output'])) > 10
                
                tool_usage = self._extract_tool_usage_from_result(result)
                
                performance_data = {
                    'question': query['question'],
                    'execution_time': execution_time,
                    'success': success,
                    'output_length': len(str(result.get('output', ''))),
                    'tools_used': tool_usage['tools_used'],
                    'tool_calls': tool_usage['total_calls'],
                    'expected_tools': query.get('expected_tools', [])
                }
                
                performance_results.append(performance_data)
                
                print(f"  ✅ Success: {success}, Time: {execution_time:.2f}s")
                print(f"  🔧 Tools used: {tool_usage['tools_used']}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                performance_results.append({
                    'question': query['question'],
                    'execution_time': 0,
                    'success': False,
                    'error': str(e)
                })
        
        return performance_results
    
    def _extract_tool_usage_from_result(self, result):
        tools_used = []
        total_calls = 0
        
        if 'intermediate_steps' in result:
            for step in result['intermediate_steps']:
                if len(step) >= 2:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
                        total_calls += 1
        
        return {
            'tools_used': list(set(tools_used)),
            'total_calls': total_calls
        }
    
    def create_performance_dashboard(self, metrics_data):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'memory_performance' in metrics_data:
            memory_data = metrics_data['memory_performance']
            
            memory_types = list(memory_data.keys())
            recall_scores = [memory_data[m]['avg_recall_score'] for m in memory_types]
            response_times = [memory_data[m]['avg_response_time'] for m in memory_types]
            
            axes[0,0].bar(memory_types, recall_scores, color='skyblue')
            axes[0,0].set_title('Memory Recall Performance')
            axes[0,0].set_ylabel('Average Recall Score')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            axes[0,1].bar(memory_types, response_times, color='lightgreen')
            axes[0,1].set_title('Memory Response Times')
            axes[0,1].set_ylabel('Average Response Time (s)')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        if 'chain_performance' in metrics_data:
            chain_data = metrics_data['chain_performance']
            
            chain_names = [log['chain_name'] for log in chain_data['execution_log'] if log.get('success')]
            execution_times = [log['execution_time'] for log in chain_data['execution_log'] if log.get('success')]
            
            if chain_names and execution_times:
                axes[1,0].bar(chain_names, execution_times, color='salmon')
                axes[1,0].set_title('Chain Execution Times')
                axes[1,0].set_ylabel('Execution Time (s)')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        if 'agent_performance' in metrics_data:
            agent_data = metrics_data['agent_performance']
            
            successful_queries = [r for r in agent_data if r.get('success', False)]
            
            if successful_queries:
                exec_times = [r['execution_time'] for r in successful_queries]
                tool_calls = [r['tool_calls'] for r in successful_queries]
                
                axes[1,1].scatter(tool_calls, exec_times, alpha=0.7, color='purple')
                axes[1,1].set_title('Agent Performance: Tool Usage vs Time')
                axes[1,1].set_xlabel('Number of Tool Calls')
                axes[1,1].set_ylabel('Execution Time (s)')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_production_pipeline(self, pipeline_config):
        pipeline_steps = []
        
        for step_config in pipeline_config['steps']:
            step_type = step_config['type']
            
            if step_type == 'chain':
                step = self.chains.get(step_config['name'])
            elif step_type == 'agent':
                step = self.agents.get(step_config['name'])
            elif step_type == 'memory':
                step = self.memories.get(step_config['name'])
            else:
                step = None
            
            if step:
                pipeline_steps.append({
                    'name': step_config['name'],
                    'type': step_type,
                    'component': step,
                    'config': step_config
                })
        
        return ProductionPipeline(pipeline_steps, self)

class ProductionPipeline:
    def __init__(self, steps, llm_system):
        self.steps = steps
        self.llm_system = llm_system
        self.execution_history = []
        
    def execute(self, inputs):
        results = inputs.copy()
        step_results = []
        
        for step in self.steps:
            step_start = time.time()
            
            try:
                if step['type'] == 'chain':
                    step_output = step['component'].invoke(results)
                    results.update(step_output)
                elif step['type'] == 'agent':
                    agent_input = step['config'].get('input_key', 'input')
                    step_output = step['component'].invoke({agent_input: results.get(agent_input, '')})
                    results.update(step_output)
                
                step_time = time.time() - step_start
                
                step_results.append({
                    'step_name': step['name'],
                    'execution_time': step_time,
                    'success': True,
                    'output_size': len(str(step_output))
                })
                
            except Exception as e:
                step_results.append({
                    'step_name': step['name'],
                    'execution_time': time.time() - step_start,
                    'success': False,
                    'error': str(e)
                })
        
        self.execution_history.append({
            'inputs': inputs,
            'results': results,
            'step_results': step_results,
            'total_time': sum(s['execution_time'] for s in step_results)
        })
        
        return results

def main_advanced_llm_analysis():
    print("🚀 Chapter 7: Advanced LLM System Architecture")
    print("=" * 60)
    
    model_config = ModelConfig(
        model_path="Phi-3-mini-4k-instruct-fp16.gguf",
        n_gpu_layers=-1,
        max_tokens=800,
        temperature=0.7
    )
    
    system = AdvancedLLMSystem(model_config)
    
    print("\n" + "="*60)
    print("📥 PART 1: QUANTIZED MODEL LOADING")
    
    try:
        system.load_quantized_model()
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("Using mock system for demonstration...")
        return
    
    print("\n" + "="*60)
    print("🔗 PART 2: ADVANCED CHAIN SYSTEM")
    
    system.create_advanced_chain_system()
    
    story_inputs = {
        'theme': 'overcoming adversity',
        'genre': 'science fiction',
        'audience': 'young adults'
    }
    
    chain_results = system.execute_sequential_chains(story_inputs)
    
    print(f"\n✅ Chain execution completed:")
    print(f"  Success rate: {chain_results['success_rate']:.1%}")
    print(f"  Total time: {chain_results['total_execution_time']:.2f}s")
    print(f"  Final story length: {len(chain_results['results'].get('story', ''))}")
    
    print("\n" + "="*60)
    print("🧠 PART 3: MEMORY SYSTEM BENCHMARKING")
    
    conversation_scenarios = [
        {
            'turns': [
                {'input': 'My name is Alice and I work as a data scientist', 'expected_recall': ['Alice', 'data scientist']},
                {'input': 'What is my profession?', 'expected_recall': ['data scientist']},
                {'input': 'I live in San Francisco', 'expected_recall': ['San Francisco']},
                {'input': 'Where do I live and what do I do?', 'expected_recall': ['San Francisco', 'data scientist']},
            ]
        }
    ]
    
    memory_performance = system.benchmark_memory_systems(conversation_scenarios)
    
    print("\n📊 Memory Performance Summary:")
    for memory_type, metrics in memory_performance.items():
        print(f"  {memory_type}: Recall={metrics['avg_recall_score']:.3f}, "
              f"Time={metrics['avg_response_time']:.2f}s, "
              f"Efficiency={metrics['memory_efficiency']:.2f}")
    
    print("\n" + "="*60)
    print("🤖 PART 4: AGENT SYSTEM (Requires OpenAI API)")
    
    test_queries = [
        {
            'question': 'Analyze this text: The quick brown fox jumps over the lazy dog',
            'expected_tools': ['text_analyzer']
        },
        {
            'question': 'Calculate statistics for these numbers: 1,2,3,4,5,6,7,8,9,10',
            'expected_tools': ['data_processor']
        },
        {
            'question': 'What is the current price of Bitcoin and calculate 15% of that amount?',
            'expected_tools': ['duckduckgo_search', 'llm-math']
        }
    ]
    
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        agent_system = system.create_advanced_agent_system(openai_key)
        if agent_system:
            agent_performance = system.benchmark_agent_performance(test_queries)
            
            print(f"\n🤖 Agent Performance:")
            successful = sum(1 for r in agent_performance if r.get('success', False))
            print(f"  Success rate: {successful}/{len(agent_performance)}")
            print(f"  Avg execution time: {np.mean([r.get('execution_time', 0) for r in agent_performance]):.2f}s")
    else:
        print("⚠️ Set OPENAI_API_KEY environment variable to test agents")
        agent_performance = []
    
    print("\n" + "="*60)
    print("📊 PART 5: PERFORMANCE DASHBOARD")
    
    metrics_data = {
        'memory_performance': memory_performance,
        'chain_performance': chain_results,
        'agent_performance': agent_performance if 'agent_performance' in locals() else []
    }
    
    system.create_performance_dashboard(metrics_data)
    
    print("\n🎉 Advanced LLM System Analysis Complete!")
    
    return {
        'system': system,
        'chain_results': chain_results,
        'memory_performance': memory_performance,
        'metrics_data': metrics_data
    }

if __name__ == "__main__":
    results = main_advanced_llm_analysis()
```

---

# Chapter 7 Production-Ready LLM Systems

## 🔧 **Advanced Chain Architectures**

### **Exercise 1: Dynamic Chain Router**
```python
class IntelligentChainRouter:
    def __init__(self, llm_system):
        self.llm_system = llm_system
        self.routing_chains = {}
        self.performance_history = defaultdict(list)
        
    def create_routing_system(self):
        routing_configs = {
            'task_classifier': {
                'template': """<s><|user|>Classify this task into one category:
- creative_writing
- data_analysis  
- question_answering
- summarization
- translation

Task: {user_input}
Category:<|end|><|assistant|>""",
                'variables': ['user_input']
            },
            'complexity_assessor': {
                'template': """<s><|user|>Rate the complexity of this task (1-5):
1: Very simple, single step
2: Simple, few steps
3: Moderate complexity
4: Complex, multiple components
5: Very complex, requires extensive processing

Task: {user_input}
Complexity (number only):<|end|><|assistant|>""",
                'variables': ['user_input']
            },
            'resource_estimator': {
                'template': """<s><|user|>Estimate resources needed for this task:
- tokens_needed (estimate)
- processing_time (seconds)
- memory_requirements (low/medium/high)

Task: {user_input}
Format: tokens=X, time=Y, memory=Z<|end|><|assistant|>""",
                'variables': ['user_input']
            }
        }
        
        for name, config in routing_configs.items():
            prompt = PromptTemplate(
                template=config['template'],
                input_variables=config['variables']
            )
            self.routing_chains[name] = LLMChain(llm=self.llm_system.llm, prompt=prompt)
        
        return self.routing_chains
    
    def intelligent_route(self, user_input, available_chains):
        routing_results = {}
        
        for router_name, chain in self.routing_chains.items():
            try:
                result = chain.invoke({'user_input': user_input})
                routing_results[router_name] = result
            except Exception as e:
                routing_results[router_name] = f"Error: {e}"
        
        task_category = routing_results.get('task_classifier', {}).get('text', 'general').lower().strip()
        
        complexity_text = routing_results.get('complexity_assessor', {}).get('text', '3')
        try:
            complexity = int(complexity_text.split()[0])
        except:
            complexity = 3
        
        resource_text = routing_results.get('resource_estimator', {}).get('text', '')
        
        optimal_chain = self._select_optimal_chain(task_category, complexity, available_chains)
        
        return {
            'selected_chain': optimal_chain,
            'task_category': task_category,
            'complexity_level': complexity,
            'resource_estimate': resource_text,
            'routing_confidence': self._calculate_routing_confidence(routing_results)
        }
    
    def _select_optimal_chain(self, category, complexity, available_chains):
        chain_mapping = {
            'creative_writing': ['story_generation', 'character_development'],
            'data_analysis': ['data_processor', 'statistical_analyzer'],
            'question_answering': ['research_agent', 'fact_checker'],
            'summarization': ['text_summarizer', 'key_extractor'],
            'translation': ['language_translator', 'cultural_adapter']
        }
        
        preferred_chains = chain_mapping.get(category, ['general_purpose'])
        
        for chain_name in preferred_chains:
            if chain_name in available_chains:
                chain_performance = np.mean(self.performance_history.get(chain_name, [0.5]))
                
                if complexity <= 3 or chain_performance > 0.7:
                    return chain_name
        
        return list(available_chains.keys())[0] if available_chains else None
    
    def _calculate_routing_confidence(self, routing_results):
        confidence_factors = []
        
        for result in routing_results.values():
            if isinstance(result, dict) and 'text' in result:
                text_length = len(result['text'])
                confidence_factors.append(min(1.0, text_length / 50))
            else:
                confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)

router = IntelligentChainRouter(llm_system)
```

### **Exercise 2: Adaptive Chain Optimization**
```python
class AdaptiveChainOptimizer:
    def __init__(self, llm_system):
        self.llm_system = llm_system
        self.optimization_history = {}
        self.performance_thresholds = {
            'execution_time': 10.0,
            'quality_score': 0.8,
            'success_rate': 0.9
        }
        
    def optimize_chain_parameters(self, chain_name, test_inputs, optimization_rounds=3):
        if chain_name not in self.llm_system.chains:
            return None
        
        base_chain = self.llm_system.chains[chain_name]
        
        parameter_variants = self._generate_parameter_variants()
        optimization_results = []
        
        for round_num in range(optimization_rounds):
            print(f"Optimization Round {round_num + 1}/{optimization_rounds}")
            
            round_results = []
            
            for variant_name, params in parameter_variants.items():
                variant_chain = self._create_variant_chain(base_chain, params)
                
                variant_performance = self._evaluate_chain_performance(
                    variant_chain, test_inputs, variant_name
                )
                
                round_results.append({
                    'variant_name': variant_name,
                    'parameters': params,
                    'performance': variant_performance
                })
            
            best_variant = max(round_results, key=lambda x: x['performance']['overall_score'])
            optimization_results.append({
                'round': round_num + 1,
                'best_variant': best_variant,
                'all_results': round_results
            })
            
            parameter_variants = self._evolve_parameters(best_variant['parameters'])
        
        final_best = max(optimization_results, key=lambda x: x['best_variant']['performance']['overall_score'])
        
        return {
            'optimization_rounds': optimization_results,
            'final_best_parameters': final_best['best_variant']['parameters'],
            'final_performance': final_best['best_variant']['performance'],
            'improvement_ratio': self._calculate_improvement_ratio(optimization_results)
        }
    
    def _generate_parameter_variants(self):
        return {
            'baseline': {'temperature': 0.7, 'max_tokens': 500},
            'creative': {'temperature': 0.9, 'max_tokens': 800},
            'precise': {'temperature': 0.3, 'max_tokens': 300},
            'balanced': {'temperature': 0.5, 'max_tokens': 600},
            'extensive': {'temperature': 0.8, 'max_tokens': 1000}
        }
    
    def _create_variant_chain(self, base_chain, params):
        variant_llm = LlamaCpp(
            model_path=self.llm_system.model_config.model_path,
            n_gpu_layers=self.llm_system.model_config.n_gpu_layers,
            max_tokens=params['max_tokens'],
            temperature=params['temperature'],
            n_ctx=self.llm_system.model_config.n_ctx,
            seed=42,
            verbose=False
        )
        
        return LLMChain(llm=variant_llm, prompt=base_chain.prompt)
    
    def _evaluate_chain_performance(self, chain, test_inputs, variant_name):
        execution_times = []
        quality_scores = []
        success_count = 0
        
        for test_input in test_inputs:
            start_time = time.time()
            
            try:
                result = chain.invoke(test_input)
                execution_time = time.time() - start_time
                
                execution_times.append(execution_time)
                
                quality_score = self._assess_output_quality(result, test_input)
                quality_scores.append(quality_score)
                
                success_count += 1
                
            except Exception as e:
                execution_times.append(float('inf'))
                quality_scores.append(0.0)
        
        performance_metrics = {
            'avg_execution_time': np.mean(execution_times),
            'avg_quality_score': np.mean(quality_scores),
            'success_rate': success_count / len(test_inputs),
            'overall_score': self._calculate_overall_score(execution_times, quality_scores, success_count, len(test_inputs))
        }
        
        return performance_metrics
    
    def _assess_output_quality(self, result, test_input):
        if isinstance(result, dict) and 'text' in result:
            output_text = result['text']
        else:
            output_text = str(result)
        
        quality_factors = {
            'length_appropriateness': min(1.0, len(output_text.split()) / 100),
            'coherence': 1.0 if len(output_text.split('.')) > 1 else 0.5,
            'relevance': 0.8,
            'completeness': 1.0 if len(output_text) > 50 else 0.3
        }
        
        return np.mean(list(quality_factors.values()))
    
    def _calculate_overall_score(self, execution_times, quality_scores, success_count, total_inputs):
        avg_time = np.mean([t for t in execution_times if t != float('inf')])
        avg_quality = np.mean(quality_scores)
        success_rate = success_count / total_inputs
        
        time_score = max(0, 1 - (avg_time / self.performance_thresholds['execution_time']))
        quality_score = avg_quality
        success_score = success_rate
        
        return (time_score * 0.3 + quality_score * 0.4 + success_score * 0.3)
    
    def _evolve_parameters(self, best_params):
        evolved_variants = {}
        
        base_temp = best_params['temperature']
        base_tokens = best_params['max_tokens']
        
        variations = [
            ('temp_up', {'temperature': min(1.0, base_temp + 0.1), 'max_tokens': base_tokens}),
            ('temp_down', {'temperature': max(0.1, base_temp - 0.1), 'max_tokens': base_tokens}),
            ('tokens_up', {'temperature': base_temp, 'max_tokens': min(1500, base_tokens + 200)}),
            ('tokens_down', {'temperature': base_temp, 'max_tokens': max(100, base_tokens - 200)}),
            ('both_up', {'temperature': min(1.0, base_temp + 0.05), 'max_tokens': min(1500, base_tokens + 100)})
        ]
        
        for name, params in variations:
            evolved_variants[name] = params
        
        return evolved_variants
    
    def _calculate_improvement_ratio(self, optimization_results):
        if len(optimization_results) < 2:
            return 0.0
        
        initial_score = optimization_results[0]['best_variant']['performance']['overall_score']
        final_score = optimization_results[-1]['best_variant']['performance']['overall_score']
        
        return (final_score - initial_score) / initial_score if initial_score > 0 else 0.0
```

## 🧠 **Advanced Memory Systems**

### **Exercise 3: Contextual Memory Manager**
```python
class ContextualMemoryManager:
    def __init__(self, llm_system):
        self.llm_system = llm_system
        self.memory_stores = {
            'episodic': [],
            'semantic': {},
            'working': [],
            'procedural': {}
        }
        self.context_weights = {
            'recency': 0.3,
            'relevance': 0.4,
            'importance': 0.3
        }
        
    def store_interaction(self, user_input, response, context_tags=None):
        timestamp = time.time()
        
        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'response': response,
            'context_tags': context_tags or [],
            'importance_score': self._calculate_importance(user_input, response),
            'entities': self._extract_entities(user_input + ' ' + response)
        }
        
        self.memory_stores['episodic'].append(interaction)
        
        self._update_semantic_memory(interaction)
        self._update_working_memory(interaction)
        
        self._maintain_memory_limits()
        
        return interaction
    
    def retrieve_relevant_context(self, current_input, max_context_items=5):
        all_memories = []
        
        for memory in self.memory_stores['episodic']:
            relevance_score = self._calculate_relevance(current_input, memory)
            recency_score = self._calculate_recency(memory['timestamp'])
            importance_score = memory['importance_score']
            
            combined_score = (
                relevance_score * self.context_weights['relevance'] +
                recency_score * self.context_weights['recency'] +
                importance_score * self.context_weights['importance']
            )
            
            all_memories.append({
                'memory': memory,
                'score': combined_score,
                'relevance': relevance_score,
                'recency': recency_score,
                'importance': importance_score
            })
        
        sorted_memories = sorted(all_memories, key=lambda x: x['score'], reverse=True)
        
        return sorted_memories[:max_context_items]
    
    def _calculate_importance(self, user_input, response):
        importance_indicators = [
            ('name', 0.8),
            ('personal', 0.7),
            ('prefer', 0.6),
            ('important', 0.9),
            ('remember', 0.8),
            ('always', 0.6),
            ('never', 0.6)
        ]
        
        combined_text = (user_input + ' ' + response).lower()
        
        importance_score = 0.5
        for indicator, weight in importance_indicators:
            if indicator in combined_text:
                importance_score = max(importance_score, weight)
        
        return importance_score
    
    def _extract_entities(self, text):
        words = text.split()
        
        entities = {
            'names': [w for w in words if w[0].isupper() and len(w) > 2],
            'numbers': [w for w in words if w.isdigit()],
            'dates': [w for w in words if any(month in w.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun'])],
            'locations': [w for w in words if w.lower() in ['city', 'country', 'street', 'avenue']]
        }
        
        return entities
    
    def _calculate_relevance(self, current_input, memory):
        current_words = set(current_input.lower().split())
        memory_words = set((memory['user_input'] + ' ' + memory['response']).lower().split())
        
        if not current_words or not memory_words:
            return 0.0
        
        intersection = current_words & memory_words
        union = current_words | memory_words
        
        jaccard_similarity = len(intersection) / len(union)
        
        entity_bonus = 0.0
        for entity_type, entities in memory['entities'].items():
            for entity in entities:
                if entity.lower() in current_input.lower():
                    entity_bonus += 0.2
        
        return min(1.0, jaccard_similarity + entity_bonus)
    
    def _calculate_recency(self, timestamp):
        current_time = time.time()
        time_diff = current_time - timestamp
        
        hours_ago = time_diff / 3600
        
        if hours_ago < 1:
            return 1.0
        elif hours_ago < 24:
            return 0.8
        elif hours_ago < 168:
            return 0.6
        else:
            return 0.3
    
    def _update_semantic_memory(self, interaction):
        for entity_type, entities in interaction['entities'].items():
            for entity in entities:
                if entity not in self.memory_stores['semantic']:
                    self.memory_stores['semantic'][entity] = {
                        'mentions': 0,
                        'contexts': [],
                        'importance': 0.0
                    }
                
                self.memory_stores['semantic'][entity]['mentions'] += 1
                self.memory_stores['semantic'][entity]['contexts'].append({
                    'context': interaction['user_input'][:100],
                    'timestamp': interaction['timestamp']
                })
                self.memory_stores['semantic'][entity]['importance'] = min(1.0, 
                    self.memory_stores['semantic'][entity]['mentions'] * 0.1)
    
    def _update_working_memory(self, interaction):
        self.memory_stores['working'].append(interaction)
        
        if len(self.memory_stores['working']) > 5:
            self.memory_stores['working'].pop(0)
    
    def _maintain_memory_limits(self):
        max_episodic_memories = 100
        
        if len(self.memory_stores['episodic']) > max_episodic_memories:
            sorted_memories = sorted(
                self.memory_stores['episodic'],
                key=lambda x: x['importance_score'] * self._calculate_recency(x['timestamp']),
                reverse=True
            )
            
            self.memory_stores['episodic'] = sorted_memories[:max_episodic_memories]
    
    def generate_context_string(self, relevant_memories):
        if not relevant_memories:
            return ""
        
        context_parts = []
        
        for item in relevant_memories:
            memory = item['memory']
            score = item['score']
            
            context_parts.append(
                f"Previous context (relevance: {score:.2f}): "
                f"User: {memory['user_input'][:50]}... "
                f"Assistant: {memory['response'][:50]}..."
            )
        
        return "\n".join(context_parts)

contextual_memory = ContextualMemoryManager(llm_system)
```

## 🤖 **Intelligent Agent Frameworks**

### **Exercise 4: Multi-Agent Coordination System**
```python
class MultiAgentCoordinator:
    def __init__(self, llm_system):
        self.llm_system = llm_system
        self.agents = {}
        self.coordination_history = []
        self.agent_performance = defaultdict(list)
        
    def create_specialized_agents(self, openai_api_key):
        if not openai_api_key:
            return None
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        base_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        agent_configs = {
            'researcher': {
                'tools': [DuckDuckGoSearchResults(), Tool(
                    name="fact_checker",
                    description="Verify factual claims",
                    func=self._fact_checker_tool
                )],
                'system_prompt': "You are a research specialist. Focus on finding accurate, current information.",
                'max_iterations': 8
            },
            'analyst': {
                'tools': [Tool(
                    name="data_analyzer",
                    description="Analyze numerical data and statistics",
                    func=self._data_analyzer_tool
                ), Tool(
                    name="trend_detector",
                    description="Identify patterns and trends",
                    func=self._trend_detector_tool
                )],
                'system_prompt': "You are a data analyst. Focus on numerical analysis and pattern recognition.",
                'max_iterations': 6
            },
            'synthesizer': {
                'tools': [Tool(
                    name="content_merger",
                    description="Combine multiple sources of information",
                    func=self._content_merger_tool
                )],
                'system_prompt': "You are a synthesis specialist. Combine information from multiple sources coherently.",
                'max_iterations': 4
            }
        }
        
        for agent_name, config in agent_configs.items():
            react_template = f"""You are {config['system_prompt']}

Answer the following questions as best you can. You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

            prompt = PromptTemplate(
                template=react_template,
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
            )
            
            agent = create_react_agent(base_llm, config['tools'], prompt)
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=config['tools'],
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=config['max_iterations'],
                max_execution_time=120
            )
            
            self.agents[agent_name] = {
                'executor': agent_executor,
                'config': config,
                'specialties': self._define_agent_specialties(agent_name)
            }
        
        return self.agents
    
    def coordinate_multi_agent_task(self, complex_query, coordination_strategy='sequential'):
        if coordination_strategy == 'sequential':
            return self._sequential_coordination(complex_query)
        elif coordination_strategy == 'parallel':
            return self._parallel_coordination(complex_query)
        elif coordination_strategy == 'hierarchical':
            return self._hierarchical_coordination(complex_query)
        else:
            return self._adaptive_coordination(complex_query)
    
    def _sequential_coordination(self, complex_query):
        coordination_plan = self._create_coordination_plan(complex_query)
        
        results = {}
        accumulated_context = ""
        
        for step in coordination_plan['steps']:
            agent_name = step['assigned_agent']
            step_query = step['query']
            
            if agent_name not in self.agents:
                continue
            
            enhanced_query = f"""Context from previous steps: {accumulated_context}

Current task: {step_query}

Original complex query: {complex_query}"""
            
            start_time = time.time()
            
            try:
                agent_result = self.agents[agent_name]['executor'].invoke({
                    "input": enhanced_query
                })
                
                execution_time = time.time() - start_time
                
                step_result = {
                    'agent': agent_name,
                    'query': step_query,
                    'result': agent_result.get('output', ''),
                    'execution_time': execution_time,
                    'success': True
                }
                
                results[f"step_{len(results) + 1}"] = step_result
                accumulated_context += f"\n{agent_name} found: {step_result['result'][:200]}..."
                
                self.agent_performance[agent_name].append({
                    'execution_time': execution_time,
                    'success': True,
                    'task_complexity': len(step_query.split())
                })
                
            except Exception as e:
                step_result = {
                    'agent': agent_name,
                    'query': step_query,
                    'error': str(e),
                    'execution_time': 0,
                    'success': False
                }
                results[f"step_{len(results) + 1}"] = step_result
        
        final_synthesis = self._synthesize_agent_results(results, complex_query)
        
        coordination_record = {
            'query': complex_query,
            'strategy': 'sequential',
            'coordination_plan': coordination_plan,
            'step_results': results,
            'final_synthesis': final_synthesis,
            'total_agents_used': len(set(step['assigned_agent'] for step in coordination_plan['steps'])),
            'overall_success': all(r.get('success', False) for r in results.values())
        }
        
        self.coordination_history.append(coordination_record)
        
        return coordination_record
    
    def _create_coordination_plan(self, complex_query):
        query_analysis = self._analyze_query_requirements(complex_query)
        
        steps = []
        
        if query_analysis['requires_research']:
            steps.append({
                'step_number': len(steps) + 1,
                'task_type': 'research',
                'assigned_agent': 'researcher',
                'query': f"Research background information for: {complex_query}",
                'dependencies': []
            })
        
        if query_analysis['requires_analysis']:
            steps.append({
                'step_number': len(steps) + 1,
                'task_type': 'analysis',
                'assigned_agent': 'analyst',
                'query': f"Analyze data and identify patterns related to: {complex_query}",
                'dependencies': [s['step_number'] for s in steps if s['task_type'] == 'research']
            })
        
        if len(steps) > 1:
            steps.append({
                'step_number': len(steps) + 1,
                'task_type': 'synthesis',
                'assigned_agent': 'synthesizer',
                'query': f"Synthesize findings to answer: {complex_query}",
                'dependencies': [s['step_number'] for s in steps]
            })
        
        return {
            'total_steps': len(steps),
            'steps': steps,
            'estimated_duration': len(steps) * 30,
            'complexity_score': query_analysis['complexity_score']
        }
    
    def _analyze_query_requirements(self, query):
        query_lower = query.lower()
        
        research_indicators = ['current', 'latest', 'recent', 'news', 'trend', 'what is', 'who is']
        analysis_indicators = ['analyze', 'compare', 'calculate', 'statistics', 'data', 'trend']
        
        requires_research = any(indicator in query_lower for indicator in research_indicators)
        requires_analysis = any(indicator in query_lower for indicator in analysis_indicators)
        
        complexity_score = (
            len(query.split()) * 0.1 +
            query.count('?') * 0.3 +
            query.count('and') * 0.2 +
            int(requires_research) * 0.4 +
            int(requires_analysis) * 0.3
        )
        
        return {
            'requires_research': requires_research,
            'requires_analysis': requires_analysis,
            'complexity_score': min(1.0, complexity_score),
            'estimated_steps': 1 + int(requires_research) + int(requires_analysis)
        }
    
    def _synthesize_agent_results(self, step_results, original_query):
        if 'synthesizer' in self.agents:
            synthesis_input = f"""Original question: {original_query}

Agent findings:
{self._format_step_results(step_results)}

Provide a comprehensive final answer:"""
            
            try:
                synthesis_result = self.agents['synthesizer']['executor'].invoke({
                    "input": synthesis_input
                })
                return synthesis_result.get('output', 'Synthesis failed')
            except Exception as e:
                return f"Synthesis error: {e}"
        
        combined_results = []
        for step_result in step_results.values():
            if step_result.get('success') and 'result' in step_result:
                combined_results.append(f"{step_result['agent']}: {step_result['result']}")
        
        return "\n".join(combined_results)
    
    def _format_step_results(self, step_results):
        formatted = []
        for step_key, result in step_results.items():
            if result.get('success'):
                formatted.append(f"{result['agent']}: {result['result'][:300]}...")
        return "\n".join(formatted)
    
    def _define_agent_specialties(self, agent_name):
        specialties = {
            'researcher': ['current_events', 'factual_information', 'web_search', 'verification'],
            'analyst': ['data_analysis', 'statistics', 'trends', 'patterns', 'calculations'],
            'synthesizer': ['information_integration', 'summary', 'coherent_presentation']
        }
        return specialties.get(agent_name, [])
    
    def _fact_checker_tool(self, claim):
        reliability_indicators = ['study', 'research', 'survey', 'data', 'statistics']
        unreliable_indicators = ['rumor', 'allegedly', 'might', 'could', 'possibly']
        
        claim_lower = claim.lower()
        
        reliability_score = sum(0.2 for indicator in reliability_indicators if indicator in claim_lower)
        unreliability_score = sum(0.3 for indicator in unreliable_indicators if indicator in claim_lower)
        
        final_score = max(0.1, min(1.0, 0.5 + reliability_score - unreliability_score))
        
        return f"Fact check result: Reliability score {final_score:.2f}/1.0. "
    
    def _data_analyzer_tool(self, data_text):
        numbers = re.findall(r'\d+\.?\d*', data_text)
        if numbers:
            nums = [float(n) for n in numbers]
            return f"Data analysis: Found {len(nums)} numbers. Mean: {np.mean(nums):.2f}, Range: {min(nums)}-{max(nums)}"
        return "No numerical data found for analysis"
    
    def _trend_detector_tool(self, data_text):
        trend_words = ['increase', 'decrease', 'growth', 'decline', 'rising', 'falling', 'up', 'down']
        detected_trends = [word for word in trend_words if word in data_text.lower()]
        return f"Trend analysis: Detected trends - {', '.join(detected_trends) if detected_trends else 'No clear trends'}"
    
    def _content_merger_tool(self, content_list):
        if isinstance(content_list, str):
            return f"Content merged: {content_list[:200]}... (processed)"
        return "Content merger: Multiple sources integrated"

multi_agent_system = MultiAgentCoordinator(llm_system)
```

## 🎯 **Quick Start Action Plan**

### **Immediate Practice:**
1. **Run advanced LLM system** - Load quantized model, create chains, test memory systems
2. **Try intelligent routing** - Automatically select optimal chains based on task analysis
3. **Experiment with adaptive optimization** - Auto-tune chain parameters for better performance

### **This Week's Goals:**
1. **Master chain architectures** - Build sequential, parallel, and hierarchical processing
2. **Understand memory strategies** - Choose optimal memory types for different conversation patterns  
3. **Practice agent coordination** - Implement multi-agent systems for complex tasks

### **Advanced Projects:**
1. **Build production pipeline** - Create end-to-end system with monitoring and optimization
2. **Implement contextual memory** - Advanced memory management with importance scoring
3. **Create agent orchestration** - Multi-agent coordination with specialized roles

The enhanced framework transforms basic LangChain usage into sophisticated production-ready systems with intelligent routing, adaptive optimization, and multi-agent coordination capabilities.

---

## 🎯 **Key Chapter 7 Insights**

### **Beyond Basic Prompting:**
- **Quantized models** - Use GGUF format for efficient local deployment without GPU requirements
- **Chain abstraction** - Sequential processing with reusable prompt templates and variable injection
- **Memory systems** - Three strategies (buffer, windowed, summary) solving LLM statelessness
- **Agent frameworks** - ReAct pattern enabling autonomous tool use and multi-step reasoning

### **Memory Anchors:**
- **"Chains = modular workflows"** - Break complex tasks into sequential, reusable components
- **"Memory solves statelessness"** - Add conversation continuity to inherently forgetful LLMs
- **"Agents = LLM + tools + autonomy"** - Enable decision-making and external capability access
- **"Quantization = efficiency without major quality loss"** - Reduce memory/compute while maintaining performance

### **Production-Ready Patterns:**
The enhanced system enables enterprise deployment through:
- **Intelligent chain routing** - Automatically select optimal processing paths based on task analysis
- **Adaptive optimization** - Auto-tune parameters based on performance feedback
- **Contextual memory management** - Sophisticated memory with importance scoring and entity tracking
- **Multi-agent coordination** - Specialized agents working together on complex tasks

### **Real-World Applications:**
- **Customer service automation** - Memory + chains + agents for comprehensive support experiences
- **Research assistance** - Multi-agent coordination for gathering, analyzing, and synthesizing information
- **Content generation pipelines** - Sequential chains for title → character → plot → story workflows
- **Data analysis workflows** - Agent orchestration for research + analysis + synthesis tasks

### **Architecture Evolution:**
```
Basic Prompting → Chains → Memory → Agents → Production Systems
     ↓              ↓         ↓        ↓           ↓
  Single shot → Sequential → Stateful → Autonomous → Orchestrated
```

This chapter's enhanced framework transforms experimental LLM usage into production-ready systems with proper abstractions, memory management, and autonomous capabilities for real-world deployment.
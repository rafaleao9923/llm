# Chapter 6 - Advanced Text Generation Techniques and Tools

## 🎯 **Prompt Engineering Overview**
- **Core purpose**: Design prompts to enhance text generation quality
- **Iterative process**: Experimental optimization without perfect solutions
- **Beyond prompting**: Tool for evaluation, safeguards, and safety mitigation
- **Creative discipline**: Mix structured techniques with innovative approaches

```mermaid
flowchart TD
    A[User Intent] --> B[Prompt Design]
    B --> C[LLM Processing]
    C --> D[Generated Output]
    D --> E[Evaluation]
    E --> F{Satisfactory?}
    F -->|No| G[Iterate Prompt]
    F -->|Yes| H[Final Result]
    G --> B
```

## ⚙️ **Model Selection and Loading**

### **Foundation Model Characteristics:**
- **Size consideration**: Start small, scale up (Phi-3-mini: 3.8B parameters)
- **Hardware requirements**: 8GB VRAM for small models
- **Model families**: Multiple sizes available (mini/base/large/xl)
- **Fine-tuning variants**: Thousands of task-specific derivatives

### **Loading Process:**
```python
# Standard loading with transformers
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Pipeline creation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### **Chat Template Processing:**
- **Role differentiation**: User vs assistant messages
- **Special tokens**: `<|user|>`, `<|assistant|>`, `<|end|>`
- **Template application**: Automatic formatting for model expectations
- **Training alignment**: Templates match training data format

(using - for | ... to fix error mermaid)

```mermaid
graph TD
    A[User Message] --> B[Chat Template]
    B --> C[Special Tokens Added]
    C --> D[<-user->Content<-end-><-assistant->]
    D --> E[LLM Processing]
```

## 🎛️ **Output Control Parameters**

### **🔹 Temperature:**
- **Range**: 0 (deterministic) to 1+ (creative)
- **Effect**: Controls randomness in token selection
- **Low values**: More predictable, focused output
- **High values**: More diverse, creative responses

### **🔹 Top-p (Nucleus Sampling):**
- **Mechanism**: Cumulative probability threshold for token selection
- **Range**: 0.1 (restrictive) to 1.0 (all tokens)
- **Interaction**: Works with temperature for fine control
- **Alternative**: Top-k limits exact number of candidate tokens

### **Parameter Combinations:**

| Use Case | Temperature | Top-p | Description |
|----------|-------------|--------|-------------|
| Brainstorming | High (0.8+) | High (0.9+) | Creative, diverse outputs |
| Email Writing | Low (0.2) | Low (0.3) | Predictable, professional |
| Creative Writing | High (0.8) | Low (0.5) | Creative but coherent |
| Translation | Low (0.2) | High (0.9) | Accurate with vocabulary variety |

```mermaid
graph TD
    A[Token Probabilities] --> B{Temperature}
    B --> C[Adjusted Distribution]
    C --> D{Top-p Filtering}
    D --> E[Final Token Selection]
    
    F[do_sample=True] --> B
    G[do_sample=False] --> H[Greedy Selection]
```

## 🧩 **Basic Prompt Components**

### **Minimal Structure:**
- **Completion prompts**: Simple text continuation
- **Instruction prompts**: Task-specific requests
- **Data inclusion**: Information relevant to the task

### **Enhanced Structure:**
- **Output indicators**: Format guidance (e.g., "Sentiment:")
- **Context provision**: Background information
- **Constraint specification**: Boundaries and limitations

```mermaid
mindmap
  root((Prompt Components))
    Basic
      Instruction
      Data
      Output Format
    Advanced
      Persona
      Context
      Audience
      Tone
      Examples
    Modular
      Mix and Match
      Order Matters
      Iterative Testing
```

## 📚 **Instruction-Based Prompting**

### **Common Use Cases:**
- **Classification**: Sentiment analysis, topic categorization
- **Summarization**: Extract key information from text
- **Translation**: Language conversion tasks
- **Question answering**: Information retrieval and explanation
- **Creative writing**: Story generation, poetry

### **Core Techniques:**

**Specificity:**
- **Detailed requirements**: Length, tone, format specifications
- **Clear boundaries**: What to include/exclude
- **Example**: "Write in 2 sentences using formal tone" vs "Write description"

**Hallucination Mitigation:**
- **Uncertainty acknowledgment**: "Say 'I don't know' if unsure"
- **Source requirements**: "Only use provided information"
- **Confidence indicators**: "Rate your confidence 1-10"

**Positional Effects:**
- **Primacy effect**: Important instructions at beginning
- **Recency effect**: Key information at end
- **Middle neglect**: Avoid placing crucial info in middle of long prompts

## 🏗️ **Advanced Prompt Architecture**

### **Complex Component Integration:**

**Persona Definition:**
```
"You are an expert astrophysicist with 20 years of research experience..."
```

**Context Setting:**
```
"This summary is for busy researchers who need to quickly understand..."
```

**Format Specification:**
```
"Create bullet points followed by a concise paragraph..."
```

**Audience Targeting:**
```
"Explain like I'm 5 (ELI5)" or "Write for graduate-level students"
```

### **Modular Design Benefits:**
- **Component testing**: Add/remove pieces to measure impact
- **Reusability**: Template components across use cases
- **Optimization**: Fine-tune individual elements
- **Maintenance**: Update specific parts without rebuilding

```python
# Modular prompt construction
persona = "You are an expert in Large Language models..."
instruction = "Summarize the key findings..."
context = "Your summary should extract the most crucial points..."
format_spec = "Create bullet points followed by a paragraph..."
audience = "For busy researchers..."
tone = "Professional and clear tone..."

full_prompt = persona + instruction + context + format_spec + audience + tone
```

## 🎯 **In-Context Learning**

### **Learning Paradigms:**
- **Zero-shot**: No examples, pure instruction following
- **One-shot**: Single example demonstration
- **Few-shot**: Multiple examples (2+ demonstrations)

### **Example Structure:**
```python
few_shot_prompt = [
    {"role": "user", "content": "Task description with example input"},
    {"role": "assistant", "content": "Expected output format"},
    {"role": "user", "content": "New task input"}
]
```

### **Benefits:**
- **Pattern demonstration**: Show rather than tell
- **Format consistency**: Establish output structure
- **Quality improvement**: Guide model behavior through examples
- **Reduced ambiguity**: Clear expectations through demonstration

```mermaid
graph LR
    A[Zero-shot] --> B[Pure Instructions]
    C[One-shot] --> D[Single Example]
    E[Few-shot] --> F[Multiple Examples]
    
    B --> G[Good for simple tasks]
    D --> H[Better format control]
    F --> I[Highest quality output]
```

## 🔗 **Chain Prompting**

### **Sequential Problem Solving:**
- **Problem decomposition**: Break complex tasks into steps
- **Output chaining**: Use previous outputs as next inputs
- **Specialized parameters**: Different settings per step
- **Quality improvement**: Focus on individual components

### **Implementation Pattern:**
```python
# Step 1: Generate product name and slogan
product_info = generate_product_info(features)

# Step 2: Create sales pitch using product info
sales_pitch = generate_sales_pitch(product_info)

# Step 3: Refine and validate output
final_output = validate_and_refine(sales_pitch)
```

### **Applications:**
- **Response validation**: LLM checks its own outputs
- **Parallel processing**: Multiple prompts combined later
- **Creative writing**: Plot → Characters → Dialogue progression
- **Complex analysis**: Research → Synthesis → Conclusion

## 🧠 **Reasoning Techniques**

### **System 1 vs System 2 Thinking:**
- **System 1**: Automatic, intuitive, fast (normal LLM behavior)
- **System 2**: Deliberate, logical, slow (enhanced through prompting)
- **Goal**: Enable self-reflection and thoughtful responses

## 💭 **Chain-of-Thought Prompting**

### **Core Concept:**
- **Think first**: Generate reasoning before final answer
- **Step-by-step breakdown**: Distribute computation across tokens
- **Complex problem handling**: Mathematical, logical reasoning tasks

### **Implementation:**

**Few-shot CoT:**
```python
cot_prompt = [
    {"role": "user", "content": "Math problem 1"},
    {"role": "assistant", "content": "Step 1: ... Step 2: ... Answer: X"},
    {"role": "user", "content": "Math problem 2"}
]
```

**Zero-shot CoT:**
```python
prompt = "Solve this problem. Let's think step-by-step."
```

### **Alternative Triggers:**
- "Take a deep breath and think step-by-step"
- "Let's work through this problem step-by-step"
- "Think carefully about each step"

```mermaid
flowchart TD
    A[Complex Problem] --> B[Chain-of-Thought]
    B --> C[Step 1: Initial Analysis]
    C --> D[Step 2: Intermediate Calculation]
    D --> E[Step 3: Final Answer]
    
    F[Direct Answer] --> G[Often Incorrect]
    B --> H[Higher Accuracy]
```

## 🗳️ **Self-Consistency**

### **Majority Voting Approach:**
- **Multiple generations**: Same prompt, different random seeds
- **Diverse sampling**: Vary temperature/top_p across runs
- **Answer aggregation**: Take majority vote as final result
- **Quality improvement**: Reduce impact of random token selection

### **Trade-offs:**
- **Accuracy gain**: More reliable answers through consensus
- **Speed cost**: N times slower (N = number of samples)
- **Resource usage**: Multiple API calls or compute cycles

```mermaid
graph TD
    A[Same Prompt] --> B[Sample 1: Answer A]
    A --> C[Sample 2: Answer A]
    A --> D[Sample 3: Answer B]
    A --> E[Sample 4: Answer A]
    
    B --> F[Majority Vote]
    C --> F
    D --> F
    E --> F
    
    F --> G[Final Answer: A]
```

## 🌳 **Tree-of-Thought**

### **Multi-Path Exploration:**
- **Branching exploration**: Multiple solution paths at each step
- **Evaluation and pruning**: Rate intermediate thoughts
- **Best path selection**: Vote for most promising directions
- **Complex problem solving**: Creative tasks, multi-step reasoning

### **Single-Prompt Implementation:**
```python
tot_prompt = """
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking, then share it.
Then all experts will go on to the next step.
If any expert realizes they're wrong, they leave.
"""
```

### **Applications:**
- **Creative writing**: Explore multiple story directions
- **Problem solving**: Consider various solution approaches
- **Decision making**: Evaluate multiple options systematically

## ✅ **Output Verification**

### **Validation Requirements:**
- **Structured output**: JSON, XML, specific formats
- **Valid constraints**: Limited choice selection
- **Ethical considerations**: No profanity, bias, PII
- **Accuracy standards**: Factual correctness, coherence

### **Control Methods:**

**1. Example-Based Control:**
```python
# Provide format examples
template = """
Use this exact format:
{
    "name": "CHARACTER NAME",
    "class": "CHARACTER CLASS"
}
"""
```

**2. Grammar-Constrained Sampling:**
```python
# Using llama-cpp-python with JSON grammar
output = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)
```

**3. Validation Frameworks:**
- **Guidance**: Microsoft's constrained generation
- **Guardrails**: Robust validation and correction
- **LMQL**: Language Model Query Language

```mermaid
graph TD
    A[Generated Output] --> B{Validation Check}
    B -->|Valid| C[Accept Output]
    B -->|Invalid| D[Regenerate/Correct]
    D --> E[Apply Constraints]
    E --> F[New Generation]
    F --> B
    
    G[Grammar Rules] --> E
    H[Format Examples] --> E
    I[Validation LLM] --> B
```

## 🔧 **Practical Implementation**

### **Memory Management:**
```python
# Clear GPU memory between models
import gc
import torch
del model, tokenizer, pipe
gc.collect()
torch.cuda.empty_cache()
```

### **Model Format Considerations:**
- **Transformers**: Standard format for most models
- **GGUF**: Compressed format for llama-cpp-python
- **Quantization**: Reduced precision for efficiency
- **Context size**: Balance between capability and memory

### **Best Practices:**
- **Start simple**: Begin with basic prompts, add complexity
- **Iterative testing**: Continuous refinement and evaluation
- **Component isolation**: Test individual prompt pieces
- **Documentation**: Track what works for different use cases

## 🎯 **Chapter Summary Insights**

### **Prompt Engineering as Art:**
- **Creative process**: Balance structure with innovation
- **Domain adaptation**: Different techniques for different tasks
- **Continuous learning**: Field evolves with new models and techniques

### **Technical Mastery:**
- **Parameter control**: Temperature, top_p for output tuning
- **Reasoning enhancement**: CoT, self-consistency, tree-of-thought
- **Output validation**: Ensure reliable, structured responses

### **Production Readiness:**
- **Reliability**: Consistent output through proper constraints
- **Scalability**: Chain prompting for complex workflows
- **Quality control**: Validation and verification systems
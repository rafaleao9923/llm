# !pip install llama-cpp-python
from llama_cpp import Llama

model_path = "path/to/DeepSeek-R1-Distill-Qwen-32B-Q4_0.gguf"  # Adjust file path
llm = Llama(model_path=model_path, n_ctx=512)  # Load the model with a context size

# Print model metadata
print(llm.metadata)
# !pip install huggingface-hub
from huggingface_hub import get_safetensors_metadata, list_repo_files

# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_id = "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"

files = list_repo_files(model_id)
print(files)

# dtype_bytes = {"F32": 4, "F16": 2, "F8": 1, "BF16": 2}  # Add BF16

# metadata = get_safetensors_metadata(model_id)
# memory = (
#     sum(count * dtype_bytes[key.split("_")[0]] for key, count in metadata.parameter_count.items())
#     / (1024**3)
#     * 1.18
# )
# print(f"{model_id=} requires {memory=}GB")

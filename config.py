"""
Configuration file for JBShield.
"""

# Data paths
path_harmful = "data/harmful.csv"
path_harmless = "data/harmless.csv"
path_harmful_test = "data/harmful_test.csv"
path_harmless_test = "data/harmless_test.csv"
path_harmful_calibration = "data/harmful_calibration.csv"
path_harmless_calibration = "data/harmless_calibration.csv"

# Model paths
model_paths = {
    "mistral": "./models/Mistral-7B-Instruct-v0.2",
    "llama-2": "./models/Llama-2-7b-chat-hf",
    "llama-3": "./models/Meta-Llama-3-8B-Instruct",
    "vicuna-7b": "./models/vicuna-7b-v1.5",
    "vicuna-13b": "./models/vicuna-13b-v1.5",
    "mistral-sorry-bench": "./models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",

    "deepseek-r1": "./models/DeepSeek-R1",
    "qwen-2.5": "./models/Qwen2.5-7B-Instruct"
}

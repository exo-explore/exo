import re
from exo.models import model_cards
from exo.tokenizer.exo_tokenizer import ExoTokenizer
from transformers import AutoTokenizer

test_strings = [
        # Basic text
        "Hello world!",
        "This is a simple test.",
        
        # Numbers and mixed content
        "In 2024, there are 365 days.",
        "The price is $99.99",
        
        # Special characters and punctuation
        "Hello! How are you? I'm doing great...",
        "Email: test@example.com",
        
        # Whitespace handling
        "   Multiple   spaces   here   ",
        "\tTabs and\nnewlines\n",
        
        # Technical content
        "function test() { return true; }",
        "HTTP/1.1 200 OK",
        
        # Contractions and apostrophes
        "I'm can't won't doesn't it's",
        
        # Long sentence
        "This is a very long sentence that contains multiple words and should test the tokenizer's ability to handle longer sequences of text properly.",
        
        # URLs and paths
        "https://www.example.com/path/to/resource",
        "/usr/local/bin/python3",
        
        # Mixed case
        "CamelCase mixedCase UPPERCASE lowercase",
        
        # Common symbols
        "& % # @ ! ? * ( ) [ ] { } < > + = - / \\ | ~ ` ^ ",
        
        # Repetitive patterns
        "ha ha ha ha ha",
        "test test test test",
        # English with punctuation and numbers
        "Hello, world! Testing 123...",
        "This is a test-case with numbers: 42.5%",
        
        # Mixed scripts and special cases
        "Hello",
        "Testing emoji: 👋 🌍 😊",
        "Numbers & symbols: 123 + 456 = 579",
        
        # Special whitespace and formatting
        "Multiple    spaces   test",
        "New\nline\tand\ttabs",
        
        # URLs and technical text
        "https://www.example.com",
        "user@email.com",
        "Python3.9 + TensorFlow2.0",
        
        # Mathematical expressions
        "∑(x²) = 5π + β",
        
        # Long concatenated words
        "SupercalifragilisticexpialidociousAndMore",
    ]

def test_tokenizer(repo_id):
    print(f"--- {repo_id} ---")

    hf_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    exo_tokenizer = ExoTokenizer(repo_id)

    for test_string in test_strings:
        hf_encoded = hf_tokenizer.encode(test_string)
        exo_encoded = exo_tokenizer.encode(test_string)
        try:
            hf_encoded.remove(exo_tokenizer.bos_token_id)
        except ValueError:
            pass

        assert hf_encoded == exo_encoded, f"{test_string} Failed"

ignore = ["TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-70B-R", "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", "mlx-community/DeepSeek-V2.5-MLX-AQ4_1_64", "llava-hf/llava-1.5-7b-hf", "mlx-community/Qwen*", "dummy", "mlx-community/Meta-Llama-3.1-405B-Instruct-8bit", "mlx-community/Mistral-Large-Instruct-2407-4bit"]
ignore_pattern = re.compile(r"^(" + "|".join(model.replace("*", ".*") for model in ignore) + r")")
models = []
for model_id in model_cards:
  for engine_type, repo_id in model_cards[model_id].get("repo", {}).items():
    if not ignore_pattern.match(repo_id):
      models.append(repo_id)
models = list(set(models))

for repo_id in models:
    test_tokenizer(repo_id)

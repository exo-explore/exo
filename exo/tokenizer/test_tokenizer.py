from exo.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer("/Users/alex/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/78f8ab44d7ce58610645a560461c7ff9e4737d32")
print(tokenizer.encode("Hello, world!"))
print(tokenizer.decode([1, 2, 3, 4, 5]))

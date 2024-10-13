import numpy as np
import random
import re

# Expanded Lorem Ipsum words list
words = """
lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud 
exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute 
irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia 
deserunt mollit anim id est laborum curabitur feugiat mauris fermentum praesent 
volutpat pellentesque libero aliquet gravida integer sagittis posuere morbi dictum 
donec tristique ultricies fringilla venenatis accumsan vestibulum vulputate integer 
nonummy augue ultrices lacinia convallis congue dictumst facilisis litora per aptent 
penatibus sociosqu placerat sociosqu himenaeos facilisi erat fortis fortuna adiuvat sic parvis 
magna
""".split()

# Vocabulary dictionary for word to index mapping
vocab = {word: idx for idx, word in enumerate(set(words))}
reverse_vocab = {idx: word for word, idx in vocab.items()}

from lorem_ipsum import DummyTokenizer

if __name__ == "__main__":
    tokenizer = DummyTokenizer()

    np.random.seed(1)
    # Generate Lorem Ipsum text
    lorem_text = tokenizer.generate_lorem_ipsum(100)
    print("Generated Lorem Ipsum:")
    print(lorem_text)

    # Tokenize the generated text
    encoded_tokens = tokenizer.encode(lorem_text)
    print("\nEncoded Tokens (as NumPy array):")
    print(encoded_tokens)

    # Decode the tokens back to text
    decoded_text = tokenizer.decode(encoded_tokens)
    print("\nDecoded Text:")
    print(decoded_text)

    print(len(encoded_tokens))

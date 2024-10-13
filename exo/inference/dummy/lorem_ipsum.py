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
penatibus sociosqu placerat sociosqu himenaeos facilisi erat fortis fortuna adiuvat sic parvis magna
""".split()
# magna is the terminating token

# Vocabulary dictionary for word to index mapping
vocab = {word: idx for idx, word in enumerate(set(words))}
reverse_vocab = {idx: word for word, idx in vocab.items()}

class DummyTokenizer: 

    eos_token_id: int = 105

    def __init__(self):
        pass

    def encode(self, prompt: str) -> np.ndarray:
        return DummyTokenizer.tokenize_to_numpy(prompt)
    
    def decode(self, tokens: np.ndarray) -> str: 
        return DummyTokenizer.detokenize_from_numpy(tokens)
    
    def model(self, incoming_tokens: np.ndarray, num_tokens_to_gen: int) -> np.ndarray:  
        """
        generate next token, 
        TODO: add a latency throttle
        """

        incoming_words = DummyTokenizer.detokenize_from_numpy(incoming_tokens).split(' ')

        if 'magna' in incoming_words:
            return DummyTokenizer.tokenize_to_numpy(' '.join(incoming_words))

        for _ in range(num_tokens_to_gen):
            word = random.choice(words)
            
            # termination 
            if word == "magna":
                incoming_words.append(word)
                break

            incoming_words.append(word)

        return DummyTokenizer.tokenize_to_numpy(' '.join(incoming_words))
    
    @staticmethod
    def generate_lorem_ipsum(word_count: int = 50) -> str:
        """
        Generate Lorem Ipsum text with a specified number of words, stopping if 'magna' is generated.
        """
        generated_words = []

        # loop through generation and break if you hit magna
        for _ in range(word_count):
            word = random.choice(words)
            
            # termination 
            if word == "magna":
                generated_words.append(word)
                break

            generated_words.append(word)

        return ' '.join(generated_words)
    
    @staticmethod
    def tokenize_to_numpy(text: str) -> np.ndarray:
        """Tokenize text into a NumPy array of integer tokens based on a vocabulary."""

        # use regex to get all the text
        tokens = re.findall(r'\b\w+\b', text.lower())

        # go through each word and tokenize
        token_ids = [vocab[word] for word in tokens if word in vocab]

        return np.array(token_ids)

    @staticmethod
    def detokenize_from_numpy(token_array: np.ndarray) -> str:
        """Convert a NumPy array of integer tokens back to a string."""

        # just use a dictionary to convert back
        if len(token_array) == 0: 
            return ''
        
        words = [reverse_vocab[token] for token in np.array(token_array).flatten()]
      
        return ' '.join(words)

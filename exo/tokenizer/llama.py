import os
import json
import re
import tiktoken
from typing import List, Dict, Any, Tuple
from jinja2 import Template
from datetime import datetime
from exo.tokenizer.tokenizer import Tokenizer

class LlamaTokenizer(Tokenizer):
    def __init__(self, model_path: str):
        with open(os.path.join(model_path, 'tokenizer.json'), 'r',  encoding="utf-8") as f:
            tokenizer_data = json.load(f)
            vocab = tokenizer_data["model"]["vocab"]
            self.pattern = tokenizer_data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            self.special_tokens = {token["content"]: int(token["id"]) for token in tokenizer_data["added_tokens"]}

        with open(os.path.join(model_path, 'tokenizer_config.json'), 'r',  encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            self.chat_template = tokenizer_config["chat_template"]
            self.bos_token = tokenizer_config["bos_token"]
            self.eos_token = tokenizer_config["eos_token"]
            # self.add_bos_token = tokenizer_config.get("add_bos_token", False) # or tokenizer_data["post_processor"]["single"][0]["SpecialToken"]["id"]
            # self.add_eos_token = tokenizer_config.get("add_eos_token", False)

        self._bos_token_id, self._eos_token_id = self.get_bos_and_eos_ids()
        
        self.vocab = {bytes(k, "utf-8"): v for k, v in vocab.items()} # convert str keys to bytes

        self.encoding = tiktoken.Encoding(
            name="custom_encoding",
            pat_str=self.pattern,
            mergeable_ranks=self.vocab,
            special_tokens=self.special_tokens
        )
    
    def decode_chars(self, text: str) -> str:
        decoding_map = {'Ġ': ' ', 'ĉ': '\t', 'Ċ': '\n'}
        result = ''
        for char in text:
            result += decoding_map.get(char, char)
        return result

    def encode_chars(self, text: str) -> str:
        encoding_map = {' ': 'Ġ', '\t': 'ĉ', '\n': 'Ċ'}
        result = ''
        for char in text:
            result += encoding_map.get(char, char)
        return result
    
    def decode(self, tokens: List[int]) -> str:
        return self.decode_chars(self.encoding.decode(tokens))

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        allowed_special = set(self.special_tokens.keys()) if allow_special else set()
        preprocessed_text = self.encode_chars(text)
        # if self.add_bos_token:
        #     preprocessed_text = self.bos_token + preprocessed_text
        # if self.add_eos_token:
        #     preprocessed_text = preprocessed_text + self.eos_token
        return self.encoding.encode(
            preprocessed_text,
            allowed_special=allowed_special,
            disallowed_special=set()
        )

    def get_bos_and_eos_ids(self) -> Tuple[int, int]:
        bos_token_id = self.special_tokens.get(self.bos_token, None)
        eos_token_id = self.special_tokens.get(self.eos_token, None)

        return bos_token_id, eos_token_id

    def apply_chat_template(
        self, 
        messages: List[Dict[str, Any]], 
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        if 'strftime_now' not in kwargs:
            kwargs['strftime_now'] = datetime.now().strftime
        
        template = Template(self.chat_template)
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs
        )
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id
    
if __name__ == "__main__":
    test_strings = [
        # English with punctuation and numbers
        "Hello, world! Testing 123...",
        "This is a test-case with numbers: 42.5%",
        
        
        # Mixed scripts and special cases
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
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
    exo_tokenizer = LlamaTokenizer("/Users/sebnico/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/e42dbdf018e0e870064622c8404d807ab1568631")

    # Test each string
    for test_string in test_strings:
        hf_tokens = hf_tokenizer.encode(test_string)
        print(hf_tokens)
        exo_tokens = exo_tokenizer.encode(test_string)
        print(exo_tokens)
        # print(f"{test_string} Success: {hf_tokens == exo_tokens}")
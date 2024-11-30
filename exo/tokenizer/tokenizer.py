import os
import json
import tiktoken
from typing import List, Dict, Any, Tuple
from jinja2 import Template
from datetime import datetime


class Tokenizer:
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

        self.bos_token_id, self.eos_token_id = self.get_bos_and_eos_ids()
        
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
        return self.encoding.encode(
            self.encode_chars(text),
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

if __name__ == "__main__":
    tokenizer = Tokenizer("/Users/sebnico/.cache/huggingface/hub/models--mlx-community--DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx/snapshots/5f60ee33ba169428b5bc249b05b5b99f827d2e5e")
    print(tokenizer.encode("Hello, world!"))
    print(tokenizer.decode(tokenizer.encode("Hello, world!")))
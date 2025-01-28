import os
import json
import tiktoken
from typing import List, Dict, Any, Tuple
from jinja2 import Template
from datetime import datetime
from exo.download.hf.hf_helpers import get_repo_root
from typing import Optional
from pathlib import Path

def get_local_snapshot_dir_sync(repo_id: str, revision: str = "main") -> Optional[Path]:
  refs_dir = get_repo_root(repo_id)/"refs"
  refs_file = refs_dir/revision
  if os.path.exists(refs_file):
    with open(refs_file, 'r') as f:
      commit_hash = f.read().strip()
      snapshot_dir = get_repo_root(repo_id)/"snapshots"/commit_hash
      return snapshot_dir
  return None

# From https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class ExoTokenizer:
    def __init__(self, model_id: str):
        model_path = get_local_snapshot_dir_sync(model_id)
        with open(os.path.join(model_path, 'tokenizer.json'), 'r',  encoding="utf-8") as f:
            tokenizer_data = json.load(f)
            vocab = tokenizer_data["model"]["vocab"]
            self.bytes_to_unicode = bytes_to_unicode()
            self.unicode_to_bytes = {v: k for k, v in self.bytes_to_unicode.items()}
            self.tokenizer_type = "BPE" if tokenizer_data["pre_tokenizer"] else "SPM"
            self.patterns = self._get_patterns(tokenizer_data) if "gemma" not in model_id else [" ?[^ ]+| +"]
            self.special_tokens = {token["content"]: int(token["id"]) for token in tokenizer_data["added_tokens"]}

        with open(os.path.join(model_path, 'tokenizer_config.json'), 'r',  encoding="utf-8") as f:
            tokenizer_config = json.load(f)
            self.chat_template = self._get_chat_template(tokenizer_config)
            self.bos_token = tokenizer_config["bos_token"]
            self.eos_token = tokenizer_config["eos_token"]
        
        self.original_vocab = vocab # used to check if a token is a byte fallback for SPM models
        self.original_decode_vocab = {v: k for k, v in self.original_vocab.items()}
        self.tiktoken_vocab = self._create_vocab(vocab)
        self.tiktoken_decode_vocab = {v: k for k, v in self.tiktoken_vocab.items()}
        self._bos_token_id, self._eos_token_id = self._get_bos_and_eos_ids()

        self.encoding = tiktoken.Encoding(
            name=f"{model_id}_encoding",
            pat_str="|".join(self.patterns),
            mergeable_ranks=self.tiktoken_vocab,
            special_tokens=self.special_tokens
        )

    @property
    def eos_token_id(self):
        return self._eos_token_id
    
    @property
    def bos_token_id(self):
        return self._bos_token_id
    
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
        
    
    def decode(self, tokens: List[int]) -> str:
        decoded = self.encoding.decode(tokens)

        return decoded
    
    def is_byte_fallback(self, token: int) -> bool:
        res = self.original_decode_vocab.get(token, None)
        return res is not None and res.startswith('<0x') and res != '<0x20'

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        allowed_special = set(self.special_tokens.keys()) if allow_special else set()
        text = text.replace('▁', ' ')
        tokens = self.encoding.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=set()
        )

        # TODO: Clean this up. 
        # This is a hack to handle the fact that tiktoken will not be able to handle emojis
        # as a single token for sentencepiece based tokenizers. 
        # We merge the tokens that represent single bytes into a single token.
        if self.tokenizer_type == "SPM":
            fallback_byte_tokens = []
            checked_tokens = []
            final = []
            i = 0
            while i < len(tokens):
                while i < len(tokens) and self.is_byte_fallback(tokens[i]):
                    decoded_token = self.tiktoken_decode_vocab[tokens[i]]
                    fallback_byte_tokens.append(decoded_token)
                    checked_tokens.append(tokens[i])
                    i += 1
                res = self.tiktoken_vocab.get(bytes().join(fallback_byte_tokens), None)
                if res is not None and not self.is_byte_fallback(res):
                    final.append(res)
                    checked_tokens = []

                if i < len(tokens):
                    checked_tokens.append(tokens[i])
                final.extend(checked_tokens)

                checked_tokens = []
                fallback_byte_tokens = []
                i += 1

            return final
        
        return tokens

    def _create_vocab(self, vocab: Dict[str, int]) -> Dict[bytes, int]:
        return {self._normalize_token_bytes(k): v for k, v in vocab.items()}
    
    def _get_bos_and_eos_ids(self) -> Tuple[int, int]:
        bos_token_id = self.special_tokens.get(self.bos_token, None)
        eos_token_id = self.special_tokens.get(self.eos_token, None)

        return bos_token_id, eos_token_id
    
    def _get_patterns(self, tokenizer_data: Dict[str, Any]) -> List[str]:
        patterns = []
        # Gemma and Mistral Large do not have a pre_tokenizer field in the tokenizer.json file, so they
        # will use the default pattern used by the HF tokenizers library, just as they do in the AutoTokenizer. 
        # see https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L44
        # AutoTikTokenizer uses a different pattern with similar behavior but more concise, 
        # see https://github.com/chonkie-ai/autotiktokenizer/blob/main/src/autotiktokenizer/autotiktokenizer.py#L199
        if tokenizer_data.get("pre_tokenizer", None) is None:
            default_BPE_pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            # TODO:Check how this relates to how metaspace splits text with SplitDelimiterBehavior::MergedWithNext case
            # See https://github.com/huggingface/tokenizers/blob/24d29f498d890638279b0d51e899b6020571719d/tokenizers/src/pre_tokenizers/metaspace.rs#L136
            default_SPM_pattern = r" ?[^ ]+| +(?![^ ])"
            default_pattern = default_BPE_pattern if self.tokenizer_type == "BPE" else default_SPM_pattern
            return [default_pattern]
        for pretokenizer in tokenizer_data["pre_tokenizer"]["pretokenizers"]:
            if pretokenizer["type"] == "Split":
                pattern = pretokenizer["pattern"]["Regex"]
                patterns.append(pattern)
            elif pretokenizer["type"] == "Digits":
                if bool(pretokenizer["individual_digits"]):
                    patterns.append(r"\d")
                else:
                    patterns.append(r"\d+")

        return patterns
    

    # From https://github.com/chonkie-ai/autotiktokenizer/blob/main/src/autotiktokenizer/autotiktokenizer.py#L95
    def _normalize_token_bytes(self, token: str) -> bytes:
        """Convert bytes to unicode.
        
        Args:
            token (str): The token to convert.
        
        Returns:
            result (bytes): The converted token.
        """
        # Sentencepiece modles using byte fallbacks will wrap bytes in < and >.
        # For example, <0x0A> will be converted to the actual byte b'\n'.
        if token.startswith("<0x") and token.endswith(">"):
            hex_str = token[1:-1]
            byte_val = int(hex_str, 16)  # Converts byte eg "0x0A" to int 10
            byte_obj = bytes([byte_val])  # Converst int to bytes)
            return byte_obj
        
        if self.tokenizer_type == "SPM":
            token = token.replace("▁", " ")

            return token.encode()
        else:
            try:
                result = bytearray([self.unicode_to_bytes[b] for b in token])
                return bytes(result)
            except Exception:
                return token.encode()
    
    def _get_chat_template(self, tokenizer_config: Dict[str, Any]) -> str:
        default_chat_template = """"""
        return tokenizer_config.get("chat_template", default_chat_template)
    
if __name__ == "__main__":
    from transformers import AutoTokenizer

    test_string = "This is a simple test."

    hf_tokenizer = AutoTokenizer.from_pretrained("mlx-community/gemma-2-9b-it-4bit")
    exo_tokenizer = ExoTokenizer("mlx-community/gemma-2-9b-it-4bit")
    hf_tokens = hf_tokenizer.encode(test_string)
    exo_tokens = exo_tokenizer.encode(test_string)

    print("HF Tokens:")
    for token in hf_tokens:
        print(f"{token} - {hf_tokenizer.decode([token])}")
    print("Exo Tokens:")
    for token in exo_tokens:
        print(f"{token} - {exo_tokenizer.decode([token])}")

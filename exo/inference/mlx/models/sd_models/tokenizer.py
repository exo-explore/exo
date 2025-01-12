# adapted from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/tokenizer.py

import regex
import json
import glob


class Tokenizer:
    """A simple port of CLIPTokenizer from https://github.com/huggingface/transformers/ ."""

    def __init__(self, bpe_ranks, vocab):
        self.bpe_ranks = bpe_ranks
        self.vocab = vocab
        self.pat = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

        self._cache = {self.bos: self.bos, self.eos: self.eos}

    @property
    def bos(self):
        return "<|startoftext|>"

    @property
    def bos_token(self):
        return self.vocab[self.bos]

    @property
    def eos(self):
        return "<|endoftext|>"

    @property
    def eos_token(self):
        return self.vocab[self.eos]

    def bpe(self, text):
        if text in self._cache:
            return self._cache[text]

        unigrams = list(text[:-1]) + [text[-1] + "</w>"]
        unique_bigrams = set(zip(unigrams, unigrams[1:]))

        if not unique_bigrams:
            return unigrams

        # In every iteration try to merge the two most likely bigrams. If none
        # was merged we are done.
        #
        # Ported from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/tokenization_clip.py
        while unique_bigrams:
            bigram = min(
                unique_bigrams, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break

            new_unigrams = []
            skip = False
            for a, b in zip(unigrams, unigrams[1:]):
                if skip:
                    skip = False
                    continue

                if (a, b) == bigram:
                    new_unigrams.append(a + b)
                    skip = True

                else:
                    new_unigrams.append(a)

            if not skip:
                new_unigrams.append(b)

            unigrams = new_unigrams
            unique_bigrams = set(zip(unigrams, unigrams[1:]))

        self._cache[text] = unigrams

        return unigrams

    def tokenize(self, text, prepend_bos=True, append_eos=True):
        if isinstance(text, list):
            return [self.tokenize(t, prepend_bos, append_eos) for t in text]

        # Lower case cleanup and split according to self.pat. Hugging Face does
        # a much more thorough job here but this should suffice for 95% of
        # cases.
        clean_text = regex.sub(r"\s+", " ", text.lower())
        tokens = regex.findall(self.pat, clean_text)

        # Split the tokens according to the byte-pair merge file
        bpe_tokens = [ti for t in tokens for ti in self.bpe(t)]

        # Map to token ids and return
        tokens = [self.vocab[t] for t in bpe_tokens]
        if prepend_bos:
            tokens = [self.bos_token] + tokens
        if append_eos:
            tokens.append(self.eos_token)

        return tokens
    
    def encode(self, prompt):
        tokens = [self.tokenize(prompt)]
        negative_text = ""
        if negative_text is not None:
            tokens += [self.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [0] * (N - len(t)) for t in tokens]
        return tokens

def load_tokenizer(
    model_path: str,
    vocab_key: str = "tokenizer_vocab",
    merges_key: str = "tokenizer_merges",
):

    vocab_file = glob.glob(str(model_path/"tokenizer"/vocab_key))[0]
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = glob.glob(str(model_path/"tokenizer"/merges_key))[0]
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab)


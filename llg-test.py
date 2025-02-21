import argparse
import json
import huggingface_hub
from transformers import AutoTokenizer
import llguidance


def main():
  parser = argparse.ArgumentParser(
    description="Command line interface for LL Guidance."
  )
  parser.add_argument("--tokenizer", help="Tokenizer name")
  parser.add_argument("--lark", help="Lark grammar file")
  parser.add_argument("--json-schema", help="JSON schema file")
  parser.add_argument("--text", help="File containing simulated generated text")
  parser.add_argument("--log-level", help="Log level", default=1, type=int)
  parser.add_argument("--ff-tokens", help="Enable fast-forward tokens", action="store_true")
  parser.add_argument("--backtrack", help="Enable backtracking", action="store_true")
  args = parser.parse_args()
  tokenizer: str = args.tokenizer

  tok_name = huggingface_hub.hf_hub_download(tokenizer, "tokenizer.json")
  with open(tok_name, "r") as f:
    text = f.read()
  tok = llguidance.LLTokenizer(text)

  grm = {}
  if args.lark:
    with open(args.lark, "r") as f:
      grm["lark_grammar"] = f.read()
  if args.json_schema:
    with open(args.json_schema, "r") as f:
      grm["json_schema"] = json.loads(f.read())

  if grm == {}:
    raise ValueError("No grammar provided; need --lark or --json-schema")

  tokens = []

  if args.text:
    with open(args.text, "r") as f:
      text = f.read()
    hf_tok = AutoTokenizer.from_pretrained(tokenizer)
    tokens = hf_tok.encode(text, add_special_tokens=False)

  interp = llguidance.LLInterpreter(
    tok,
    json.dumps({"grammars": [grm]}),
    enable_ff_tokens=args.ff_tokens,
    enable_backtrack=args.backtrack,
    log_level=args.log_level,
  )
  interp.start_without_prompt()

  print(tok.dbg_tokens(tokens))
  for t in tokens:
    mask, r = interp.compute_mask()
    obj = json.loads(r)
    for p in obj["progress"]:
      if p["object"] != "text":
        print("\n  ", end="")
        print(p)
    print(tok.dbg_tokens([t]), end=" ")
    if mask[t] == 0:
      print("Token not in mask")
      break
    interp.commit_token(t)
  print("\n")


if __name__ == "__main__":
  main()

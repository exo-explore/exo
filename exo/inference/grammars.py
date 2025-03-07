import json
import os

json_lark_grammar_path = os.path.join(os.path.dirname(__file__), "grammars/json.lark")

with open(json_lark_grammar_path, "r") as f:
  JSON_LARK_GRAMMAR = f.read()


def json_object_grammar() -> str:
  return lark_grammar(JSON_LARK_GRAMMAR)


def json_schema_grammar(json_schema: dict) -> str:
  return json.dumps({"grammars": [{"json_schema": json_schema}]})


def lark_grammar(grammar: str) -> str:
  return json.dumps({"grammars": [{"lark_grammar": grammar}]})

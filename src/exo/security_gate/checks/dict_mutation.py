"""
Check for dict/set mutation during iteration.

Detects:
  for k in d:         d[k] = v        -> RuntimeError at runtime
  for k in d:         del d[k]        -> RuntimeError at runtime
  for k, v in d:      d[k] = ...      -> RuntimeError at runtime
  for item in s:      s.add(item)     -> RuntimeError at runtime
  for item in s:      s.discard(item) -> RuntimeError at runtime
  for item in s:      s.remove(item)  -> RuntimeError at runtime

Fix: use list(d.items()), list(d.keys()), or list(s) to snapshot before iterating.
"""
from __future__ import annotations

import ast

from exo.security_gate.issue import Issue


def _iterated_names(for_node: ast.For) -> set[str]:
    """Extract the names of iterables being iterated over in a for loop."""
    names: set[str] = set()
    iter_node = for_node.iter

    # for k in d:  →  iter_node = Name("d")
    if isinstance(iter_node, ast.Name):
        names.add(iter_node.id)

    # for k in d.keys():  or  for k, v in d.items():  →  iter_node = Call(func=Attr(value=Name("d")))
    elif isinstance(iter_node, ast.Call):
        func = iter_node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.attr in ("keys", "values", "items"):
            names.add(func.value.id)

    return names


class _MutationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[Issue] = []
        self._filepath: str = ""
        self._iter_stack: list[set[str]] = []

    def set_filepath(self, filepath: str) -> None:
        self._filepath = filepath

    def _check_stmt(self, node: ast.stmt) -> None:
        """Check a single statement inside a for body for mutation of iterated names."""
        if not self._iter_stack:
            return
        current = self._iter_stack[-1]

        # dict mutation: d[k] = v  or  d[k] += v  etc.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in current:
                    self.issues.append(
                        Issue(
                            filepath=self._filepath,
                            lineno=node.lineno,
                            check_id="DICT_MUTATION",
                            message=(
                                f"Dict '{target.value.id}' mutated during iteration — "
                                "use list(d.items()) snapshot to avoid RuntimeError"
                            ),
                            severity="block",
                        )
                    )

        elif isinstance(node, ast.AugAssign):
            target = node.target
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in current:
                self.issues.append(
                    Issue(
                        filepath=self._filepath,
                        lineno=node.lineno,
                        check_id="DICT_MUTATION",
                        message=(
                            f"Dict '{target.value.id}' mutated during iteration — "
                            "use list(d.items()) snapshot to avoid RuntimeError"
                        ),
                        severity="block",
                    )
                )

        # del d[k]
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in current:
                    self.issues.append(
                        Issue(
                            filepath=self._filepath,
                            lineno=node.lineno,
                            check_id="DICT_MUTATION",
                            message=(
                                f"Dict '{target.value.id}' deleted-during-iteration — "
                                "use list(d.items()) snapshot to avoid RuntimeError"
                            ),
                            severity="block",
                        )
                    )

        # set mutation: s.add(), s.discard(), s.remove(), s.pop(), s.clear()
        # dict mutation: d.pop(), d.clear(), d.update()
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in current
                and call.func.attr in ("add", "discard", "remove", "pop", "clear", "update")
            ):
                self.issues.append(
                    Issue(
                        filepath=self._filepath,
                        lineno=node.lineno,
                        check_id="DICT_MUTATION",
                        message=(
                            f"Collection '{call.func.value.id}'.{call.func.attr}() called "
                            "during iteration — use list() snapshot to avoid RuntimeError"
                        ),
                        severity="block",
                    )
                )

    def visit_For(self, node: ast.For) -> None:
        iterated = _iterated_names(node)
        self._iter_stack.append(iterated)
        for stmt in node.body:
            self._check_stmt(stmt)
        # Recurse into body (handles nested loops)
        self.generic_visit(node)
        self._iter_stack.pop()


def check_dict_mutation(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    visitor = _MutationVisitor()
    visitor.set_filepath(filepath)
    visitor.visit(tree)
    return visitor.issues

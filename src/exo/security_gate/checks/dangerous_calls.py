import ast

from exo.security_gate.issue import Issue

_SUBPROCESS_ATTRS = ("run", "call", "Popen", "check_output", "check_call")


def _is_shell_true(keywords: list[ast.keyword]) -> bool:
    for kw in keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


class _DangerousCallVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.issues: list[Issue] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # eval / exec
        if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="DANGEROUS_CALL",
                    message=f"{func.id}() call detected — use ast.literal_eval() or structured parsing",
                    severity="block",
                )
            )

        # __import__
        elif isinstance(func, ast.Name) and func.id == "__import__":
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="DANGEROUS_CALL",
                    message="__import__() call detected — use importlib.import_module() instead",
                    severity="block",
                )
            )

        elif isinstance(func, ast.Attribute):
            attr = func.attr
            value = func.value

            # pickle.loads / pickle.load
            if attr in ("loads", "load") and isinstance(value, ast.Name) and value.id == "pickle":
                self.issues.append(
                    Issue(
                        filepath=self.filepath,
                        lineno=node.lineno,
                        check_id="DANGEROUS_CALL",
                        message=f"pickle.{attr}() detected — pickle deserializes arbitrary code; use a safe format",
                        severity="block",
                    )
                )

            # subprocess shell=True
            elif (
                attr in _SUBPROCESS_ATTRS
                and isinstance(value, ast.Name)
                and value.id == "subprocess"
                and _is_shell_true(node.keywords)
            ):
                self.issues.append(
                    Issue(
                        filepath=self.filepath,
                        lineno=node.lineno,
                        check_id="DANGEROUS_CALL",
                        message=f"subprocess.{attr}(..., shell=True) detected — shell injection risk; pass a list instead",
                        severity="block",
                    )
                )

            # os.system
            elif attr == "system" and isinstance(value, ast.Name) and value.id == "os":
                self.issues.append(
                    Issue(
                        filepath=self.filepath,
                        lineno=node.lineno,
                        check_id="DANGEROUS_CALL",
                        message="os.system() detected — use subprocess.run() with a list of arguments",
                        severity="block",
                    )
                )

        self.generic_visit(node)


def check_dangerous_calls(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    issues: list[Issue] = []

    # Check for pickle imports (advisory)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pickle" or alias.name.startswith("pickle."):
                    issues.append(
                        Issue(
                            filepath=filepath,
                            lineno=node.lineno,
                            check_id="DANGEROUS_CALL",
                            message="pickle import detected — ensure no untrusted data is deserialized",
                            severity="advisory",
                        )
                    )
        elif isinstance(node, ast.ImportFrom) and node.module is not None and (node.module == "pickle" or node.module.startswith("pickle.")):
            issues.append(
                Issue(
                    filepath=filepath,
                    lineno=node.lineno,
                    check_id="DANGEROUS_CALL",
                    message="pickle import detected — ensure no untrusted data is deserialized",
                    severity="advisory",
                )
            )

    visitor = _DangerousCallVisitor(filepath)
    visitor.visit(tree)
    issues.extend(visitor.issues)
    return issues

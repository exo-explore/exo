import ast

from exo.security_gate.issue import Issue

_HTTP_ATTRS = ("get", "post", "put", "delete", "patch", "head")


class _AsyncHazardVisitor(ast.NodeVisitor):
    """Visit async function defs and check for hazards within them.

    Stops descending into nested synchronous function defs (ast.FunctionDef)
    because those have different rules.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.issues: list[Issue] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Walk the body of this async function, but stop at nested sync functions
        _AsyncBodyVisitor(self.filepath, node.name, self.issues).visit_body(node.body)
        # Still recurse to find nested async functions
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.AsyncFunctionDef):
                self.visit_AsyncFunctionDef(child)

    # Do not auto-recurse via generic_visit; we handle traversal manually
    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                self.visit_AsyncFunctionDef(child)
            elif not isinstance(child, ast.FunctionDef):
                self.generic_visit(child)


class _AsyncBodyVisitor(ast.NodeVisitor):
    """Walks the body of a single async function (not descending into nested sync defs)."""

    def __init__(self, filepath: str, func_name: str, issues: list[Issue]) -> None:
        self.filepath = filepath
        self.func_name = func_name
        self.issues = issues

    def visit_body(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Stop — do not descend into nested synchronous functions
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Stop — outer visitor handles nested async functions
        pass

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # time.sleep in async
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "sleep"
            and isinstance(func.value, ast.Name)
            and func.value.id == "time"
        ):
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="ASYNC_HAZARD",
                    message=f"time.sleep() in async function '{self.func_name}' blocks the event loop — use await anyio.sleep()",
                    severity="block",
                )
            )

        # requests.* in async
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in _HTTP_ATTRS
            and isinstance(func.value, ast.Name)
            and func.value.id == "requests"
        ):
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="ASYNC_HAZARD",
                    message=f"requests.{func.attr}() in async function '{self.func_name}' blocks the event loop — use httpx.AsyncClient",
                    severity="block",
                )
            )

        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Bare except:
        if node.type is None:
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="ASYNC_HAZARD",
                    message=f"Bare 'except:' in async function '{self.func_name}' swallows CancelledError — use 'except Exception:'",
                    severity="block",
                )
            )
        # except BaseException:
        elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="ASYNC_HAZARD",
                    message=f"Catching BaseException in async function '{self.func_name}' swallows CancelledError",
                    severity="block",
                )
            )
        self.generic_visit(node)


def check_async_hazards(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    visitor = _AsyncHazardVisitor(filepath)
    visitor.visit(tree)
    return visitor.issues

import ast

from exo.security_gate.issue import Issue

_LOGGING_ATTRS = (
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "log",
)


class _ApplyBodyVisitor(ast.NodeVisitor):
    """Walk body of apply() and flag side effects. Stop at nested function defs."""

    def __init__(self, filepath: str, issues: list[Issue]) -> None:
        self.filepath = filepath
        self.issues = issues

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Stop descent into nested function defs
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # print()
        if isinstance(func, ast.Name) and func.id == "print":
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="EVENT_SOURCING",
                    message="Side effect in apply() — apply() must be pure: (State, Event) → State (print detected)",
                    severity="advisory",
                )
            )

        # open()
        elif isinstance(func, ast.Name) and func.id == "open":
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="EVENT_SOURCING",
                    message="Side effect in apply() — apply() must be pure: (State, Event) → State (open() detected)",
                    severity="advisory",
                )
            )

        # logging.* (e.g. logging.info, logger.info, log.debug, etc.)
        elif isinstance(func, ast.Attribute) and func.attr in _LOGGING_ATTRS:
            self.issues.append(
                Issue(
                    filepath=self.filepath,
                    lineno=node.lineno,
                    check_id="EVENT_SOURCING",
                    message=f"Side effect in apply() — apply() must be pure: (State, Event) → State (logging.{func.attr}() detected)",
                    severity="advisory",
                )
            )

        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self.issues.append(
            Issue(
                filepath=self.filepath,
                lineno=node.lineno,
                check_id="EVENT_SOURCING",
                message="await in apply() — apply() must be synchronous and pure",
                severity="advisory",
            )
        )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.issues.append(
            Issue(
                filepath=self.filepath,
                lineno=node.lineno,
                check_id="EVENT_SOURCING",
                message="Global/nonlocal in apply() — apply() must be pure",
                severity="advisory",
            )
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.issues.append(
            Issue(
                filepath=self.filepath,
                lineno=node.lineno,
                check_id="EVENT_SOURCING",
                message="Global/nonlocal in apply() — apply() must be pure",
                severity="advisory",
            )
        )


def _check_mutable_event_fields(class_node: ast.ClassDef, filepath: str) -> list[Issue]:
    issues: list[Issue] = []
    for stmt in class_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        annotation = stmt.annotation
        # Direct mutable type annotation: list, dict, set
        if isinstance(annotation, ast.Name) and annotation.id in ("list", "dict", "set") and isinstance(stmt.target, ast.Name):
            issues.append(
                Issue(
                    filepath=filepath,
                    lineno=stmt.lineno,
                    check_id="EVENT_SOURCING",
                    message=(
                        f"Mutable field '{stmt.target.id}: {annotation.id}' in event class '{class_node.name}'"
                        " — use tuple/frozenset/MappingProxyType"
                    ),
                    severity="advisory",
                )
            )
    return issues


def _is_event_class(class_node: ast.ClassDef) -> bool:
    """Return True if the class name contains 'Event' or a base contains 'Event'."""
    if "Event" in class_node.name:
        return True
    for base in class_node.bases:
        if isinstance(base, ast.Name) and "Event" in base.id:
            return True
        if isinstance(base, ast.Attribute) and "Event" in base.attr:
            return True
    return False


def check_event_sourcing(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    issues: list[Issue] = []

    for node in ast.walk(tree):
        # 5a: Side effects in apply()
        if isinstance(node, ast.FunctionDef) and node.name == "apply":
            visitor = _ApplyBodyVisitor(filepath, issues)
            for stmt in node.body:
                visitor.visit(stmt)

        # 5b: Mutable fields in Event classes
        elif isinstance(node, ast.ClassDef) and _is_event_class(node):
            issues.extend(_check_mutable_event_fields(node, filepath))

    return issues

import ast

from exo.security_gate.issue import Issue


def _has_frozen_config(class_node: ast.ClassDef) -> bool:
    """Return True if the class has model_config = ConfigDict(frozen=True)."""
    for stmt in class_node.body:
        # model_config = ConfigDict(frozen=True)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "model_config" and isinstance(stmt.value, ast.Call):
                    for kw in stmt.value.keywords:
                            if (
                                kw.arg == "frozen"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True
                            ):
                                return True
        # model_config: ClassVar[...] = ConfigDict(frozen=True)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "model_config" and isinstance(stmt.value, ast.Call):
            for kw in stmt.value.keywords:
                if (
                    kw.arg == "frozen"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    return True
    return False


def _is_model_base(base: ast.expr) -> bool:
    """Return True if a base class looks like a Pydantic model base."""
    if isinstance(base, ast.Name):
        return base.id == "BaseModel" or base.id.endswith("Model")
    if isinstance(base, ast.Attribute):
        return base.attr == "BaseModel" or base.attr.endswith("Model")
    return False


def _collect_frozen_classes(tree: ast.Module) -> set[str]:
    """Phase 1: find all frozen Pydantic model class names in this file."""
    frozen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_model_base(b) for b in node.bases):
            continue
        if _has_frozen_config(node):
            frozen.add(node.name)
    return frozen


def _collect_frozen_vars(tree: ast.Module, frozen_classes: set[str]) -> dict[str, str]:
    """Return mapping of variable_name -> frozen_class_name based on annotations."""
    frozen_vars: dict[str, str] = {}

    for node in ast.walk(tree):
        # Annotated assignments: var: FrozenClass = ...
        if isinstance(node, ast.AnnAssign):
            annotation = node.annotation
            class_name = _resolve_annotation_name(annotation)
            if class_name in frozen_classes and isinstance(node.target, ast.Name):
                frozen_vars[node.target.id] = class_name

        # Function parameters: def f(var: FrozenClass)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
            if node.args.vararg:
                all_args = all_args + [node.args.vararg]
            if node.args.kwarg:
                all_args = all_args + [node.args.kwarg]
            for arg in all_args:
                if arg.annotation is None:
                    continue
                class_name = _resolve_annotation_name(arg.annotation)
                if class_name in frozen_classes:
                    frozen_vars[arg.arg] = class_name

    return frozen_vars


def _resolve_annotation_name(annotation: ast.expr) -> str:
    """Extract the base class name from an annotation expression."""
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    # Handle Optional[X], etc. — just ignore subscript outer
    if isinstance(annotation, ast.Subscript):
        return _resolve_annotation_name(annotation.value)
    return ""


class _MutationVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str, frozen_vars: dict[str, str]) -> None:
        self.filepath = filepath
        self.frozen_vars = frozen_vars
        self.issues: list[Issue] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in self.frozen_vars
            ):
                var_name = target.value.id
                class_name = self.frozen_vars[var_name]
                self.issues.append(
                    Issue(
                        filepath=self.filepath,
                        lineno=node.lineno,
                        check_id="PYDANTIC_MUTATION",
                        message=(
                            f"Assignment to '{var_name}.{target.attr}' but '{class_name}' is a frozen Pydantic model"
                            " — use model.model_copy(update={...}) instead"
                        ),
                        severity="block",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # setattr(frozen_var, ...)
        func = node.func
        if isinstance(func, ast.Name) and func.id == "setattr" and (
            len(node.args) >= 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.frozen_vars
        ):
                var_name = node.args[0].id
                class_name = self.frozen_vars[var_name]
                attr = ""
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    attr = str(node.args[1].value)
                self.issues.append(
                    Issue(
                        filepath=self.filepath,
                        lineno=node.lineno,
                        check_id="PYDANTIC_MUTATION",
                        message=(
                            f"setattr('{var_name}', '{attr}', ...) but '{class_name}' is a frozen Pydantic model"
                            " — use model.model_copy(update={...}) instead"
                        ),
                        severity="block",
                    )
                )
        self.generic_visit(node)


def check_pydantic_mutation(filepath: str, source: str, tree: ast.Module) -> list[Issue]:
    frozen_classes = _collect_frozen_classes(tree)
    if not frozen_classes:
        return []
    frozen_vars = _collect_frozen_vars(tree, frozen_classes)
    if not frozen_vars:
        return []
    visitor = _MutationVisitor(filepath, frozen_vars)
    visitor.visit(tree)
    return visitor.issues

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from tinygrad import dtypes
from tinygrad.codegen.kernel import Kernel
# from tinygrad import Linearizer

@dataclass
class UOp:
    op: str
    dtype: Any
    args: List[Any]

@dataclass
class Kernel:
    name: str
    uops: List[UOp]

class ASTNode:
    def __init__(self, op: str, dtype: Any, args: List[Any]):
        self.op = op
        self.dtype = dtype
        self.args = args

class Linearizer:
    def __init__(self, name: str, ast: List[ASTNode], opts: Dict[str, Any]):
        self.name = name
        self.ast = ast
        self.opts = opts
        self.uops: List[UOp] = []

    def get_optimized_ast(self) -> List[ASTNode]:
        # Simple constant folding optimization
        optimized_ast = []
        for node in self.ast:
            if node.op in ['add', 'mul', 'sub', 'div'] and all(isinstance(arg, (int, float)) for arg in node.args):
                # Constant folding
                result = self.evaluate_constant_expression(node)
                optimized_ast.append(ASTNode('const', node.dtype, [result]))
            else:
                optimized_ast.append(node)
        return optimized_ast

    def evaluate_constant_expression(self, node: ASTNode) -> Union[int, float]:
        op = node.op
        args = node.args
        if op == 'add':
            return args[0] + args[1]
        elif op == 'mul':
            return args[0] * args[1]
        elif op == 'sub':
            return args[0] - args[1]
        elif op == 'div':
            return args[0] / args[1]
        else:
            raise ValueError(f"Unsupported operation for constant folding: {op}")

    def linearize(self) -> Kernel:
        modified_ast = self.get_optimized_ast()
        self.uops = self.ast_to_uop(modified_ast)
        self.uops = self.optimize_uops(self.uops)
        return Kernel(self.name, self.uops)

    def ast_to_uop(self, ast: List[ASTNode]) -> List[UOp]:
        uops = []
        for node in ast:
            if node.op == 'const':
                uops.append(UOp(op='load_const', dtype=node.dtype, args=node.args))
            elif node.op in ['add', 'mul', 'sub', 'div']:
                for arg in node.args:
                    if not isinstance(arg, (int, float)):
                        uops.append(UOp(op='load', dtype=node.dtype, args=[arg]))
                uops.append(UOp(op=node.op, dtype=node.dtype, args=node.args))
            elif node.op == 'assign':
                uops.append(UOp(op='store', dtype=node.dtype, args=node.args))
            else:
                uops.append(UOp(op=node.op, dtype=node.dtype, args=node.args))
        return uops

    def optimize_uops(self, uops: List[UOp]) -> List[UOp]:
        # Simple dead code elimination
        used_vars = set()
        optimized_uops = []
        
        # Backward pass to mark used variables
        for uop in reversed(uops):
            if uop.op == 'store':
                used_vars.add(uop.args[0])
            elif uop.op != 'load_const':
                used_vars.update(arg for arg in uop.args if isinstance(arg, str))
        
        # Forward pass to keep only necessary operations
        for uop in uops:
            if uop.op == 'store' or uop.op == 'load_const' or any(arg in used_vars for arg in uop.args if isinstance(arg, str)):
                optimized_uops.append(uop)
                if uop.op == 'store':
                    used_vars.remove(uop.args[0])
        
        return optimized_uops

# Algorithm Implementation — nvfp4

**Model:** `mlx-community/Qwen3.5-397B-A17B-nvfp4`

| Metric | Value |
|---|---|
| Thinking tokens | 176 |
| Completion tokens | 2083 |
| Decode tok/s | 38.8 |
| TTFT | 2013ms |
| Total time | 55.7s |

## Thinking

```
I'll implement a persistent (immutable) red-black tree in Python with proper type hints. Let me think through the key aspects:

1. **Immutability**: Every operation (like insert) returns a new tree rather than modifying the existing one
2. **Red-Black Tree Properties**: 
   - Every node is either red or black
   - Root is black
   - Red nodes cannot have red children (no two consecutive red nodes)
   - Every path from root to leaf has the same number of black nodes
3. **Operations needed**: insert, lookup, in-order traversal
4. **Type hints**: Use Python's typing module for proper annotations

Let me structure this:
- Node class to represent tree nodes
- RedBlackTree class as the main interface
- Proper balancing logic for insertions

Here's my implementation:

```

## Response



```python
"""
Persistent (Immutable) Red-Black Tree Implementation

This module provides a fully immutable red-black tree where all modification
operations return new tree instances without altering the original.

Complexity Analysis:
- Insert: O(log n) time, O(log n) space (creates new path from root to leaf)
- Lookup: O(log n) time, O(1) space (iterative)
- In-order Traversal: O(n) time, O(n) space for result list
- Space Complexity: O(log n) additional space per insertion due to path copying

Red-Black Tree Properties (maintained after each operation):
1. Every node is either red or black
2. The root is always black\n3. Red nodes cannot have red children (no consecutive red nodes)\n4. Every path from root to leaf contains the same number of black nodes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterator, List, Optional, TypeVar
from enum import Enum, auto


class Color(Enum):
    RED = auto()\n    BLACK = auto()


T = TypeVar('T')


@dataclass(frozen=True)\nclass Node(Generic[T]):\n    \"\"\"Immutable tree node with color, key, value, and children.\"\"\"\n    key: T\n    value: T\n    color: Color\n    left: Optional[Node[T]] = None\n    right: Optional[Node[T]] = None\n\n    def is_red(self) -> bool:\n        return self.color == Color.RED\n\n\nclass RedBlackTree(Generic[T]):\n    \"\"\"Persistent immutable red-black tree.\"\"\"\n\n    def __init__(self, root: Optional[Node[T]] = None) -> None:\n        self._root = root\n\n    @property\n    def root(self) -> Optional[Node[T]]:\n        return self._root\n\n    def is_empty(self) -> bool:\n        return self._root is None\n\n    def lookup(self, key: T) -> Optional[T]:\n        \"\"\"Return value for key, or None if not found.\"\"\"\n        node = self._root\n        while node is not None:\n            if key == node.key:\n                return node.value\n            elif key < node.key:\n                node = node.left\n            else:\n                node = node.right\n        return None\n\n    def insert(self, key: T, value: T) -> 'RedBlackTree[T]':\n        \"\"\"Return new tree with key-value pair inserted.\"\"\"\n        new_root = self._insert(self._root, key, value)\n        # Ensure root is black\n        if new_root is not None and new_root.is_red():\n            new_root = new_root.replace(color=Color.BLACK)\n        return RedBlackTree(new_root)\n\n    def _insert(self, node: Optional[Node[T]], key: T, value: T) -> Optional[Node[T]]:\n        if node is None:\n            return Node(key=key, value=value, color=Color.RED)\n\n        if key < node.key:\n            new_left = self._insert(node.left, key, value)\n            node = node.replace(left=new_left)\n        elif key > node.key:\n            new_right = self._insert(node.right, key, value)\n            node = node.replace(right=new_right)\n        else:\n            # Update value for existing key\n            node = node.replace(value=value)\n\n        return self._balance(node)\n\n    def _balance(self, node: Node[T]) -> Node[T]:\n        # Case 1: Left child red, left-left grandchild red\n        if (node.left and node.left.is_red() and\n                node.left.left and node.left.left.is_red()):\n            node = self._rotate_right(node)\n            node = node.replace(\n                color=Color.RED,\n                left=node.left.replace(color=Color.BLACK) if node.left else node.left\n            )\n        # Case 2: Left child red, left-right grandchild red\n        elif (node.left and node.left.is_red() and\n              node.left.right and node.left.right.is_red()):\n            node = node.replace(left=self._rotate_left(node.left))\n            node = self._rotate_right(node)\n            node = node.replace(\n                color=Color.RED,\n                left=node.left.replace(color=Color.BLACK) if node.left else node.left\n            )\n        # Case 3: Right child red, right-right grandchild red\n        elif (node.right and node.right.is_red() and\n              node.right.right and node.right.right.is_red()):\n            node = self._rotate_left(node)\n            node = node.replace(\n                color=Color.RED,\n                right=node.right.replace(color=Color.BLACK) if node.right else node.right\n            )\n        # Case 4: Right child red, right-left grandchild red\n        elif (node.right and node.right.is_red() and\n              node.right.left and node.right.left.is_red()):\n            node = node.replace(right=self._rotate_right(node.right))\n            node = self._rotate_left(node)\n            node = node.replace(\n                color=Color.RED,\n                right=node.right.replace(color=Color.BLACK) if node.right else node.right\n            )\n        # Case 5: Both children red\n        elif (node.left and node.left.is_red() and\n              node.right and node.right.is_red()):\n            node = node.replace(\n                left=node.left.replace(color=Color.BLACK),\n                right=node.right.replace(color=Color.BLACK),\n                color=Color.RED\n            )\n\n        return node\n\n    def _rotate_left(self, node: Node[T]) -> Node[T]:\n        right = node.right\n        assert right is not None\n        return right.replace(\n            left=node.replace(right=right.left)\n        )\n\n    def _rotate_right(self, node: Node[T]) -> Node[T]:\n        left = node.left\n        assert left is not None\n        return left.replace(\n            right=node.replace(left=left.right)\n        )\n\n    def in_order(self) -> Iterator[tuple[T, T]]:\n        \"\"\"Yield (key, value) pairs in sorted order.\"\"\"\n        def traverse(node: Optional[Node[T]]) -> Iterator[tuple[T, T]]:\n            if node is None:\n                return\n            yield from traverse(node.left)\n            yield (node.key, node.value)\n            yield from traverse(node.right)\n\n        yield from traverse(self._root)\n\n    def to_list(self) -> List[tuple[T, T]]:\n        \"\"\"Return list of (key, value) pairs in sorted order.\"\"\"\n        return list(self.in_order())\n\n    def __len__(self) -> int:\n        def count(node: Optional[Node[T]]) -> int:\n            if node is None:\n                return 0\n            return 1 + count(node.left) + count(node.right)\n        return count(self._root)\n\n    def __contains__(self, key: T) -> bool:\n        return self.lookup(key) is not None\n\n    def __repr__(self) -> str:\n        items = ', '.join(f'{k}: {v}' for k, v in self.to_list())\n        return f'RedBlackTree({items})'


# Helper to support immutable dataclass field replacement\ndef _replace_node(node: Node[T], **kwargs) -> Node[T]:\n    return Node(\n        key=kwargs.get('key', node.key),\n        value=kwargs.get('value', node.value),\n        color=kwargs.get('color', node.color),\n        left=kwargs.get('left', node.left),\n        right=kwargs.get('right', node.right),\n    )\n\n\n# Monkey-patch replace method onto Node\nNode.replace = _replace_node  # type: ignore[attr-defined]


if __name__ == '__main__':\n    # Example usage\n    tree = RedBlackTree[int]()\n    for i in [10, 5, 15, 3, 7, 12, 17]:\n        tree = tree.insert(i, i * 2)\n\n    print(\"Tree contents:\", tree)\n    print(\"Lookup 7:\", tree.lookup(7))\n    print(\"Contains 15:\", 15 in tree)\n    print(\"Size:\", len(tree))\n    print(\"In-order:\", tree.to_list())\n```

# Algorithm Implementation — 4bit

**Model:** `mlx-community/Qwen3.5-397B-A17B-4bit`

| Metric | Value |
|---|---|
| Thinking tokens | 3572 |
| Completion tokens | 6113 |
| Decode tok/s | 35.8 |
| TTFT | 2186ms |
| Total time | 172.9s |

## Thinking

```
Here's a thinking process that leads to the suggested implementation:

1.  **Understand the Goal:** The objective is to implement a persistent (immutable) Red-Black Tree (RBT) in Python. Key requirements include:
    *   Immutable data structures (updates return new instances).
    *   Type hints (modern Python style).
    *   Operations: `insert`, `lookup`, `in_order_traversal`.
    *   Complexity analysis in the docstring.
    *   Production-quality code (clean, documented, robust).

2.  **Analyze "Persistent" & "Immutable":**
    *   In a mutable RBT, rotations and recoloring modify nodes in place.
    *   In a persistent RBT, every modification along the path from the root to the modified node must create a new node. Unaffected subtrees can be shared between the old and new versions of the tree.
    *   This path copying ensures $O(\log N)$ space and time for updates, maintaining the persistence property.

3.  **Red-Black Tree Properties:**
    *   Every node is either Red or Black.
    *   Root is Black.
    *   Leaves (NIL) are Black.
    *   If a node is Red, both children are Black (no double red).
    *   Every path from a node to its descendant leaves contains the same number of Black nodes (black-height).
    *   *Implementation Detail:* To simplify, we can treat `None` as the NIL leaf (Black), or use a explicit `Nil` node. Using `None` is more Pythonic but requires careful handling of colors. A common functional approach is to have explicit `Empty` or `Nil` nodes, but for Python, `Optional` is standard. Let's use a specific `Color` enum and handle `None` as black leaves, or better yet, use a sentinel `Nil` node to avoid `AttributeError` on children of leaves. Actually, for immutability, explicit nodes are safer. Let's define a `Node` class.
    *   *Optimization:* To make it truly functional/persistent, the structure should be defined using `dataclasses` or simple classes with `__slots__` (though `__slots__` makes copying slightly more verbose, it saves memory). Let's stick to standard dataclasses for readability and immutability guarantees (`frozen=True`).

4.  **Designing the Data Structure:**
    *   `Color`: Enum (RED, BLACK).
    *   `Node`:
        *   `key`: Comparable type.
        *   `value`: Generic type.
        *   `color`: Color.
        *   `left`: Optional['Node'].
        *   `right`: Optional['Node'].
        *   `frozen=True`: Ensures immutability.
    *   `RedBlackTree`:
        *   `root`: Optional['Node'].
        *   Methods: `insert`, `get`, `in_order`.

5.  **Algorithm for Persistent Insert:**
    *   Standard RBT insert involves BST insertion followed by a fix-up phase.
    *   In a persistent setting, we do this recursively.
    *   As we recurse down, we clone nodes on the path.
    *   As we recurse up (unwind the stack), we perform rotations and color flips. Since nodes are immutable, a rotation returns a new subtree root.
    *   *Key Insight:* In functional RBT implementations (like in Okasaki's "Purely Functional Data Structures"), the balancing is often done *on the way up* or immediately upon insertion violation detection.
    *   *Approach:*
        1.  Recursively insert into left or right. This returns a new subtree.
        2.  Check for Red-Black violations (specifically Red-Red conflicts).
        3.  Apply balancing rules (rotations/color flips) locally.
        4.  Ensure the root is always Black.

6.  **Balancing Logic (The Tricky Part):**
    *   There are 4 cases for Red-Red violation where a parent and child are both Red.
    *   Since we are immutable, we don't "rotate" in place. We construct a new subtree structure that satisfies the properties.
    *   Chris Okasaki's approach is very elegant for this:
        *   If we have a Black node with two Red children, and one of the children has a Red child, we balance.
        *   Actually, a simpler way for imperative-to-functional translation:
            *   Insert as Red.
            *   If parent is Red, we have a violation.
            *   Check Uncle.
            *   Recolor or Rotate.
    *   *Functional Pattern:* A `balance` function.
        *   Input: Color, Key, Value, Left Subtree, Right Subtree.
        *   Logic: If we have a Black node with two Red children, or specific Red-Red configurations, restructure.
        *   Wait, the standard functional RBT insertion (Okasaki) works like this:
            *   Insert recursively.
            *   The recursive call returns a subtree.
            *   Check if the current node + returned subtree creates a violation.
            *   Actually, the most common functional implementation (like in Haskell or Scala libraries) uses a `balance` function that is called after the recursive step.
            *   Pattern: `balance(color, key, value, left, right)`.
            *   If `left` is Red and `left.left` is Red -> Rotate.
            *   If `left` is Red and `left.right` is Red -> Rotate.
            *   (Symmetric for Right).
            *   Otherwise, just construct the node.
            *   Finally, force root to Black.

7.  **Refining the Implementation Details:**
    *   **Type Hinting:** Use `TypeVar`, `Generic`, `Optional`, `Iterator`.
    *   **Immutability:** Use `@dataclass(frozen=True)`.
    *   **Sentinel:** Using `None` for leaves is fine, but we need to handle `None` checks carefully. Let's use `None` for empty trees to keep it Pythonic, but ensure type safety.
    *   **Lookup:** Standard BST search.
    *   **Traversal:** Generator for memory efficiency.

8.  **Drafting the Code (Mental or Scratchpad):**
    *   Need `Enum` for Color.
    *   Need `Node` dataclass.
    *   Need `RedBlackTree` class wrapping the root.
    *   `insert` method calls `_insert` helper.
    *   `_insert` returns a `Node`.
    *   `balance` helper handles the rotation logic.
    *   **Crucial:** In persistent RBT, the `balance` function is usually called *after* the recursive insertion returns.
    *   Wait, Okasaki's implementation is slightly different. It inserts a Red node. If it creates a Red-Red conflict, the `balance` function fixes it *during the unwind*.
    *   Let's follow the Okasaki pattern:
        *   `insert(x, t)`:
            *   If `t` is Empty: Return Red Node.
            *   If `x < t.key`: `t.copy(left=insert(x, t.left))`
            *   If `x > t.key`: `t.copy(right=insert(x, t.right))`
            *   Then pass the result through `balance`.
        *   Actually, to avoid double traversal, the `balance` logic is usually integrated into the node construction after recursion.
        *   Better approach for Python:
            *   Recursive `_insert` returns a `Node`.
            *   Inside `_insert`, after getting the new left/right child, we call a `balance` function that takes the current node's color, key, value, new left, new right.
            *   This `balance` function checks for violations and returns a corrected `Node`.
            *   Finally, the public `insert` ensures the root is Black.

9.  **Complexity Analysis:**
    *   Lookup: $O(\log N)$.
    *   Insert: $O(\log N)$ (Path copying + constant number of rotations).
    *   Space: $O(\log N)$ for new nodes per insert (structural sharing).
    *   Traversal: $O(N)$.

10. **Refining the Code Structure:**
    *   Imports: `dataclasses`, `enum`, `typing`, `generator`.
    *   Define `Color`.
    *   Define `Node`.
    *   Define `RedBlackTree`.
    *   Add `__len__`? Maybe not strictly required but good practice. Let's stick to the prompt requirements: insert, lookup, traversal.
    *   Add docstrings.

11. **Self-Correction/Edge Case Consideration:**
    *   *Duplicate Keys:* RBTs usually handle unique keys. I should decide whether to update value or ignore. Standard Map behavior is to update. I will implement update-on-insert.
    *   *Root Color:* Must always be Black. The `insert` method must enforce this after the recursive `_insert` finishes.
    *   *Type Safety:* Use `KT` (Key Type) and `VT` (Value Type). Keys must be comparable.
    *   *Performance:* `@dataclass(frozen=True)` is good. `__slots__` might be overkill for this demonstration but good for memory. I'll stick to dataclass for clarity.
    *   *Balancing Logic:* This is the hardest part to get right without mutable state.
    *   *Okasaki's Balance:*
        *   Case 1: Black node, Left child Red, Left-Left grandchild Red.
        *   Case 2: Black node, Left child Red, Left-Right grandchild Red.
        *   Case 3: Black node, Right child Red, Right-Right grandchild Red.
        *   Case 4: Black node, Right child Red, Right-Left grandchild Red.
        *   In all cases, restructure to push the Red color up.
        *   Wait, the functional logic is usually: `balance(color, key, value, left, right)`.
        *   If `left` is Red and `left.left` is Red: Rotate right, flip colors.
        *   If `left` is Red and `left.right` is Red: Rotate left then right, flip colors.
        *   (Symmetric for Right).
        *   This ensures no Red-Red conflict locally.

12. **Finalizing Code Structure:**
    *   `Color` Enum.
    *   `Node` Frozen Dataclass.
    *   `RedBlackTree` Class.
    *   Helper `_balance` function (static or method).
    *   Helper `_insert` recursive function.
    *   `lookup` iterative or recursive (iterative is safer for depth, but recursive is cleaner for immutable structure construction. Python recursion limit is usually 1000, sufficient for demo, but worth noting). Let's use iterative for lookup, recursive for insert (since we need to rebuild the path).

13. **Writing the Code:** (Mental check of the `balance` function)
    *   Input: `color`, `key`, `value`, `left`, `right`.
    *   Check `left.color == RED` and `left.left.color == RED`.
    *   Check `left.color == RED` and `left.right.color == RED`.
    *   ... same for right.
    *   If match, return a new Black node with Red children (the restructured tree).
    *   Else, return the original constructed node.
    *   Wait, if I pass `None` as leaves, I need to handle `AttributeError`.
    *   Solution: Use a `Nil` node singleton or check for `None`. `None` is more Pythonic. I will check `is not None`.
    *   Actually, to make the `balance` function cleaner, it helps if `None` behaves like a Black Node. But `None` has no attributes.
    *   Decision: I will use `Optional[Node]`. In `balance`, I must check if children exist before accessing their colors.
    *   Wait, if a child is `None`, it's Black. So `child.color` would fail.
    *   Better: Helper method `is_red(node)` that returns False if node is None.

14. **Review against constraints:**
    *   Persistent/Immutable? Yes (`frozen=True`, returns new instances).
    *   Type hints? Yes (`Generic`, `Optional`, etc.).
    *   Insert, Lookup, Traversal? Yes.
    *   Complexity docstring? Yes.

15. **Final Polish:**
    *   Ensure docstrings are clear.
    *   Add an example usage block (`if __name__ == "__main__"`).
    *   Check PEP 8.

*(Self-Correction during drafting code)*:
One specific detail on Persistent RBT:
When inserting, we always insert a RED node.
The `balance` function is called *after* the recursive call returns.
The `balance` function takes the *current* node context (color, key, left, right).
If the recursive call returned a Red node that creates a Red-Red conflict with the current node, `balance` fixes it.
Actually, the standard functional implementation (Okasaki) passes the color, key, value, left, right to a `balance` function.
If `left` is Red and `left.left` is Red -> Fix.
If `left` is Red and `left.right` is Red -> Fix.
This covers the violations bubbling up.
Finally, the root is forced to Black.

Let's write it.

```python
from __ import dataclasses, enum, typing, generator
# ... implementation ...
```
Wait, one nuance: If I use `frozen=True`, I cannot modify attributes. I must create new instances. This aligns perfectly with persistence.

Let's refine the `balance` logic.
It receives `color, key, value, left, right`.
It checks for violations.
If violation found, it returns a new Node (Black) with new children (Red).
This effectively pushes the red color up the tree.
If the root becomes Red after balancing, the top-level `insert` method forces it to Black.

Okay, ready to generate.

```python
import enum
import dataclasses
from typing import Generic, TypeVar, Optional, Iterator, Tuple

# ...
```
Complexity:
Insert: O(log n) time, O(log n) extra space (path copying).
Lookup: O(log n) time, O(1) extra space.
Traversal: O(n) time, O(n) space (for the list) or O(1) auxiliary (generator).

Let's ensure the `balance` function handles `None` correctly.
`is_red(node)` helper is essential.

Okay, plan is solid. I will write the code now.

One more check: `TypeVar`.
`KT = TypeVar('KT', bound=Comparable)` -> Python doesn't have a direct `Comparable` bound. Usually just `TypeVar('KT')`. I'll add a comment or use `typing.Protocol` if I want to be strict, but simple `TypeVar` is standard for generic keys.

Let's construct the `balance` function carefully.
It needs to handle the 4 cases of Red-Red violation.
1. Left-Left
2. Left-Right
3. Right-Right
4. Right-Left
In all cases, the result is a Black node with two Red children.
The "middle" node becomes the new root of this subtree, colored Black. The outer nodes become children, colored Red.
This is the core of the functional RBT.

Wait, `balance` signature:
`def _balance(color: Color, key: KT, value: VT, left: Node, right: Node) -> Node:`
Wait, `left` and `right` are `Optional[Node]`.
If `left` is None, `left.color` raises error.
So `is_red` helper: `return node is not None and node.color == Color.RED`.

Okay, proceeding.

```

## Response



```python
"""
Persistent (Immutable) Red-Black Tree Implementation.

This module implements a fully immutable Red-Black Tree in Python.
Every modification operation (insert) returns a new tree instance,
sharing unchanged structure with the original version (persistence).

Complexity Analysis:
---------------------
Let N be the number of nodes in the tree.

1. Insertion:
   - Time: O(log N). We traverse from root to leaf (height h = O(log N)).
           Balancing operations are constant time per node on the path.
   - Space: O(log N). Due to immutability, we allocate new nodes for the
            path from root to the inserted node. Unchanged subtrees are shared.

2. Lookup:
   - Time: O(log N). Standard BST search.
   - Space: O(1) (iterative) or O(log N) (recursive stack).

3. In-order Traversal:
   - Time: O(N). Visits every node exactly once.
   - Space: O(log N) for stack/recursion depth. Generator yields O(1) per step.

Properties:
-----------
- Every node is either Red or Black.
- The root is always Black (enforced externally).
- All leaves (None) are considered Black.
- If a node is Red, both its children are Black (no double Red).
- Every path from a node to its descendant leaves contains the same 
  number of Black nodes (Black-height).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Generic, Iterator, Optional, Tuple, TypeVar, List


# --- Type Definitions ---

KT = TypeVar('KT')  # Key Type
VT = TypeVar('VT')  # Value Type


class Color(enum.Enum):
    """Enum representing the color of a Red-Black Tree node."""
    RED = 0
    BLACK = 1


@dataclasses.dataclass(frozen=True)
class Node(Generic[KT, VT]):
    """
    Immutable Tree Node.
    
    frozen=True ensures that once created, the node's attributes 
    cannot be modified, enforcing immutability.
    """
    key: KT
    value: VT
    color: Color
    left: Optional[Node[KT, VT]] = None
    right: Optional[Node[KT, VT]] = None

    def is_red(self) -> bool:
        """Check if this node is Red."""
        return self.color == Color.RED


class RedBlackTree(Generic[KT, VT]):
    """
    Persistent Red-Black Tree.
    
    Operations return new instances of RedBlackTree, preserving the 
    original instance unchanged.
    """

    def __init__(self, root: Optional[Node[KT, VT]] = None):
        self._root = root

    @property
    def root(self) -> Optional[Node[KT, VT]]:
        """Access the root node (read-only)."""
        return self._root

    def is_empty(self) -> bool:
        """Check if the tree is empty."""
        return self._root is None

    # --- Helper Methods ---

    @staticmethod
    def _is_red(node: Optional[Node[KT, VT]]) -> bool:
        """
        Check if a node is Red. Returns False if node is None.
        This simplifies balancing logic by treating None as Black.
        """
        return node is not None and node.color == Color.RED

    @staticmethod
    def _balance(
        color: Color,
        key: KT,
        value: VT,
        left: Optional[Node[KT, VT]],
        right: Optional[Node[KT, VT]]
    ) -> Node[KT, VT]:
        """
        Restore Red-Black properties locally.
        
        This function is called during the unwind phase of insertion.
        It checks for Red-Red violations and performs rotations/color flips
        by constructing new nodes.
        """
        # Case 1: Left child is Red and Left-Left grandchild is Red
        if (color == Color.BLACK and 
            RedBlackTree._is_red(left) and 
            left and RedBlackTree._is_red(left.left)):
            
            # Rotate Right, Flip Colors
            # New root becomes the left child (Red -> Black)
            # Old root becomes right child of new root (Black -> Red)
            new_left = left.right
            new_right = Node(key, value, Color.RED, left.right, right)
            return Node(left.key, left.value, Color.BLACK, left.left, new_right)

        # Case 2: Left child is Red and Left-Right grandchild is Red
        if (color == Color.BLACK and 
            RedBlackTree._is_red(left) and 
            left and RedBlackTree._is_red(left.right)):
            
            # Rotate Left-Right, Flip Colors
            new_left = left.left
            new_right = right
            mid_node = left.right
            if mid_node:
                return Node(mid_node.key, mid_node.value, Color.BLACK, 
                            Node(left.key, left.value, Color.RED, new_left, mid_node.left),
                            Node(key, value, Color.RED, mid_node.right, new_right))
            # Fallback (should be covered by logic above, but for type safety)
            return Node(key, value, color, left, right)

        # Case 3: Right child is Red and Right-Right grandchild is Red
        if (color == Color.BLACK and 
            RedBlackTree._is_red(right) and 
            right and RedBlackTree._is_red(right.right)):
            
            # Rotate Left, Flip Colors
            new_left = Node(key, value, Color.RED, left, right.left)
            return Node(right.key, right.value, Color.BLACK, new_left, right.right)

        # Case 4: Right child is Red and Right-Left grandchild is Red
        if (color == Color.BLACK and 
            RedBlackTree._is_red(right) and 
            right and RedBlackTree._is_red(right.left)):
            
            # Rotate Right-Left, Flip Colors
            new_left = left
            new_right = right.right
            mid_node = right.left
            if mid_node:
                return Node(mid_node.key, mid_node.value, Color.BLACK,
                            Node(key, value, Color.RED, new_left, mid_node.left),
                            Node(right.key, right.value, Color.RED, mid_node.right, new_right))
            return Node(key, value, color, left, right)

        # No violation, return node as is
        return Node(key, value, color, left, right)

    def _insert(
        self, 
        node: Optional[Node[KT, VT]], 
        key: KT, 
        value: VT
    ) -> Node[KT, VT]:
        """
        Recursive helper to insert a key-value pair.
        Returns the new root of the subtree.
        """
        if node is None:
            # Insertion point: Always insert as Red
            return Node(key, value, Color.RED)

        # Standard BST Insertion (Immutable Path Copying)
        if key < node.key:
            new_left = self._insert(node.left, key, value)
            # Reconstruct node with new left child
            return self._balance(node.color, node.key, node.value, new_left, node.right)
        elif key > node.key:
            new_right = self._insert(node.right, key, value)
            # Reconstruct node with new right child
            return self._balance(node.color, node.key, node.value, node.left, new_right)
        else:
            # Key exists: Update value (Immutable update)
            # We treat this as replacing the node to maintain immutability
            return self._balance(node.color, node.key, value, node.left, node.right)

    # --- Public API ---

    def insert(self, key: KT, value: VT) -> RedBlackTree[KT, VT]:
        """
        Insert a key-value pair into the tree.
        
        Returns a new RedBlackTree instance with the key-value pair inserted.
        If the key exists, the value is updated.
        """
        new_root = self._insert(self._root, key, value)
        # Ensure root is always Black
        if new_root.color == Color.RED:
            new_root = dataclasses.replace(new_root, color=Color.BLACK)
        
        return RedBlackTree(new_root)

    def lookup(self, key: KT) -> Optional[VT]:
        """
        Lookup a value by key.
        
        Returns the value if found, None otherwise.
        Time Complexity: O(log N)
        """
        current = self._root
        while current is not None:
            if key == current.key:
                return current.value
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return None

    def in_order_traversal(self) -> Iterator[Tuple[KT, VT]]:
        """
        Perform an in-order traversal of the tree.
        
        Yields (key, value) tuples in sorted key order.
        Time Complexity: O(N)
        """
        def _traverse(node: Optional[Node[KT, VT]]) -> Iterator[Tuple[KT, VT]]:
            if node is None:
                return
            
            yield from _traverse(node.left)
            yield (node.key, node.value)
            yield from _traverse(node.right)

        yield from _traverse(self._root)

    def __len__(self) -> int:
        """Return the number of nodes in the tree."""
        # Simple recursive count. For production, maintain a size attribute.
        def _count(node: Optional[Node[KT, VT]]) -> int:
            if node is None:
                return 0
            return 1 + _count(node.left) + _count(node.right)
        
        return _count(self._root)
    
    def __bool__(self) -> bool:
        """True if tree is not empty."""
        return self._root is not None


# --- Example Usage ---

if __name__ == "__main__":
    # Create an empty tree
    tree: RedBlackTree[int, str] = RedBlackTree()
    
    # Insert data (returns new trees)
    tree1 = tree.insert(10, "ten")
    tree2 = tree1.insert(20, "twenty")
    tree3 = tree2.insert(5, "five")
    tree4 = tree3.insert(15, "fifteen")
    
    # Original tree remains empty
    assert tree.is_empty()
    assert tree4.lookup(10) == "ten"
    assert tree4.lookup(5) == "five"
    assert tree4.lookup(99) is None
    
    # Verify persistence (tree1 is unchanged by tree2 operations)
    assert tree1.lookup(20) is None
    
    # Traversal
    print("In-order traversal:")
    for key, value in tree4.in_order_traversal():
        print(f"  {key}: {value}")
        
    # Verify Root Color
    if tree4.root:
        assert tree4.root.color == Color.BLACK, "Root must be Black"
        
    print(f"\nTree Size: {len(tree4)}")
```

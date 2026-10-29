import time

from exo.shared.types.radix_cache import (
    RadixMatchResult,
    RadixNode,
    RadixTelemetry,
    compute_token_hash,
)


class RadixTreeEngine:
    """Implements a content-addressable Radix prefix tree for global KV cache deduplication."""

    ROOT_ID: str = "radix_root"

    def __init__(self) -> None:
        self.nodes: dict[str, RadixNode] = {}
        self._node_counter: int = 0
        self._telemetry_queries: int = 0
        self._telemetry_hits: int = 0
        self._telemetry_avoided: int = 0
        self._telemetry_computed: int = 0

        # Initialize root node
        root_hash = compute_token_hash(())
        root_node = RadixNode(
            node_id=self.ROOT_ID,
            parent_id=None,
            token_subsequence=(),
            block_ids=(),
            node_hash=root_hash,
            children={},
            reference_count=1,
            last_accessed_timestamp=time.time(),
            is_pinned=True,
        )
        self.nodes[self.ROOT_ID] = root_node

    def _generate_node_id(self) -> str:
        self._node_counter += 1
        return f"radix_n{self._node_counter}_{int(time.time() * 1000) % 1000000}"

    def match_prefix(self, tokens: list[int] | tuple[int, ...]) -> RadixMatchResult:
        """Find the maximum common prefix for a token sequence against cached prefix tree."""
        query_tokens = tuple(tokens)
        curr_node = self.nodes[self.ROOT_ID]
        matched_blocks: list[str] = []
        matched_tokens_count = 0
        remaining_tokens = query_tokens

        while remaining_tokens:
            first_token = remaining_tokens[0]
            if first_token not in curr_node.children:
                break

            child_id = curr_node.children[first_token]
            child_node = self.nodes[child_id]

            # Compare tokens along child edge
            edge_tokens = child_node.token_subsequence
            min_len = min(len(edge_tokens), len(remaining_tokens))
            common_len = 0
            while common_len < min_len and edge_tokens[common_len] == remaining_tokens[common_len]:
                common_len += 1

            if common_len == 0:
                break

            if common_len == len(edge_tokens):
                # Full edge match: advance to child
                matched_blocks.extend(child_node.block_ids)
                matched_tokens_count += common_len
                remaining_tokens = remaining_tokens[common_len:]

                # Update child access
                self.nodes[child_id] = RadixNode(
                    node_id=child_node.node_id,
                    parent_id=child_node.parent_id,
                    token_subsequence=child_node.token_subsequence,
                    block_ids=child_node.block_ids,
                    node_hash=child_node.node_hash,
                    children=child_node.children,
                    reference_count=child_node.reference_count,
                    last_accessed_timestamp=time.time(),
                    is_pinned=child_node.is_pinned,
                )
                curr_node = self.nodes[child_id]
            else:
                # Partial edge match
                # Tokens matched common_len within this node
                matched_tokens_count += common_len
                remaining_tokens = remaining_tokens[common_len:]
                break

        is_full_match = len(remaining_tokens) == 0
        return RadixMatchResult(
            matched_tokens_count=matched_tokens_count,
            matched_block_ids=tuple(matched_blocks),
            last_matched_node_id=curr_node.node_id,
            remaining_tokens=remaining_tokens,
            is_full_match=is_full_match,
        )

    def insert_sequence(
        self,
        tokens: list[int] | tuple[int, ...],
        block_ids: list[str] | tuple[str, ...],
        is_pinned: bool = False,
    ) -> str:
        """Insert a sequence into the Radix tree, dynamically splitting nodes on partial matches."""
        seq_tokens = tuple(tokens)
        seq_blocks = tuple(block_ids)
        if not seq_tokens:
            return self.ROOT_ID

        curr_id = self.ROOT_ID
        rem_tokens = seq_tokens
        rem_blocks = seq_blocks

        while rem_tokens:
            curr_node = self.nodes[curr_id]
            first_tok = rem_tokens[0]

            if first_tok not in curr_node.children:
                # Direct branch: create a new leaf node
                new_id = self._generate_node_id()
                new_hash = compute_token_hash(rem_tokens, curr_node.node_hash)

                new_node = RadixNode(
                    node_id=new_id,
                    parent_id=curr_id,
                    token_subsequence=rem_tokens,
                    block_ids=rem_blocks,
                    node_hash=new_hash,
                    children={},
                    reference_count=1,
                    last_accessed_timestamp=time.time(),
                    is_pinned=is_pinned,
                )
                self.nodes[new_id] = new_node

                new_children = dict(curr_node.children)
                new_children[first_tok] = new_id
                self.nodes[curr_id] = RadixNode(
                    node_id=curr_node.node_id,
                    parent_id=curr_node.parent_id,
                    token_subsequence=curr_node.token_subsequence,
                    block_ids=curr_node.block_ids,
                    node_hash=curr_node.node_hash,
                    children=new_children,
                    reference_count=curr_node.reference_count,
                    last_accessed_timestamp=time.time(),
                    is_pinned=curr_node.is_pinned,
                )
                return new_id

            child_id = curr_node.children[first_tok]
            child_node = self.nodes[child_id]
            edge_tokens = child_node.token_subsequence

            # Calculate common prefix length
            min_len = min(len(edge_tokens), len(rem_tokens))
            common_len = 0
            while common_len < min_len and edge_tokens[common_len] == rem_tokens[common_len]:
                common_len += 1

            if common_len == len(edge_tokens):
                # Complete match on edge: advance
                curr_id = child_id
                rem_tokens = rem_tokens[common_len:]
                rem_blocks = rem_blocks[common_len:] if len(rem_blocks) >= common_len else ()
                # Increment reference
                self.nodes[child_id] = RadixNode(
                    node_id=child_node.node_id,
                    parent_id=child_node.parent_id,
                    token_subsequence=child_node.token_subsequence,
                    block_ids=child_node.block_ids,
                    node_hash=child_node.node_hash,
                    children=child_node.children,
                    reference_count=child_node.reference_count + 1,
                    last_accessed_timestamp=time.time(),
                    is_pinned=child_node.is_pinned or is_pinned,
                )
                if not rem_tokens:
                    return child_id
            else:
                # Partial match: SPLIT NODE NEEDED
                # 1. Create split middle node
                split_id = self._generate_node_id()
                split_tokens = edge_tokens[:common_len]
                split_blocks = child_node.block_ids[:common_len] if len(child_node.block_ids) >= common_len else ()
                split_hash = compute_token_hash(split_tokens, curr_node.node_hash)

                # 2. Update existing child to become child of split_node
                remaining_edge_tokens = edge_tokens[common_len:]
                remaining_edge_blocks = (
                    child_node.block_ids[common_len:] if len(child_node.block_ids) >= common_len else ()
                )
                child_new_hash = compute_token_hash(remaining_edge_tokens, split_hash)

                self.nodes[child_id] = RadixNode(
                    node_id=child_node.node_id,
                    parent_id=split_id,
                    token_subsequence=remaining_edge_tokens,
                    block_ids=remaining_edge_blocks,
                    node_hash=child_new_hash,
                    children=child_node.children,
                    reference_count=child_node.reference_count,
                    last_accessed_timestamp=time.time(),
                    is_pinned=child_node.is_pinned,
                )

                split_children = {remaining_edge_tokens[0]: child_id}

                # 3. If there are remaining incoming tokens, create a sibling leaf node
                incoming_remaining_tokens = rem_tokens[common_len:]
                incoming_remaining_blocks = (
                    rem_blocks[common_len:] if len(rem_blocks) >= common_len else ()
                )

                result_id = split_id
                if incoming_remaining_tokens:
                    sibling_id = self._generate_node_id()
                    sibling_hash = compute_token_hash(incoming_remaining_tokens, split_hash)
                    self.nodes[sibling_id] = RadixNode(
                        node_id=sibling_id,
                        parent_id=split_id,
                        token_subsequence=incoming_remaining_tokens,
                        block_ids=incoming_remaining_blocks,
                        node_hash=sibling_hash,
                        children={},
                        reference_count=1,
                        last_accessed_timestamp=time.time(),
                        is_pinned=is_pinned,
                    )
                    split_children[incoming_remaining_tokens[0]] = sibling_id
                    result_id = sibling_id

                # Save split node
                self.nodes[split_id] = RadixNode(
                    node_id=split_id,
                    parent_id=curr_id,
                    token_subsequence=split_tokens,
                    block_ids=split_blocks,
                    node_hash=split_hash,
                    children=split_children,
                    reference_count=child_node.reference_count + 1,
                    last_accessed_timestamp=time.time(),
                    is_pinned=child_node.is_pinned or is_pinned,
                )

                # Point curr_node to split_node
                curr_children = dict(curr_node.children)
                curr_children[first_tok] = split_id
                self.nodes[curr_id] = RadixNode(
                    node_id=curr_node.node_id,
                    parent_id=curr_node.parent_id,
                    token_subsequence=curr_node.token_subsequence,
                    block_ids=curr_node.block_ids,
                    node_hash=curr_node.node_hash,
                    children=curr_children,
                    reference_count=curr_node.reference_count,
                    last_accessed_timestamp=time.time(),
                    is_pinned=curr_node.is_pinned,
                )
                return result_id

        return curr_id

    def evict_lru_branches(self, max_evictions: int = 1) -> list[str]:
        """Evicts unpinned leaf nodes with reference_count == 0 based on LRU order."""
        evicted_block_ids: list[str] = []
        evictions_done = 0

        while evictions_done < max_evictions:
            # Find eligible leaf candidates
            candidates = [
                node
                for node in self.nodes.values()
                if node.is_leaf and node.node_id != self.ROOT_ID and not node.is_pinned and node.reference_count <= 0
            ]
            if not candidates:
                break

            # Pick oldest accessed leaf
            candidates.sort(key=lambda n: n.last_accessed_timestamp)
            victim = candidates[0]

            evicted_block_ids.extend(victim.block_ids)

            # Detach from parent
            parent_id = victim.parent_id
            if parent_id and parent_id in self.nodes:
                parent = self.nodes[parent_id]
                new_children = {tok: cid for tok, cid in parent.children.items() if cid != victim.node_id}
                self.nodes[parent_id] = RadixNode(
                    node_id=parent.node_id,
                    parent_id=parent.parent_id,
                    token_subsequence=parent.token_subsequence,
                    block_ids=parent.block_ids,
                    node_hash=parent.node_hash,
                    children=new_children,
                    reference_count=parent.reference_count,
                    last_accessed_timestamp=parent.last_accessed_timestamp,
                    is_pinned=parent.is_pinned,
                )

            # Delete victim
            self.nodes.pop(victim.node_id, None)
            evictions_done += 1

        return evicted_block_ids

    def record_query(self, matched_tokens: int, computed_tokens: int) -> None:
        """Records query telemetry for token savings ratio and hit rate tracking."""
        self._telemetry_queries += 1
        if matched_tokens > 0:
            self._telemetry_hits += 1
        self._telemetry_avoided += matched_tokens
        self._telemetry_computed += computed_tokens

    def get_telemetry(self) -> RadixTelemetry:
        return RadixTelemetry(
            total_queries=self._telemetry_queries,
            total_prefix_hits=self._telemetry_hits,
            total_tokens_avoided=self._telemetry_avoided,
            total_tokens_computed=self._telemetry_computed,
        )

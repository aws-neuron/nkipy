"""Tree-based radix prefix cache.

Provides O(L) prefix matching and insertion where L is the token sequence
length.  The tree is parameterised by *page_size* which controls the
matching granularity — tokens are compared in chunks of *page_size* and
keys are truncated to multiples of *page_size* before any operation.

With ``page_size == kv_cache_block_size`` (the default in nkipy-serving,
both 32), the tree effectively operates at block granularity while still
storing individual token IDs for correctness.

Complexity summary (n = sequence length, E = cached entries):

    match_prefix   O(n / page_size)   tree walk, one page-compare per step
    insert         O(n / page_size)   tree walk + possible split + leaf create
    evict          O(k log k)         heap over evictable leaves (k ≤ E)
    inc/dec_lock   O(depth)           walk node → root
    evictable_size O(1)               incrementally tracked
    protected_size O(1)               incrementally tracked

Design follows upstream sglang's ``RadixCache`` (tree walk + node
splitting) simplified for nkipy-serving: numpy arrays instead of torch
tensors, no EAGLE/bigram, no hierarchical caching, no KV-cache events.
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from nkipy_serving.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from nkipy_serving.mem_cache.common import req_to_tokens

# Pre-allocated empty array returned on cache miss — avoids per-call allocation.
_EMPTY_INDICES = np.empty(0, dtype=np.int32)
_EMPTY_INDICES.flags.writeable = False


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


class TreeNode:
    """A node in the radix trie.

    Each node represents an *edge* in the compressed trie.  ``key`` holds
    the token-ID segment for this edge and ``value`` holds the
    corresponding KV-cache slot indices (one per token).

    Attributes:
        children: Map from child-key (first token or first *page_size*
            tokens as a tuple) to child ``TreeNode``.
        parent:   Back-pointer to parent (``None`` only for a detached node).
        key:      Token-ID segment stored on this edge.
        value:    1-D ``np.int32`` array of KV-cache slot indices, same
            length as *key*.
        lock_ref: Reference count preventing eviction while a request is
            using the cached slots.  Incremented/decremented along the
            entire path from node to root.
        last_access_time: Monotonic timestamp of last read/write; used
            for LRU eviction ordering.
        payload:  Opaque user data attached at insert time (e.g. for
            cache metadata).  Only meaningful on leaf nodes.
    """

    __slots__ = (
        "children",
        "parent",
        "key",
        "value",
        "lock_ref",
        "last_access_time",
        "payload",
    )

    def __init__(self) -> None:
        self.children: dict[int | tuple, TreeNode] = {}
        self.parent: TreeNode | None = None
        self.key: list[int] = []
        self.value: np.ndarray | None = None
        self.lock_ref: int = 0
        self.last_access_time: float = 0.0
        self.payload: object | None = None

    def __lt__(self, other: TreeNode) -> bool:
        return self.last_access_time < other.last_access_time


# ---------------------------------------------------------------------------
# Radix cache
# ---------------------------------------------------------------------------


class RadixCache(BasePrefixCache):
    """Compressed radix trie for KV-cache prefix reuse.

    The scheduler calls :meth:`match_prefix` on each new request to find
    reusable KV-cache slots, :meth:`inc_lock_ref` / :meth:`dec_lock_ref`
    to protect active entries from eviction, and
    :meth:`cache_finished_req` to donate completed request slots back
    into the cache.

    Parameters:
        page_size:   Matching granularity in tokens.  Keys are truncated
            to multiples of *page_size* and comparisons advance in steps
            of *page_size*.  Should equal ``kv_cache_block_size`` so that
            cache hits are always block-aligned.
        disable:     If ``True`` every public method is a fast no-op.
    """

    def __init__(
        self,
        page_size: int = 1,
        disable: bool = False,
    ):
        if page_size <= 0:
            raise RuntimeError(f"page_size must be > 0, got {page_size}")
        self.page_size = page_size
        # Cap edge length to bound over-lock waste on partial matches.
        # 8 pages ≈ 256 tokens at page_size=32 — partial hit on a full
        # edge over-locks at most 7 pages (~224 tokens), which is
        # negligible vs device memory.  Keeps edge count reasonable
        # (4K prefix → ~16 edges, 128K → ~512).
        self.max_edge_tokens = page_size * 8
        self.disable = disable
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cached data and reset size counters."""
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = _EMPTY_INDICES
        self.root_node.lock_ref = 1  # root is never evictable
        self.evictable_size_: int = 0
        self.protected_size_: int = 0
        self.evictable_leaves: set[TreeNode] = set()
        self._EMPTY_RESULT: MatchResult | None = None

    def match_prefix(self, key: list[int] | Any, **kwargs: Any) -> MatchResult:
        """Return the longest cached prefix of *key*.

        The returned ``MatchResult.last_device_node`` (a :class:`TreeNode`)
        should be passed to :meth:`inc_lock_ref` to protect the cached
        slots from eviction while the request is in flight.

        Hot path — called once per request admission.
        """
        if self.disable:
            return self._empty_result()

        tokens = self._normalize_key(key)
        if not tokens:
            return self._empty_result()

        values, last_node = self._match_prefix_helper(self.root_node, tokens)
        if values:
            device_indices = np.concatenate(values)
        else:
            device_indices = _EMPTY_INDICES

        return MatchResult(
            device_indices=device_indices,
            last_device_node=last_node,
            last_host_node=last_node,
            host_hit_length=int(device_indices.size),
            payload=last_node.payload if last_node is not self.root_node else None,
        )

    def insert(
        self,
        key: list[int] | Any,
        value: np.ndarray | list[int] | None = None,
        payload: object | None = None,
    ) -> int:
        """Insert *key* → *value* into the cache.

        Returns the prefix length that was already present in the tree
        (i.e. the number of tokens that did NOT need to be stored).
        """
        if self.disable:
            return 0

        tokens = self._normalize_key(key)
        if not tokens:
            return 0

        if value is None:
            value_arr = np.arange(1, len(tokens) + 1, dtype=np.int32)
        else:
            value_arr = np.asarray(value, dtype=np.int32)[: len(tokens)]

        return self._insert_helper(self.root_node, tokens, value_arr, payload)

    def cache_finished_req(self, req: Any, **kwargs: Any) -> int:
        """Donate a completed request's prompt slots into the cache."""
        return self._cache_req(req)

    def cache_unfinished_req(self, req: Any, **kwargs: Any) -> int:
        """Cache an in-progress request (e.g. for chunked prefill)."""
        return self._cache_req(req)

    def _cache_req(self, req: Any) -> int:
        tokens = req_to_tokens(req)
        if not tokens:
            return 0
        value = getattr(req, "kv_indices", None)
        payload = getattr(req, "cache_payload", None)
        return self.insert(tokens, value=value, payload=payload)

    def evict(self, num_tokens: int) -> list[np.ndarray]:
        """Evict at least *num_tokens* worth of unlocked cache entries.

        Uses LRU ordering over evictable leaf nodes.  After removing a
        leaf, its parent is pushed onto the heap if it became an
        evictable leaf itself (no remaining children, unlocked).

        Returns a list of KV-index arrays that were removed from the
        tree.  The caller must free these back to the allocator.
        """
        if self.disable or num_tokens <= 0:
            return []

        heap: list[tuple[float, TreeNode]] = [
            (node.last_access_time, node) for node in self.evictable_leaves
        ]
        heapq.heapify(heap)

        freed: list[np.ndarray] = []
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, node = heapq.heappop(heap)
            if node.parent is None:
                # Stale entry: node was already evicted (e.g. its sibling's
                # eviction cascaded to the parent which was then also evicted).
                continue
            if node.value is not None:
                num_evicted += int(node.value.size)
                freed.append(node.value)
            parent = node.parent
            self._delete_leaf(node)
            if (
                parent is not self.root_node
                and not parent.children
                and parent.lock_ref == 0
            ):
                heapq.heappush(heap, (parent.last_access_time, parent))
        return freed

    def inc_lock_ref(self, node: Any) -> int:
        """Lock *node* and all ancestors to prevent eviction.

        Returns the (negative) change in evictable token count, which
        the scheduler can use to track available capacity.
        """
        if self.disable or not isinstance(node, TreeNode) or node is self.root_node:
            return 0
        delta = 0
        cur: TreeNode | None = node
        while cur is not self.root_node and cur is not None:
            if cur.lock_ref == 0:
                sz = int(cur.value.size) if cur.value is not None else 0
                self.evictable_size_ -= sz
                self.protected_size_ += sz
                delta -= sz
            cur.lock_ref += 1
            # Unconditional discard is fine: interior nodes are never in the
            # set (no-op), and the leaf is correctly removed.  Unlike
            # dec_lock_ref we don't call _update_leaf_status because a
            # locked node can never be an evictable leaf.
            self.evictable_leaves.discard(cur)
            cur = cur.parent
        return delta

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: str | None = None) -> int:
        """Unlock *node* and all ancestors.

        Returns the (positive) change in evictable token count.
        """
        if self.disable or not isinstance(node, TreeNode) or node is self.root_node:
            return 0
        delta = 0
        cur: TreeNode | None = node
        while cur is not self.root_node and cur is not None:
            if cur.lock_ref <= 0:
                cur = cur.parent
                continue
            if cur.lock_ref == 1:
                sz = int(cur.value.size) if cur.value is not None else 0
                self.evictable_size_ += sz
                self.protected_size_ -= sz
                delta += sz
            cur.lock_ref -= 1
            self._update_leaf_status(cur)
            cur = cur.parent
        return delta

    def evictable_size(self) -> int:
        """Total tokens in unlocked (evictable) cache entries — O(1)."""
        return self.evictable_size_

    def protected_size(self) -> int:
        """Total tokens in locked (protected) cache entries — O(1)."""
        return self.protected_size_

    def total_size(self) -> int:
        """Total cached tokens (evictable + protected) — O(1)."""
        return self.evictable_size_ + self.protected_size_

    def pretty_print(self) -> str:
        return (
            "RadixCache("
            f"total={self.total_size()}, "
            f"evictable={self.evictable_size_}, "
            f"protected={self.protected_size_})"
        )

    # ------------------------------------------------------------------
    # Internals — hot path
    # ------------------------------------------------------------------

    def _empty_result(self) -> MatchResult:
        """Return a cache-miss result.  Re-uses a cached MatchResult to
        avoid repeated NamedTuple + ndarray allocation.  Correctness
        relies on ``_EMPTY_INDICES`` being read-only."""
        r = self._EMPTY_RESULT
        if r is None or r.last_device_node is not self.root_node:
            r = MatchResult(
                device_indices=_EMPTY_INDICES,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
                payload=None,
            )
            self._EMPTY_RESULT = r
        return r

    def _normalize_key(self, key: Any) -> list[int]:
        """Extract a page-aligned ``list[int]`` from *key*.

        Fast-path: when the caller already provides ``list`` (e.g. from
        ``ndarray.tolist()``), no copy is made.
        """
        tokens = key.token_ids if hasattr(key, "token_ids") else key
        if not isinstance(tokens, list):
            tokens = [int(t) for t in tokens]
        if self.page_size > 1:
            aligned = (len(tokens) // self.page_size) * self.page_size
            if aligned < len(tokens):
                tokens = tokens[:aligned]
        return tokens

    def _get_child_key(self, tokens: list[int], off: int = 0) -> int | tuple:
        """Return the children-dict key for an edge starting at *tokens[off]*."""
        if self.page_size == 1:
            return tokens[off]
        return tuple(tokens[off : off + self.page_size])

    @staticmethod
    def _require_node_value(node: TreeNode) -> np.ndarray:
        if node.value is None:
            raise RuntimeError("radix cache tree invariant violated: node has no value")
        return node.value

    @staticmethod
    def _require_node_parent(node: TreeNode) -> TreeNode:
        if node.parent is None:
            raise RuntimeError(
                "radix cache tree invariant violated: node has no parent"
            )
        return node.parent

    def _key_match(self, edge: list[int], query: list[int], qoff: int = 0) -> int:
        """Count matching tokens from *edge[0]* vs *query[qoff]*.

        Returns a page-aligned match length.  Fast-path: when the entire
        edge matches (the common case in serving), a single list
        comparison replaces the per-page loop — ~2x faster.
        """
        elen = len(edge)
        qrem = len(query) - qoff
        ps = self.page_size
        if ps == 1:
            min_len = elen if elen < qrem else qrem
            i = 0
            while i < min_len:
                if edge[i] != query[qoff + i]:
                    break
                i += 1
            return i
        # Fast path: check entire edge at once (avoids per-page slicing).
        if elen <= qrem and edge == query[qoff : qoff + elen]:
            return (elen // ps) * ps
        # Slow path: find the page-aligned divergence point.
        min_len = elen if elen < qrem else qrem
        i = 0
        while i + ps <= min_len:
            if edge[i : i + ps] != query[qoff + i : qoff + i + ps]:
                break
            i += ps
        return i

    def _match_prefix_helper(
        self, node: TreeNode, tokens: list[int]
    ) -> tuple[list[np.ndarray], TreeNode]:
        """Walk the tree collecting KV-index arrays for the longest match.

        Uses an integer offset into *tokens* to avoid O(depth × n) list
        copies.  On partial match the child's value is *truncated* (a
        zero-copy numpy view) rather than splitting the node — this
        prevents read traffic from fragmenting hot prefixes into chains
        of tiny edges.  The returned ``last_node`` is the full child,
        which over-locks slightly but is safe: the scheduler only uses
        ``device_indices[:hit_length]``, and the lock is released on
        request completion.  Structural splits happen only on
        :meth:`insert`.
        """
        access_time = time.monotonic()
        node.last_access_time = access_time
        values: list[np.ndarray] = []
        off = 0
        total = len(tokens)

        while off < total:
            child_key = self._get_child_key(tokens, off)
            child = node.children.get(child_key)
            if child is None:
                break
            child.last_access_time = access_time
            prefix_len = self._key_match(child.key, tokens, off)

            child_value = self._require_node_value(child)
            if prefix_len < len(child.key):
                # Partial match — return truncated value, no structural split.
                values.append(child_value[:prefix_len])
                node = child
                break

            values.append(child_value)
            node = child
            off += prefix_len

        return values, node

    # ------------------------------------------------------------------
    # Internals — structure mutation
    # ------------------------------------------------------------------

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        """Split *child* at *split_len*, inserting an intermediate node.

        Before::

            parent ──[child.key]──▶ child

        After::

            parent ──[prefix]──▶ new_node ──[suffix]──▶ child

        ``new_node`` inherits *child*'s ``lock_ref`` so that locked paths
        remain fully protected.  Payload stays on the original (suffix)
        child — intermediate nodes carry no payload.

        Total token count is preserved (prefix + suffix == original).
        """
        child_value = self._require_node_value(child)
        child_parent = self._require_node_parent(child)

        new_node = TreeNode()
        new_node.key = child.key[:split_len]
        new_node.value = child_value[:split_len].copy()
        new_node.parent = child_parent
        new_node.lock_ref = child.lock_ref
        new_node.last_access_time = child.last_access_time

        # Replace child in parent's children dict.
        parent_child_key = self._get_child_key(child.key)
        child_parent.children[parent_child_key] = new_node

        # Demote child to suffix under new_node.
        suffix_key = child.key[split_len:]
        new_node.children[self._get_child_key(suffix_key)] = child
        child.key = suffix_key
        child.value = child_value[split_len:].copy()
        child.parent = new_node

        # Leaf status: child retains its original children (if any) so its
        # leaf/non-leaf status is unchanged.  new_node has child as a child
        # so it is never a leaf.
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        tokens: list[int],
        value: np.ndarray,
        payload: object | None,
    ) -> int:
        """Walk the tree and insert remaining tokens as a new leaf.

        Uses an integer offset to avoid copying *tokens* and *value* on
        each edge transition.  Returns the prefix length already in tree.
        """
        access_time = time.monotonic()
        node.last_access_time = access_time
        total = len(tokens)
        if total == 0:
            return 0

        off = 0
        total_prefix_length = 0

        while off < total:
            child_key = self._get_child_key(tokens, off)
            child = node.children.get(child_key)
            if child is None:
                break

            child.last_access_time = access_time
            prefix_len = self._key_match(child.key, tokens, off)
            total_prefix_length += prefix_len

            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                new_node.last_access_time = access_time
                node = new_node
                off += prefix_len
                break

            node = child
            off += prefix_len

        if off < total:
            # Build a chain of capped-length edges directly (O(n), each
            # token copied once).  This avoids the O(n²/cap) cost of
            # create-then-chop via repeated _split_node suffix copies.
            cap = self.max_edge_tokens
            remaining = total - off
            self.evictable_leaves.discard(node)  # parent is no longer a leaf
            self.evictable_size_ += remaining

            while remaining > cap:
                chunk = TreeNode()
                chunk.parent = node
                chunk.key = tokens[off : off + cap]
                chunk.value = value[off : off + cap].copy()
                chunk.last_access_time = access_time
                node.children[self._get_child_key(tokens, off)] = chunk
                node = chunk
                off += cap
                remaining -= cap

            # Final leaf (≤ cap tokens).
            leaf = TreeNode()
            leaf.parent = node
            leaf.key = tokens[off:]
            leaf.value = value[off:].copy()
            leaf.last_access_time = access_time
            leaf.payload = payload
            node.children[self._get_child_key(tokens, off)] = leaf
            self.evictable_leaves.add(leaf)
        elif payload is not None:
            node.payload = payload  # exact match — update payload only

        return total_prefix_length

    # ------------------------------------------------------------------
    # Internals — eviction helpers
    # ------------------------------------------------------------------

    def _delete_leaf(self, node: TreeNode) -> None:
        """Remove a leaf node and update bookkeeping."""
        if node is self.root_node or node.parent is None:
            return

        node.parent.children.pop(self._get_child_key(node.key), None)

        sz = int(node.value.size) if node.value is not None else 0
        if node.lock_ref == 0:
            self.evictable_size_ -= sz
        else:
            self.protected_size_ -= sz

        self.evictable_leaves.discard(node)
        parent = node.parent
        node.parent = None  # detach so heap can detect stale entries

        self._update_leaf_status(parent)

    def _update_leaf_status(self, node: TreeNode) -> None:
        """Ensure *node* is in ``evictable_leaves`` iff it is an
        unlocked leaf (no children, ``lock_ref == 0``)."""
        if node is self.root_node:
            return
        if not node.children and node.lock_ref == 0:
            self.evictable_leaves.add(node)
        else:
            self.evictable_leaves.discard(node)

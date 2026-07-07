from __future__ import annotations

from dataclasses import dataclass
import random

MIN_BRANCH_LENGTH = 1e-6


@dataclass
class PhyloTree:
    """Rooted binary tree with branch lengths on child nodes.
    
    Attributes:
        root: The root node ID.
        children: A dictionary mapping internal node IDs to their two child node IDs.
        parent: A dictionary mapping child node IDs to their parent node ID.
        leaf_name: A dictionary mapping leaf node IDs to their names.
        next_id: The next available node ID for adding new nodes.
        branch_length (optional): A dictionary mapping child node IDs to their branch lengths (parent to child).
    """

    root: int
    children: dict[int, tuple[int, int]]
    parent: dict[int, int]
    leaf_name: dict[int, str]
    next_id: int
    branch_length: dict[int, float] = None

    def copy(self) -> PhyloTree:
        return PhyloTree(
            root=self.root,
            children=dict(self.children),
            parent=dict(self.parent),
            branch_length=dict(self.branch_length),
            leaf_name=dict(self.leaf_name),
            next_id=self.next_id,
        )

    def is_leaf(self, node: int) -> bool:
        return node in self.leaf_name

    def postorder_traversal(self) -> list[int]:
        order: list[int] = []

        def _walk(node: int) -> None:
            if node in self.children:
                left, right = self.children[node]
                _walk(left)
                _walk(right)
            order.append(node)

        _walk(self.root)
        return order

    def leaf_nodes(self) -> list[int]:
        return list(self.leaf_name.keys())

    def internal_edges_for_nni(self) -> list[tuple[int, int]]:
        edges: list[tuple[int, int]] = []
        for child, parent in self.parent.items():
            if child == self.root:
                continue
            if child in self.children and parent in self.children:
                edges.append((parent, child))
        return edges

    def propose_nni(self, rng: random.Random) -> bool:
        """In-place rooted NNI proposal around a random internal edge."""
        candidates = self.internal_edges_for_nni()
        if not candidates:
            return False

        p, c = rng.choice(candidates)
        p_left, p_right = self.children[p]
        sibling = p_right if p_left == c else p_left

        c_left, c_right = self.children[c]
        if rng.random() < 0.5:
            chosen, other = c_left, c_right
        else:
            chosen, other = c_right, c_left

        # Parent p keeps edge to c; sibling and one child of c are swapped.
        if p_left == c:
            self.children[p] = (c, chosen)
        else:
            self.children[p] = (chosen, c)

        if c_left == chosen:
            self.children[c] = (sibling, other)
        else:
            self.children[c] = (other, sibling)

        self.parent[chosen] = p
        self.parent[sibling] = c
        return True

    def topology_signature(self) -> str:
        """Canonical rooted topology string with sorted child subtrees."""

        def _sig(node: int) -> str:
            if node in self.leaf_name:
                return _sanitize_label(self.leaf_name[node])
            left, right = self.children[node]
            a = _sig(left)
            b = _sig(right)
            if a <= b:
                return f"({a},{b})"
            return f"({b},{a})"

        return _sig(self.root) + ";"

    def to_newick(self, include_lengths: bool = True, decimals: int = 5) -> str:
        def _subtree(node: int) -> str:
            if node in self.leaf_name:
                label = _sanitize_label(self.leaf_name[node])
                return label
            left, right = self.children[node]
            return f"({_branch(left)},{_branch(right)})"

        def _branch(child: int) -> str:
            subtree = _subtree(child)
            if not include_lengths:
                return subtree
            length = self.branch_length[child]
            return f"{subtree}:{length:.{decimals}f}"

        return _subtree(self.root) + ";"


def random_binary_tree(
    taxa: list[str], rng: random.Random, init_branch_mean: float = 0.1
) -> PhyloTree:
    if len(taxa) < 2:
        raise ValueError("Need at least two taxa to form a tree.")

    children: dict[int, tuple[int, int]] = {}
    parent: dict[int, int] = {}
    branch_length: dict[int, float] = {}
    leaf_name: dict[int, str] = {}

    active: list[int] = []
    for i, name in enumerate(taxa):
        leaf_name[i] = name
        active.append(i)

    next_id = len(taxa)
    rate = 1.0 / max(init_branch_mean, MIN_BRANCH_LENGTH)

    while len(active) > 1:
        a, b = rng.sample(active, 2)
        active.remove(a)
        active.remove(b)
        p = next_id
        next_id += 1
        children[p] = (a, b)
        parent[a] = p
        parent[b] = p
        branch_length[a] = max(MIN_BRANCH_LENGTH, rng.expovariate(rate))
        branch_length[b] = max(MIN_BRANCH_LENGTH, rng.expovariate(rate))
        active.append(p)

    root = active[0]
    return PhyloTree(
        root=root,
        children=children,
        parent=parent,
        branch_length=branch_length,
        leaf_name=leaf_name,
        next_id=next_id,
    )


def _sanitize_label(label: str) -> str:
    safe = label.strip().replace(" ", "_")
    return safe if safe else "unnamed"


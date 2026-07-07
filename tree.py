from __future__ import annotations

from dataclasses import dataclass, field
import html
import math
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


@dataclass
class _VizNode:
    name: str | None = None
    length: float = 0.0
    children: list["_VizNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.children


def render_newick_svg(newick: str, width: int = 780) -> str:
    """Render a rooted Newick tree as a rectangular SVG phylogram."""
    root = _parse_newick(newick)
    leaves = _collect_leaves(root)
    if not leaves:
        raise ValueError("Tree has no leaves.")

    _assign_depth(root, 0)
    _assign_y_positions(root, y0=28.0, leaf_gap=32.0)
    _assign_x_positions(root, parent_x=0.0)

    max_x = _max_x(root)
    max_depth = _max_depth(root)
    if (not math.isfinite(max_x)) or max_x <= 0.0:
        _assign_x_by_depth(root)
        max_x = float(max_depth if max_depth > 0 else 1)

    margin_left = 24.0
    margin_right = 180.0
    content_width = max(140.0, width - margin_left - margin_right)
    scale_x = content_width / max_x
    for node in _iter_nodes(root):
        node.x = margin_left + node.x * scale_x

    height = int(max(120.0, max(node.y for node in _iter_nodes(root)) + 26.0))
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" ',
        'xmlns="http://www.w3.org/2000/svg" role="img" ',
        'aria-label="Phylogenetic tree">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#fcfdff"/>',
    ]

    _draw_tree(svg_parts, root)

    for leaf in leaves:
        label = html.escape(leaf.name or "leaf")
        svg_parts.append(
            f'<text x="{leaf.x + 8:.2f}" y="{leaf.y + 4:.2f}" '
            'font-size="12" fill="#2b3642" font-family="Segoe UI, Tahoma, sans-serif">'
            f"{label}</text>"
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def _parse_newick(newick: str) -> _VizNode:
    text = newick.strip()
    if not text.endswith(";"):
        raise ValueError("Newick string must end with ';'.")
    n = len(text)
    i = 0

    def parse_subtree() -> _VizNode:
        nonlocal i
        if i >= n:
            raise ValueError("Unexpected end of Newick string.")

        if text[i] == "(":
            i += 1
            left = parse_subtree()
            if i >= n or text[i] != ",":
                raise ValueError("Expected ',' separating children.")
            i += 1
            right = parse_subtree()
            if i >= n or text[i] != ")":
                raise ValueError("Expected ')' closing subtree.")
            i += 1
            node = _VizNode(children=[left, right], length=0.0)
            if i < n and text[i] == ":":
                i += 1
                node.length = _parse_number()
            return node

        label = _parse_label()
        node = _VizNode(name=label, length=0.0)
        if i < n and text[i] == ":":
            i += 1
            node.length = _parse_number()
        return node

    def _parse_number() -> float:
        nonlocal i
        start = i
        while i < n and text[i] not in ",();":
            i += 1
        if start == i:
            raise ValueError("Expected numeric branch length.")
        token = text[start:i].strip()
        value = float(token)
        if not math.isfinite(value) or value < 0.0:
            return 0.0
        return value

    def _parse_label() -> str:
        nonlocal i
        start = i
        while i < n and text[i] not in ":,();":
            i += 1
        label = text[start:i].strip()
        if not label:
            raise ValueError("Encountered empty leaf label.")
        return label

    root = parse_subtree()
    if i >= n or text[i] != ";":
        raise ValueError("Unexpected trailing text in Newick string.")
    return root


def _assign_depth(node: _VizNode, depth: int) -> None:
    node.depth = depth
    for child in node.children:
        _assign_depth(child, depth + 1)


def _assign_y_positions(node: _VizNode, y0: float, leaf_gap: float) -> None:
    leaves = _collect_leaves(node)
    for idx, leaf in enumerate(leaves):
        leaf.y = y0 + idx * leaf_gap
    _assign_internal_y(node)


def _assign_internal_y(node: _VizNode) -> None:
    if node.is_leaf:
        return
    for child in node.children:
        _assign_internal_y(child)
    y_values = [child.y for child in node.children]
    node.y = 0.5 * (min(y_values) + max(y_values))


def _assign_x_positions(node: _VizNode, parent_x: float) -> None:
    node.x = parent_x
    for child in node.children:
        length = child.length
        if not math.isfinite(length) or length < 0.0:
            length = 0.0
        _assign_x_positions(child, parent_x + length)


def _assign_x_by_depth(node: _VizNode) -> None:
    node.x = float(node.depth)
    for child in node.children:
        _assign_x_by_depth(child)


def _draw_tree(svg_parts: list[str], node: _VizNode) -> None:
    if node.is_leaf:
        svg_parts.append(
            f'<circle cx="{node.x:.2f}" cy="{node.y:.2f}" r="2.3" fill="#2b3642"/>'
        )
        return

    child_ys = [child.y for child in node.children]
    y_min = min(child_ys)
    y_max = max(child_ys)
    svg_parts.append(
        f'<line x1="{node.x:.2f}" y1="{y_min:.2f}" x2="{node.x:.2f}" y2="{y_max:.2f}" '
        'stroke="#2f4a67" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<circle cx="{node.x:.2f}" cy="{node.y:.2f}" r="2.3" fill="#2f4a67"/>'
    )

    for child in node.children:
        svg_parts.append(
            f'<line x1="{node.x:.2f}" y1="{child.y:.2f}" x2="{child.x:.2f}" y2="{child.y:.2f}" '
            'stroke="#2f4a67" stroke-width="1.5"/>'
        )
        _draw_tree(svg_parts, child)


def _collect_leaves(node: _VizNode) -> list[_VizNode]:
    if node.is_leaf:
        return [node]
    leaves: list[_VizNode] = []
    for child in node.children:
        leaves.extend(_collect_leaves(child))
    return leaves


def _iter_nodes(node: _VizNode):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _max_x(node: _VizNode) -> float:
    return max(n.x for n in _iter_nodes(node))


def _max_depth(node: _VizNode) -> int:
    return max(n.depth for n in _iter_nodes(node))


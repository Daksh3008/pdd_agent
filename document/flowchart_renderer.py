# document/flowchart_renderer.py

"""
Flowchart generation from DOT code.
Renders LLM-generated DOT code using Graphviz.
"""

import re
import os
from typing import Dict, List, Optional

try:
    from graphviz import Source, Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("    [Flowchart] graphviz not installed. Install: pip install graphviz")


def _inject_minimal_style(dot_code: str) -> str:
    """
    Inject minimal global attributes (DPI, font) WITHOUT destroying
    the existing node/edge declarations.
    """
    if not dot_code or not dot_code.strip():
        return dot_code

    match = re.search(r'(digraph\s+\w*\s*\{)', dot_code)
    if not match:
        return dot_code

    style_block = """
    dpi=150;
    size="8,10";
    graph [fontname="Arial", fontsize=10];
"""

    insert_pos = match.end()
    styled = dot_code[:insert_pos] + style_block + dot_code[insert_pos:]

    return styled


def _validate_dot_code(dot_code: str) -> bool:
    """Basic validation that DOT code is structurally sound."""
    if not dot_code:
        return False

    if 'digraph' not in dot_code:
        return False

    open_count = dot_code.count('{')
    close_count = dot_code.count('}')
    if open_count != close_count:
        print(f"    [Flowchart] Warning: Unbalanced braces ({open_count} open, {close_count} close)")
        return False

    if '->' not in dot_code:
        print("    [Flowchart] Warning: No edges found in DOT code")
        return False

    return True


def _fix_common_dot_issues(dot_code: str) -> str:
    """Fix common issues in LLM-generated DOT code."""
    # Fix unbalanced braces
    open_count = dot_code.count('{')
    close_count = dot_code.count('}')
    if open_count > close_count:
        dot_code = dot_code + '\n}' * (open_count - close_count)
    elif close_count > open_count:
        for _ in range(close_count - open_count):
            last_brace = dot_code.rfind('}')
            if last_brace != -1:
                dot_code = dot_code[:last_brace] + dot_code[last_brace + 1:]

    # Fix invisible end node
    dot_code = re.sub(
        r'(end\s*\[.*?)style\s*=\s*invis(.*?\])',
        r'\1style=filled, fillcolor=lightcoral\2',
        dot_code,
        flags=re.IGNORECASE
    )

    return dot_code


def render_dot_direct(dot_code: str, output_path: str) -> Optional[str]:
    """
    Render DOT code directly using graphviz.Source.
    Preserves all structure: subgraphs, grouped declarations, etc.
    """
    if not GRAPHVIZ_AVAILABLE:
        print("    [Flowchart] Graphviz not available")
        return None

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        src = Source(dot_code, format='png')
        rendered_path = src.render(output_path, cleanup=True)

        print(f"    [Flowchart] Rendered: {rendered_path}")
        return rendered_path

    except Exception as e:
        print(f"    [Flowchart] Render failed: {e}")
        return None


def generate_flowchart_from_dot(
    dot_code: str,
    output_path: str = "flowchart"
) -> Optional[str]:
    """
    Generate flowchart PNG from DOT code.

    Pipeline:
    1. Validate DOT code
    2. Fix common issues
    3. Inject minimal styling
    4. Render via graphviz.Source
    5. Fall back to simplified rendering if needed

    Args:
        dot_code: Raw DOT code.
        output_path: Output file path (without extension).

    Returns:
        Path to generated PNG file, or None if failed.
    """
    if not dot_code or not dot_code.strip():
        print("    [Flowchart] No DOT code provided")
        return None

    if not GRAPHVIZ_AVAILABLE:
        print("    [Flowchart] Graphviz not available, cannot render flowchart")
        return None

    print(f"    [Flowchart] Processing {len(dot_code)} chars of DOT code...")

    # Step 1: Fix common issues
    fixed_code = _fix_common_dot_issues(dot_code)

    # Step 2: Inject minimal style
    styled_code = _inject_minimal_style(fixed_code)

    # Step 3: Validate
    if not _validate_dot_code(styled_code):
        print("    [Flowchart] Validation failed, attempting render anyway...")

    # Step 4: Direct render
    result = render_dot_direct(styled_code, output_path)
    if result:
        return result

    # Step 5: Try without style injection
    print("    [Flowchart] Retrying with original DOT code...")
    result = render_dot_direct(fixed_code, output_path)
    if result:
        return result

    # Step 6: Try raw DOT code
    print("    [Flowchart] Retrying with raw DOT code...")
    result = render_dot_direct(dot_code, output_path)
    if result:
        return result

    # Step 7: Fallback
    print("    [Flowchart] All renders failed, using fallback...")
    return _fallback_render(dot_code, output_path)


def _fallback_render(dot_code: str, output_path: str) -> Optional[str]:
    """Last-resort fallback: parse what we can and build a simple flowchart."""
    if not GRAPHVIZ_AVAILABLE:
        return None

    try:
        data = _extract_flowchart_data_simple(dot_code)

        if not data['steps']:
            print("    [Flowchart] Fallback: No steps found")
            return None

        dot = Digraph(format='png')
        dot.attr(rankdir='TB', size='8,10', dpi='150')
        dot.attr('node', fontname='Arial', fontsize='10', style='filled')

        for step in data['steps']:
            nid = step['id']
            label = step['label']
            shape = step.get('shape', 'box')

            fillcolor = 'lightblue'
            ll = label.lower()
            if 'start' in ll or 'begin' in ll:
                fillcolor = 'lightgreen'
                shape = 'ellipse'
            elif 'end' in ll or 'stop' in ll or 'finish' in ll:
                fillcolor = 'lightcoral'
                shape = 'ellipse'
            elif shape == 'diamond':
                fillcolor = 'gold'

            dot.node(nid, label=label, shape=shape, fillcolor=fillcolor)

        for conn in data['connections']:
            if conn.get('label'):
                dot.edge(conn['from'], conn['to'], label=conn['label'])
            else:
                dot.edge(conn['from'], conn['to'])

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        dot.render(output_path, cleanup=True)
        result_path = f"{output_path}.png"
        print(f"    [Flowchart] Fallback render: {result_path}")
        return result_path

    except Exception as e:
        print(f"    [Flowchart] Fallback render failed: {e}")
        return None


def _extract_flowchart_data_simple(dot_code: str) -> Dict[str, list]:
    """Simple extraction for fallback only."""
    steps = []
    connections = []
    step_ids = set()

    # Individual node declarations
    node_pattern = re.compile(
        r'(\w+)\s*\[\s*label\s*=\s*"([^"]+)"(?:.*?shape\s*=\s*(\w+))?.*?\]',
        re.IGNORECASE
    )

    # Grouped node declarations
    group_pattern = re.compile(
        r'node\s*\[.*?shape\s*=\s*(\w+).*?\]\s+([\w\s]+);',
        re.IGNORECASE
    )

    # Edges
    edge_pattern = re.compile(
        r'(\w+)\s*->\s*(\w+)(?:\s*\[.*?label\s*=\s*"([^"]*)".*?\])?',
        re.IGNORECASE
    )

    # Collect shapes from grouped declarations
    node_shapes = {}
    for match in group_pattern.finditer(dot_code):
        shape = match.group(1).lower()
        ids = match.group(2).strip().split()
        for nid in ids:
            nid = nid.strip().rstrip(';')
            if nid and nid not in ('node', 'edge', 'graph', 'digraph', 'subgraph'):
                node_shapes[nid] = shape

    # Extract individual node declarations
    for match in node_pattern.finditer(dot_code):
        nid, label, shape = match.groups()
        if nid in ('node', 'edge', 'graph', 'digraph', 'subgraph', 'rank'):
            continue
        if nid not in step_ids:
            final_shape = (shape or node_shapes.get(nid, 'box')).lower()
            steps.append({
                "id": nid,
                "label": label.replace("\\n", "\n"),
                "shape": final_shape
            })
            step_ids.add(nid)

    # Add nodes from grouped declarations
    for nid, shape in node_shapes.items():
        if nid not in step_ids:
            label = nid.replace('_', ' ').title()
            steps.append({"id": nid, "label": label, "shape": shape})
            step_ids.add(nid)

    # Add nodes from edges
    for match in edge_pattern.finditer(dot_code):
        from_node, to_node, _ = match.groups()
        for nid in (from_node, to_node):
            if nid not in step_ids and nid not in ('node', 'edge', 'graph', 'subgraph'):
                shape = node_shapes.get(nid, 'box')
                label = nid.replace('_', ' ').title()
                steps.append({"id": nid, "label": label, "shape": shape})
                step_ids.add(nid)

    # Extract edges
    for match in edge_pattern.finditer(dot_code):
        from_node, to_node, label = match.groups()
        conn = {"from": from_node, "to": to_node}
        if label:
            conn["label"] = label
        connections.append(conn)

    return {"steps": steps, "connections": connections}


# ============================================================
# Legacy API (backward compatibility)
# ============================================================

def fix_dot_code(dot_code: str) -> str:
    """Legacy: minimal fixes."""
    return _inject_minimal_style(_fix_common_dot_issues(dot_code))


def extract_flowchart_data(dot_code: str) -> Dict[str, list]:
    """Legacy: uses improved parser."""
    return _extract_flowchart_data_simple(dot_code)
# src/flowchart_generator.py

"""
Flowchart generation from DOT code.
Parses DOT code and renders PNG flowchart.
"""

import re
from typing import Dict, List, Optional
from graphviz import Digraph


def fix_dot_code(dot_code: str) -> str:
    """
    Inject custom styling into DOT code.
    
    Args:
        dot_code: Raw DOT code string.
        
    Returns:
        Styled DOT code.
    """
    if not dot_code or not dot_code.strip():
        return dot_code
    
    styled_header = '''digraph {
    rankdir=TB;
    size="8,10";
    dpi=150;
    ranksep=0.8;
    nodesep=0.4;
    node [shape=box, width=2.5, height=0.8, style=filled, fillcolor=lightblue, fontsize=11, fontname="Arial"];
    edge [fontsize=10, fontname="Arial"];'''
    
    # Replace the digraph header with styled version
    fixed_code = re.sub(
        r'digraph\s+\w*\s*\{',
        styled_header,
        dot_code,
        flags=re.DOTALL
    )
    
    return fixed_code


def extract_flowchart_data(dot_code: str) -> Dict[str, List[Dict]]:
    """
    Extract structured data from DOT code.
    
    Args:
        dot_code: DOT code string.
        
    Returns:
        Dictionary with 'steps' and 'connections' lists.
    """
    steps = []
    connections = []
    step_ids = set()
    
    # Pattern for nodes: node_id [label="...", shape=...];
    node_pattern = re.compile(
        r'(\w+)\s*\[\s*label\s*=\s*"([^"]+)"(?:\s*,\s*shape\s*=\s*(\w+))?.*?\]',
        re.IGNORECASE
    )
    
    # Pattern for edges: from -> to [label="..."];
    edge_pattern = re.compile(
        r'(\w+)\s*->\s*(\w+)(?:\s*\[\s*label\s*=\s*"([^"]*)"\s*\])?',
        re.IGNORECASE
    )
    
    # Extract nodes
    for match in node_pattern.finditer(dot_code):
        node_id, label, shape = match.groups()
        if node_id not in step_ids:
            steps.append({
                "id": node_id,
                "label": label.replace("\\n", "\n"),
                "shape": shape.lower() if shape else "box"
            })
            step_ids.add(node_id)
    
    # Extract edges
    for match in edge_pattern.finditer(dot_code):
        from_node, to_node, label = match.groups()
        connection = {"from": from_node, "to": to_node}
        if label:
            connection["label"] = label
        connections.append(connection)
    
    return {"steps": steps, "connections": connections}


def generate_flowchart(
    data: Dict[str, List[Dict]],
    output_path: str = "flowchart"
) -> Optional[str]:
    """
    Generate flowchart PNG from structured data.
    
    Args:
        data: Dictionary with 'steps' and 'connections'.
        output_path: Output file path (without extension).
        
    Returns:
        Path to generated PNG file, or None if failed.
    """
    try:
        dot = Digraph(format='png')
        dot.attr(rankdir='TB', size='8,10', dpi='150')
        
        # Add nodes
        for step in data.get('steps', []):
            node_id = step['id']
            label = step['label']
            shape = step.get('shape', 'box').lower()
            
            # Determine styling based on node type
            fillcolor = "lightblue"
            fontcolor = "black"
            
            label_lower = label.lower()
            if "start" in label_lower or "begin" in label_lower:
                fillcolor = "lightgreen"
                shape = "ellipse"
            elif "end" in label_lower or "stop" in label_lower or "finish" in label_lower:
                fillcolor = "lightcoral"
                shape = "ellipse"
            elif shape == "diamond":
                fillcolor = "gold"
            elif shape == "oval" or shape == "ellipse":
                if "start" in label_lower:
                    fillcolor = "lightgreen"
                else:
                    fillcolor = "lightcoral"
            
            dot.node(
                node_id,
                label=label,
                shape=shape,
                style="filled",
                fillcolor=fillcolor,
                fontcolor=fontcolor
            )
        
        # Add edges
        for conn in data.get('connections', []):
            from_id = conn['from']
            to_id = conn['to']
            label = conn.get('label', '')
            
            if label:
                dot.edge(from_id, to_id, label=label, fontsize="10", fontcolor="gray30")
            else:
                dot.edge(from_id, to_id)
        
        # Render
        dot.render(output_path, cleanup=True)
        output_file = f"{output_path}.png"
        print(f"Flowchart saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error generating flowchart: {e}")
        return None


def generate_flowchart_from_dot(
    dot_code: str,
    output_path: str = "flowchart"
) -> Optional[str]:
    """
    Generate flowchart directly from DOT code.
    
    Args:
        dot_code: Raw or styled DOT code.
        output_path: Output file path.
        
    Returns:
        Path to generated PNG file.
    """
    # Fix styling
    styled_code = fix_dot_code(dot_code)
    
    # Extract data
    data = extract_flowchart_data(styled_code)
    
    if not data['steps']:
        print("Warning: No steps found in DOT code")
        return None
    
    # Generate flowchart
    return generate_flowchart(data, output_path)


if __name__ == "__main__":
    # Test with sample DOT code
    sample_dot = '''
    digraph Test {
        Start [label="Start", shape=oval];
        Step1 [label="Open Application", shape=box];
        Step2 [label="Enter Data", shape=box];
        Decision [label="Valid?", shape=diamond];
        End [label="End", shape=oval];
        
        Start -> Step1;
        Step1 -> Step2;
        Step2 -> Decision;
        Decision -> End [label="Yes"];
        Decision -> Step2 [label="No"];
    }
    '''
    
    result = generate_flowchart_from_dot(sample_dot, "test_flowchart")
    print(f"Generated: {result}")
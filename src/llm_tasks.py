# src/llm_tasks.py

"""
LLM-based tasks for PDD generation.
Uses Ollama to extract information from transcripts.
Optimized prompts for accurate business process extraction.
"""

import re
from typing import Optional, Dict, List, Tuple, Set
from llm_client import llm_client


# Configuration
CHUNK_SIZE = 4000
OVERLAP_SIZE = 200


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> List[str]:
    """Split text into overlapping chunks for processing."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            break_point = text.rfind('. ', start + chunk_size - 200, end)
            if break_point != -1:
                end = break_point + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
        
        if start < 0:
            start = 0
    
    return chunks


def extract_entities(transcript: str) -> Dict[str, List[str]]:
    """
    Extract key entities (companies, applications, systems) from transcript.
    Helps correct Whisper transcription errors.
    
    Args:
        transcript: The meeting transcript.
        
    Returns:
        Dictionary of entity types and their values.
    """
    # Use first and last portions where entities are often mentioned
    sample = transcript[:3000]
    if len(transcript) > 4000:
        sample += "\n...\n" + transcript[-1500:]
    
    prompt = f"""Extract the following entities from this meeting transcript.

IMPORTANT: 
- Fix any obvious spelling/transcription errors (e.g., "Eventy" should be "Ivanti", "Micro Soft" should be "Microsoft")
- Use correct capitalization for company and product names
- Only include entities that are actually mentioned

Return in this exact format:
COMPANY: [company name 1], [company name 2]
APPLICATIONS: [app 1], [app 2], [app 3]
SYSTEMS: [system 1], [system 2]
DEPARTMENTS: [dept 1], [dept 2]
PROCESSES: [process name 1], [process name 2]

Transcript:
{sample}

Entities:"""

    response = llm_client.generate_with_stream(prompt)
    
    entities = {
        "companies": [],
        "applications": [],
        "systems": [],
        "departments": [],
        "processes": []
    }
    
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('COMPANY:'):
                entities["companies"] = [x.strip() for x in line[8:].split(',') if x.strip()]
            elif line.startswith('APPLICATIONS:'):
                entities["applications"] = [x.strip() for x in line[13:].split(',') if x.strip()]
            elif line.startswith('SYSTEMS:'):
                entities["systems"] = [x.strip() for x in line[8:].split(',') if x.strip()]
            elif line.startswith('DEPARTMENTS:'):
                entities["departments"] = [x.strip() for x in line[12:].split(',') if x.strip()]
            elif line.startswith('PROCESSES:'):
                entities["processes"] = [x.strip() for x in line[10:].split(',') if x.strip()]
    
    return entities


def get_project_name(transcript: str) -> str:
    """
    Extract or generate accurate project name from transcript.
    
    Args:
        transcript: The meeting transcript text.
        
    Returns:
        Project name.
    """
    # Extract entities first to get correct spellings
    entities = extract_entities(transcript)
    
    # Build context from entities
    context = ""
    if entities["companies"]:
        context += f"Companies mentioned: {', '.join(entities['companies'])}\n"
    if entities["processes"]:
        context += f"Processes discussed: {', '.join(entities['processes'])}\n"
    if entities["applications"]:
        context += f"Applications used: {', '.join(entities['applications'])}\n"
    
    sample = transcript[:2500]
    if len(transcript) > 3500:
        sample += "\n...\n" + transcript[-1000:]
    
    prompt = f"""Identify the main project or process name from this meeting transcript.

Known Entities (use correct spellings from here):
{context}

Instructions:
- Use the EXACT correct spelling of company/product names from the entities above
- If a specific project name is mentioned, use it exactly
- If no project name is mentioned, create a descriptive name like "[Company] [Process] Automation"
- Return ONLY the project name (3-7 words)
- No explanations, quotes, or extra text

Transcript:
{sample}

Project Name:"""

    response = llm_client.generate_with_stream(prompt)
    
    if response:
        name = response.strip().split('\n')[0].strip()
        name = name.strip('"\'').strip()
        words = name.split()[:7]
        return ' '.join(words) if words else "Process Automation Project"
    
    return "Process Automation Project"


def get_document_purpose(transcript: str, project_name: str) -> str:
    """
    Generate a specific document purpose based on the actual process.
    
    Args:
        transcript: The meeting transcript.
        project_name: The project name.
        
    Returns:
        Document purpose text.
    """
    sample = transcript[:4000]
    
    prompt = f"""Based on this meeting transcript for "{project_name}", write a specific Document Purpose section.

The Document Purpose should explain:
1. What specific process is being documented
2. Why this process is being automated
3. What business problem it solves
4. Who will use this document

Write 2-3 paragraphs in professional language.
Do NOT use generic placeholder text.
Be specific to this actual process.

Transcript:
{sample}

Document Purpose:"""

    response = llm_client.generate_with_stream(prompt)
    
    if response and len(response) > 100:
        return response.strip()
    
    # Fallback to generic but include project name
    return f"""The purpose of this Process Definition Document (PDD) is to capture the business-related details of the {project_name} process being automated. It describes how the automated solution will operate and serves as a key input for the technical design.

This document ensures that:
• Process requirements for {project_name} are captured in line with organizational standards
• It provides detailed information on the process flow and step-by-step procedures
• Stakeholders have a clear understanding of the expected results and automation objectives
• The development team has accurate specifications for building the solution"""


def get_process_summary(transcript: str, entities: Dict = None) -> str:
    """
    Generate accurate business process summary from transcript.
    
    Args:
        transcript: The meeting transcript text.
        entities: Pre-extracted entities for context.
        
    Returns:
        Process summary text.
    """
    if entities is None:
        entities = extract_entities(transcript)
    
    # Build entity context
    entity_context = "Use these correct entity names:\n"
    if entities["companies"]:
        entity_context += f"- Companies: {', '.join(entities['companies'])}\n"
    if entities["applications"]:
        entity_context += f"- Applications: {', '.join(entities['applications'])}\n"
    if entities["systems"]:
        entity_context += f"- Systems: {', '.join(entities['systems'])}\n"
    
    chunks = split_into_chunks(transcript, chunk_size=5000)
    
    if len(chunks) == 1:
        return _summarize_single(chunks[0], entity_context)
    
    print(f"    [Chunking] Processing {len(chunks)} chunks for summary...")
    
    # Step 1: Extract key points from each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"    [Chunk {i+1}/{len(chunks)}] Analyzing...")
        
        prompt = f"""Extract the key business process information from this transcript section.

{entity_context}

Focus on:
1. What actions are being performed
2. Which applications/systems are used
3. What data is being processed
4. What decisions or validations occur
5. What is the expected outcome

Be specific and use correct entity names. List as bullet points.

Transcript Section {i+1}:
{chunk}

Key Process Points:"""

        summary = llm_client.generate_with_stream(prompt)
        if summary:
            chunk_summaries.append(summary)
    
    if not chunk_summaries:
        return "Unable to generate process summary."
    
    combined = "\n\n".join(chunk_summaries)
    
    print(f"    [Final] Creating cohesive summary...")
    
    final_prompt = f"""Create a professional business process summary from these extracted points.

{entity_context}

Instructions:
- Write 3-4 paragraphs describing the complete process
- Start with the process objective/purpose
- Describe the main steps in logical order
- Mention the applications and systems used
- End with the expected outcome
- Use professional business language
- NO bullet points, write flowing paragraphs
- Use CORRECT spellings of all company/product names

Extracted Points:
{combined}

Process Summary:"""

    final_summary = llm_client.generate_with_stream(final_prompt)
    return final_summary if final_summary else combined


def _summarize_single(transcript: str, entity_context: str) -> str:
    """Summarize a short transcript."""
    prompt = f"""Write a professional business process summary from this meeting transcript.

{entity_context}

Instructions:
- Write 3-4 paragraphs describing the complete process
- Start with the process objective/purpose  
- Describe the main steps in logical order
- Mention the applications and systems used
- End with the expected outcome
- Use professional business language
- NO bullet points

Transcript:
{transcript}

Process Summary:"""

    response = llm_client.generate_with_stream(prompt)
    return response if response else "Unable to generate process summary."


def get_inputs_outputs(transcript: str, entities: Dict = None) -> str:
    """
    Extract detailed process inputs and outputs.
    
    Args:
        transcript: The meeting transcript text.
        entities: Pre-extracted entities.
        
    Returns:
        Formatted inputs and outputs text.
    """
    if entities is None:
        entities = extract_entities(transcript)
    
    entity_context = ""
    if entities["applications"]:
        entity_context = f"Applications in use: {', '.join(entities['applications'])}\n"
    
    chunks = split_into_chunks(transcript, chunk_size=5000)
    
    all_inputs = set()
    all_outputs = set()
    
    print(f"    [Chunking] Processing {len(chunks)} chunks for I/O...")
    
    for i, chunk in enumerate(chunks):
        print(f"    [Chunk {i+1}/{len(chunks)}] Extracting I/O...")
        
        prompt = f"""Analyze this transcript section and identify process INPUTS and OUTPUTS.

{entity_context}

INPUTS are:
- Files or documents received/uploaded (e.g., "Excel spreadsheet with employee data")
- Data entered by users (e.g., "Employee ID", "Date range")
- Information retrieved from systems (e.g., "Customer records from CRM")
- Credentials or access requirements
- Trigger events that start the process

OUTPUTS are:
- Files or reports generated (e.g., "PDF invoice", "Excel report")
- Data saved to systems (e.g., "Updated customer record in database")
- Emails or notifications sent
- Status updates or confirmations
- Decisions or approvals recorded

Format each item as:
INPUT: [specific description]
OUTPUT: [specific description]

Be SPECIFIC - include file types, field names, system names where mentioned.

Transcript Section:
{chunk}

Inputs and Outputs:"""

        response = llm_client.generate_with_stream(prompt)
        
        if response:
            for line in response.split('\n'):
                line = line.strip()
                if line.upper().startswith('INPUT:'):
                    item = line[6:].strip().strip('-•*').strip()
                    if item and len(item) > 3:
                        all_inputs.add(item)
                elif line.upper().startswith('OUTPUT:'):
                    item = line[7:].strip().strip('-•*').strip()
                    if item and len(item) > 3:
                        all_outputs.add(item)
    
    # Format nicely
    result = "**Inputs:**\n"
    if all_inputs:
        for inp in sorted(all_inputs):
            result += f"  ➤ {inp}\n"
    else:
        result += "  ➤ No specific inputs identified\n"
    
    result += "\n**Outputs:**\n"
    if all_outputs:
        for out in sorted(all_outputs):
            result += f"  ➤ {out}\n"
    else:
        result += "  ➤ No specific outputs identified\n"
    
    return result


def generate_dot_code(transcript: str, entities: Dict = None) -> Optional[str]:
    """
    Generate accurate Graphviz DOT code for process flowchart.
    
    Args:
        transcript: The meeting transcript text.
        entities: Pre-extracted entities.
        
    Returns:
        DOT code string or None if failed.
    """
    if entities is None:
        entities = extract_entities(transcript)
    
    entity_context = ""
    if entities["applications"]:
        entity_context = f"Applications used: {', '.join(entities['applications'])}\n"
    
    chunks = split_into_chunks(transcript, chunk_size=4000)
    
    print(f"    [Chunking] Processing {len(chunks)} chunks for flowchart...")
    
    # Step 1: Extract ordered process steps from each chunk
    all_steps = []
    
    for i, chunk in enumerate(chunks):
        print(f"    [Chunk {i+1}/{len(chunks)}] Extracting steps...")
        
        prompt = f"""Extract the PROCESS STEPS from this transcript section.

{entity_context}

Rules:
- Each step should be a clear action (verb + object)
- Include the application/system name if mentioned
- Keep steps concise (5-10 words each)
- Mark decision points with [DECISION] prefix
- Number each step in order

Examples of good steps:
1. Open Ivanti Service Management portal
2. Search for ticket using ticket number
3. [DECISION] Check if ticket status is Open
4. Update ticket status to In Progress
5. Add resolution notes to ticket

Transcript Section {i+1}:
{chunk}

Process Steps:"""

        response = llm_client.generate_with_stream(prompt)
        
        if response:
            for line in response.split('\n'):
                line = line.strip()
                step = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
                if step and len(step) > 5:
                    all_steps.append(step)
    
    if not all_steps:
        print("    [Warning] No steps extracted")
        return None
    
    # Remove duplicates while preserving order
    seen = set()
    unique_steps = []
    for step in all_steps:
        step_key = re.sub(r'[^a-zA-Z]', '', step.lower())
        if step_key not in seen and len(step_key) > 3:
            seen.add(step_key)
            unique_steps.append(step)
    
    # Limit to reasonable number
    if len(unique_steps) > 12:
        # Keep important steps: first 5, last 5, and 2 from middle
        middle_start = len(unique_steps) // 2 - 1
        unique_steps = unique_steps[:5] + unique_steps[middle_start:middle_start+2] + unique_steps[-5:]
    
    print(f"    [Final] Generating flowchart with {len(unique_steps)} steps...")
    
    # Step 2: Generate proper DOT code
    steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(unique_steps)])
    
    prompt = f"""Generate a Graphviz DOT flowchart from these process steps.

Process Steps:
{steps_text}

Requirements:
1. Start with a "Start" node (shape=oval, color=lightgreen)
2. End with an "End" node (shape=oval, color=lightcoral)
3. Regular steps use shape=box, color=lightblue
4. Steps marked [DECISION] use shape=diamond, color=gold
5. Decision nodes must have exactly 2 outgoing edges labeled "Yes" and "No"
6. All nodes must be connected - no orphan nodes
7. Use descriptive but concise labels (wrap long text with \\n)
8. Node IDs should be simple: Start, Step1, Step2, Decision1, End

Generate ONLY valid DOT code, no explanation:

digraph ProcessFlow {{
    // Graph settings
    rankdir=TB;
    node [fontname="Arial", fontsize=10];
    edge [fontname="Arial", fontsize=9];
    
    // Nodes
    ...
    
    // Edges
    ...
}}"""

    response = llm_client.generate_with_stream(prompt)
    
    if not response:
        return None
    
    # Extract DOT code
    match = re.search(r'```(?:dot|graphviz)?\n?(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'digraph\s+\w*\s*\{.*\}', response, re.DOTALL)
    if match:
        return match.group(0).strip()
    
    if 'digraph' in response:
        return response.strip()
    
    return None


def get_applications_table(transcript: str, entities: Dict = None) -> List[Dict]:
    """
    Extract applications/systems table data.
    
    Args:
        transcript: The meeting transcript.
        entities: Pre-extracted entities.
        
    Returns:
        List of application dictionaries.
    """
    if entities is None:
        entities = extract_entities(transcript)
    
    sample = transcript[:6000]
    
    apps_hint = ""
    if entities["applications"]:
        apps_hint = f"Known applications: {', '.join(entities['applications'])}\n"
    
    prompt = f"""From this transcript, extract information about each application/system used in the process.

{apps_hint}

For each application, provide:
- Application: Full correct name
- Interface: How it's accessed (Web Browser, Desktop App, API, etc.)
- Key Operation: Main URL or primary action performed
- Purpose: Brief description of its role in the process

Format as:
APP: [name] | INTERFACE: [type] | URL: [url or action] | PURPOSE: [description]

Transcript:
{sample}

Applications:"""

    response = llm_client.generate_with_stream(prompt)
    
    applications = []
    
    if response:
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('APP:'):
                parts = line.split('|')
                app = {
                    "application": "",
                    "interface": "",
                    "url": "",
                    "purpose": ""
                }
                for part in parts:
                    part = part.strip()
                    if part.startswith('APP:'):
                        app["application"] = part[4:].strip()
                    elif part.startswith('INTERFACE:'):
                        app["interface"] = part[10:].strip()
                    elif part.startswith('URL:'):
                        app["url"] = part[4:].strip()
                    elif part.startswith('PURPOSE:'):
                        app["purpose"] = part[8:].strip()
                
                if app["application"]:
                    applications.append(app)
    
    return applications if applications else [{"application": "", "interface": "", "url": "", "purpose": ""}]


def identify_key_timestamps(transcript: str, transcript_path: str) -> List[Dict]:
    """
    Identify timestamps where actual process actions occur.
    More accurate than keyword matching.
    
    Args:
        transcript: Full transcript text.
        transcript_path: Path to timestamped transcript file.
        
    Returns:
        List of {timestamp, action, description} dicts.
    """
    # Read timestamped transcript
    timestamped_lines = []
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)', line.strip())
                if match:
                    start = float(match.group(1))
                    text = match.group(3).strip()
                    timestamped_lines.append({"timestamp": start, "text": text})
    except:
        return []
    
    if not timestamped_lines:
        return []
    
    # Sample lines for LLM analysis (every Nth line to cover full video)
    step = max(1, len(timestamped_lines) // 30)
    sampled = timestamped_lines[::step][:30]
    
    lines_text = "\n".join([f"[{l['timestamp']:.1f}s] {l['text']}" for l in sampled])
    
    prompt = f"""Analyze these transcript lines and identify which ones describe KEY PROCESS ACTIONS that should have screenshots.

Good actions to capture (respond YES):
- Opening an application or website
- Clicking a button or menu
- Entering data in a form
- Submitting or saving something
- Viewing results or reports
- Decision points or validations

NOT good for screenshots (respond NO):
- General conversation or greetings
- Explaining what will happen (not doing it)
- Asking questions
- Describing background information

For each line, respond:
[timestamp] YES/NO - Brief action description (if YES)

Transcript lines:
{lines_text}

Analysis:"""

    response = llm_client.generate_with_stream(prompt)
    
    key_moments = []
    
    if response:
        for line in response.split('\n'):
            match = re.search(r'\[(\d+\.?\d*)s?\]\s*YES\s*[-:]\s*(.*)', line, re.IGNORECASE)
            if match:
                timestamp = float(match.group(1))
                description = match.group(2).strip()
                key_moments.append({
                    "timestamp": timestamp,
                    "description": description
                })
    
    # Limit to ~15 key moments, spread across video
    if len(key_moments) > 15:
        step = len(key_moments) // 15
        key_moments = key_moments[::step][:15]
    
    return key_moments


def paraphrase_transcript(text: str) -> str:
    """Paraphrase transcript text into professional description."""
    if len(text) > 400:
        text = text[:400]
    
    prompt = f"""Convert this transcript text into a professional step description (1-2 sentences).

Rules:
- Use third person (e.g., "The user clicks..." not "I click...")
- Be specific about the action
- Mention the application/field name if present
- Fix any transcription errors

Text: "{text}"

Step Description:"""

    response = llm_client.generate(prompt, timeout=60)
    return response.strip() if response else text


def paraphrase_batch(texts: List[str], batch_size: int = 5) -> List[str]:
    """Paraphrase multiple texts efficiently in batches."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        numbered = "\n".join([f"{j+1}. \"{t[:120]}\"" for j, t in enumerate(batch)])
        
        prompt = f"""Convert each transcript line into a professional step description.

Rules:
- Use third person ("The user..." not "I...")
- Be specific and concise (1 sentence each)
- Fix any obvious transcription errors
- Keep the same numbering

{numbered}

Professional Descriptions:"""
        
        response = llm_client.generate_with_stream(prompt)
        
        if response:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            for line in lines:
                clean = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
                clean = clean.strip('"')
                if clean:
                    results.append(clean)
        
        while len(results) < i + len(batch):
            idx = len(results) - i
            if idx < len(batch):
                results.append(batch[idx][:80])
            else:
                break
    
    return results


if __name__ == "__main__":
    if llm_client.is_available():
        print("✓ Ollama connected")
        
        test = """
        In this meeting we will discuss the Ivanti ticketing process.
        First, open the Ivanti Service Management portal in your browser.
        Navigate to the tickets section and search for open tickets.
        Select a ticket and update its status.
        """
        
        print("\nTesting entity extraction...")
        entities = extract_entities(test)
        print(f"Entities: {entities}")
        
        print("\nTesting project name...")
        name = get_project_name(test)
        print(f"Project: {name}")
    else:
        print("✗ Ollama not available")
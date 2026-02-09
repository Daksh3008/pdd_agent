# src/pdd_document.py
"""
PDD (Process Definition Document) Generator.
Creates professional Word documents with proper formatting.
"""

import os
from typing import List, Tuple, Dict
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Twips
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


class PDDGenerator:
    """Generator for Process Definition Documents."""
    
    def __init__(self):
        self.doc = Document()
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure document styles."""
        # Set default font
        style = self.doc.styles['Normal']
        style.font.name = 'Arial'
        style.font.size = Pt(11)
    
    def _add_heading(
        self,
        text: str,
        level: int = 1,
        color: str = "0D3B66",
        space_before: int = 12,
        space_after: int = 6
    ):
        """Add a styled heading with proper spacing."""
        heading = self.doc.add_paragraph()
        
        # Set spacing
        heading.paragraph_format.space_before = Pt(space_before)
        heading.paragraph_format.space_after = Pt(space_after)
        
        run = heading.add_run(text)
        run.bold = True
        run.font.name = 'Arial'
        
        # Set font size based on level
        sizes = {1: 14, 2: 12, 3: 11}
        run.font.size = Pt(sizes.get(level, 11))
        
        # Set color
        run.font.color.rgb = RGBColor.from_string(color)
        heading.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    
    def _add_paragraph(self, text: str, space_after: int = 8):
        """Add a paragraph with proper spacing."""
        para = self.doc.add_paragraph(text)
        para.paragraph_format.space_after = Pt(space_after)
        return para
    
    def _create_table(
        self,
        data: List[List[str]],
        header_color: str = "4F81BD",
        col_widths: List[float] = None
    ):
        """Create a styled table with proper formatting."""
        if not data:
            return
        
        table = self.doc.add_table(
            rows=len(data),
            cols=len(data[0]),
            style="Table Grid"
        )
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Set column widths if provided
        if col_widths:
            for i, width in enumerate(col_widths):
                for row in table.rows:
                    row.cells[i].width = Inches(width)
        
        for row_idx, row_data in enumerate(data):
            for col_idx, cell_text in enumerate(row_data):
                cell = table.cell(row_idx, col_idx)
                cell.text = str(cell_text)
                
                # Style cell paragraph
                for para in cell.paragraphs:
                    para.paragraph_format.space_before = Pt(4)
                    para.paragraph_format.space_after = Pt(4)
                
                if row_idx == 0:  # Header row
                    cell.paragraphs[0].alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    for run in cell.paragraphs[0].runs:
                        run.bold = True
                    
                    # Add shading
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:fill"), header_color)
                    tc_pr = cell._element.get_or_add_tcPr()
                    tc_pr.append(shading)
        
        # Add space after table
        self.doc.add_paragraph()
    
    def generate(
        self,
        project_name: str,
        process_summary: str,
        inputs_outputs: str,
        flowchart_path: str,
        document_purpose: str = None,
        applications: List[Dict] = None,
        output_path: str = "PDD_Document.docx"
    ) -> str:
        """
        Generate complete PDD document.
        
        Args:
            project_name: Name of the project.
            process_summary: Process summary text.
            inputs_outputs: Inputs and outputs text.
            flowchart_path: Path to flowchart image.
            document_purpose: Custom document purpose (optional).
            applications: List of application dicts (optional).
            output_path: Output document path.
            
        Returns:
            Path to generated document.
        """
        # ===== FRONT PAGE =====
        self.doc.add_paragraph("\n\n\n")
        
        title = self.doc.add_paragraph()
        title_run = title.add_run(project_name)
        title_run.bold = True
        title_run.font.size = Pt(28)
        title_run.font.color.rgb = RGBColor(13, 59, 102)  # Dark blue
        title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        self.doc.add_paragraph("\n")
        
        subtitle = self.doc.add_paragraph()
        sub_run = subtitle.add_run("Process Definition Document (PDD)")
        sub_run.font.size = Pt(18)
        sub_run.font.color.rgb = RGBColor(100, 100, 100)
        subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        
        self.doc.add_paragraph("\n\n\n")
        
        # Document info
        info = self.doc.add_paragraph()
        info.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        info_run = info.add_run("CONFIDENTIAL")
        info_run.font.size = Pt(12)
        info_run.font.color.rgb = RGBColor(150, 0, 0)
        
        self.doc.add_page_break()
        
        # ===== TABLE OF CONTENTS =====
        self._add_heading("TABLE OF CONTENTS", 1, space_before=0)
        
        toc_items = [
            "1. Document Control",
            "    1.1 Version Control",
            "    1.2 Document Review and Sign-Off",
            "2. General Process Description",
            "    2.1 Document Purpose",
            "    2.2 Process Summary",
            "    2.3 Applications Used",
            "    2.4 Inputs and Outputs",
            "    2.5 High-Level Process Flow",
            "3. Exception Handling",
            "    3.1 Known Exceptions",
            "    3.2 Unknown Exceptions",
            "4. Step-by-Step Process Documentation"
        ]
        
        for item in toc_items:
            para = self.doc.add_paragraph(item)
            para.paragraph_format.space_after = Pt(2)
        
        self.doc.add_page_break()
        
        # ===== SECTION 1: DOCUMENT CONTROL =====
        self._add_heading("1. DOCUMENT CONTROL", 1, space_before=0)
        
        self._add_heading("1.1 VERSION CONTROL", 2)
        self._create_table(
            [
                ["Version", "Date", "Description", "Author"],
                ["1.0", "", "Initial Draft", ""],
                ["", "", "", ""]
            ],
            col_widths=[1.0, 1.5, 2.5, 1.5]
        )
        
        note = self._add_paragraph(
            "** Document version should be updated when shared with primary stakeholders. **"
        )
        note.runs[0].italic = True
        
        self._add_heading("1.2 DOCUMENT REVIEW AND SIGN-OFF", 2)
        self._create_table(
            [
                ["Name", "Business Role", "Action", "Date"],
                ["", "", "", ""],
                ["", "", "", ""]
            ],
            col_widths=[2.0, 2.0, 1.5, 1.5]
        )
        
        self.doc.add_page_break()
        
        # ===== SECTION 2: GENERAL PROCESS DESCRIPTION =====
        self._add_heading("2. GENERAL PROCESS DESCRIPTION", 1, space_before=0)
        
        self._add_heading("2.1 DOCUMENT PURPOSE", 2)
        
        if document_purpose:
            # Use custom document purpose
            for para_text in document_purpose.split('\n\n'):
                if para_text.strip():
                    self._add_paragraph(para_text.strip())
        else:
            # Default purpose
            self._add_paragraph(
                f"The purpose of this Process Definition Document (PDD) is to capture the "
                f"business-related details of the {project_name} process. It describes how "
                f"the automated solution will operate and serves as a key input for the "
                f"technical design of the solution."
            )
            
            self._add_paragraph("This document ensures:")
            self._add_paragraph("• Process requirements are captured in line with organizational standards")
            self._add_paragraph("• Detailed information on the process flow and step-by-step procedures is provided")
            self._add_paragraph("• Stakeholders have a clear understanding of the expected results and objectives")
        
        self._add_heading("2.2 PROCESS SUMMARY", 2)
        
        # Handle multi-paragraph summaries
        for para_text in process_summary.split('\n\n'):
            if para_text.strip():
                self._add_paragraph(para_text.strip())
        
        self._add_heading("2.3 APPLICATIONS USED", 2)
        
        if applications:
            app_data = [["Application", "Interface", "Key Operation / URL", "Purpose"]]
            for app in applications:
                app_data.append([
                    app.get("application", ""),
                    app.get("interface", ""),
                    app.get("url", ""),
                    app.get("purpose", "")
                ])
            self._create_table(app_data, col_widths=[1.8, 1.3, 2.0, 1.8])
        else:
            self._create_table(
                [
                    ["Application", "Interface", "Key Operation / URL", "Purpose"],
                    ["", "", "", ""],
                    ["", "", "", ""]
                ],
                col_widths=[1.8, 1.3, 2.0, 1.8]
            )
        
        self._add_heading("2.4 INPUTS AND OUTPUTS", 2)
        
        # Format inputs/outputs properly
        for line in inputs_outputs.split('\n'):
            line = line.strip()
            if line:
                if line.startswith('**') and line.endswith('**'):
                    # Bold header
                    para = self.doc.add_paragraph()
                    run = para.add_run(line.strip('*'))
                    run.bold = True
                    para.paragraph_format.space_before = Pt(8)
                    para.paragraph_format.space_after = Pt(4)
                elif line.startswith('➤') or line.startswith('-'):
                    self._add_paragraph(line, space_after=2)
                else:
                    self._add_paragraph(line)
        
        # Add spacing before flowchart
        self.doc.add_paragraph()
        
        self._add_heading("2.5 HIGH-LEVEL PROCESS FLOW", 2)
        
        if flowchart_path and os.path.exists(flowchart_path):
            # Add flowchart with proper sizing
            self.doc.add_picture(flowchart_path, width=Inches(6.5))
            
            # Center the image
            last_para = self.doc.paragraphs[-1]
            last_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            
            # Add caption
            caption = self.doc.add_paragraph()
            caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            cap_run = caption.add_run(f"Figure 1: {project_name} Process Flow")
            cap_run.italic = True
            cap_run.font.size = Pt(10)
        else:
            self._add_paragraph("[Flowchart image not available]")
        
        # Important: Add proper spacing after flowchart
        self.doc.add_paragraph()
        self.doc.add_paragraph()
        
        self.doc.add_page_break()
        
        # ===== SECTION 3: EXCEPTION HANDLING =====
        self._add_heading("3. EXCEPTION HANDLING", 1, space_before=0)
        
        self._add_paragraph(
            "This section documents the different types of exceptions that may occur "
            "during process execution and how they should be handled."
        )
        
        self._add_heading("3.1 KNOWN EXCEPTIONS", 2)
        self._add_paragraph(
            "The following table lists process exceptions that have been identified "
            "and their corresponding handling procedures:"
        )
        
        self._create_table(
            [
                ["Exception Code", "Description", "Handling Action"],
                ["EXC_001", "", ""],
                ["EXC_002", "", ""],
                ["EXC_003", "", ""]
            ],
            col_widths=[1.5, 3.0, 2.5]
        )
        
        self._add_heading("3.2 UNKNOWN EXCEPTIONS", 2)
        self._add_paragraph(
            "Unknown exceptions are errors that may occur during processing that were "
            "not previously identified. These will be logged and escalated according "
            "to the standard exception handling procedure."
        )
        
        self._create_table(
            [
                ["Exception Code", "Description", "Handling Action"],
                ["ERR_UNKNOWN", "Unexpected system error", "Log error, notify support team"],
                ["ERR_TIMEOUT", "Application timeout", "Retry operation, escalate if persistent"]
            ],
            col_widths=[1.5, 3.0, 2.5]
        )
        
        self.doc.add_page_break()
        
        # ===== SECTION 4: STEP-BY-STEP =====
        self._add_heading("4. STEP-BY-STEP PROCESS DOCUMENTATION", 1, space_before=0)
        
        self._add_paragraph(
            "This section provides detailed step-by-step documentation of the process, "
            "including screenshots captured during process execution."
        )
        
        self._add_heading("4.1 DETAILED PROCESS STEPS", 2)
        
        # Save document
        self.doc.save(output_path)
        print(f"PDD Document saved to: {output_path}")
        return output_path
    
    def append_frames_with_text(
        self,
        doc_path: str,
        frame_text_pairs: List[Tuple[str, str]],
        start_step: int = 1
    ):
        """
        Append frames and descriptions to existing document.
        
        Args:
            doc_path: Path to existing document.
            frame_text_pairs: List of (frame_path, description) tuples.
            start_step: Starting step number.
        """
        if not os.path.exists(doc_path):
            print(f"Document not found: {doc_path}")
            return
        
        doc = Document(doc_path)
        
        for i, (frame_path, text) in enumerate(frame_text_pairs):
            step_num = start_step + i
            
            # Add step heading
            step_heading = doc.add_paragraph()
            step_heading.paragraph_format.space_before = Pt(16)
            run = step_heading.add_run(f"Step {step_num}")
            run.bold = True
            run.font.size = Pt(11)
            
            # Add image if exists
            if os.path.exists(frame_path):
                doc.add_picture(frame_path, width=Inches(5.5))
                last_para = doc.paragraphs[-1]
                last_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                last_para.paragraph_format.space_after = Pt(8)
            
            # Add description
            desc = doc.add_paragraph(text)
            desc.paragraph_format.space_after = Pt(12)
            
            # Add separator line (optional)
            if i < len(frame_text_pairs) - 1:
                separator = doc.add_paragraph()
                separator.paragraph_format.space_after = Pt(8)
        
        doc.save(doc_path)
        print(f"Appended {len(frame_text_pairs)} steps to document")


if __name__ == "__main__":
    # Test document generation
    generator = PDDGenerator()
    generator.generate(
        project_name="Ivanti Ticket Management",
        process_summary="This process handles the management of IT tickets in the Ivanti system.",
        inputs_outputs="**Inputs:**\n  ➤ Ticket ID\n  ➤ User credentials\n\n**Outputs:**\n  ➤ Updated ticket",
        flowchart_path="",
        document_purpose="This document describes the Ivanti ticket management automation process.",
        output_path="test_pdd.docx"
    )
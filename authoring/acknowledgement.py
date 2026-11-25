from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

root = Path(__file__).resolve().parents[1]
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.paragraph_format.space_after = Pt(10)
st.paragraph_format.line_spacing = 1.15
section = doc.sections[0]
section.top_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)
q = doc.add_paragraph()
r = q.add_run("Generative AI acknowledgement")
r.bold = True
r.font.size = Pt(21)
r.font.color.rgb = RGBColor.from_string("422333")
doc.add_paragraph("SYDE 411  |  Group 28")
doc.add_paragraph("Pruning neural networks for traffic sign recognition")
doc.add_paragraph("Shahmeer Ali, Amin Ambike and Brayden Petersen")
doc.add_paragraph(
    "We used ChatGPT to find research papers and draft the project proposal and report. It also assisted with writing the Python code, running the experiments, checking numerical results, and preparing figures and presentation materials. The reported measurements came from the saved models and experiment logs."
)
doc.add_paragraph(
    "OpenAI. (2026). ChatGPT [large language model]. https://chatgpt.com/"
)
doc.save(root / "deliverables/Group28_AI_Acknowledgement.docx")

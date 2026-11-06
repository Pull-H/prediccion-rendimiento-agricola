"""Render the evidence-based article to Markdown, native Word and PDF.

Uses the same six-table/six-figure catalogue for all formats and the web module.
"""
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from matplotlib import font_manager
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path("docs/manuscrito")


def plain(text):
    return re.sub(r"[*`]", "", text)


def reference_text(record):
    authors = []
    for author in record["authors"]:
        initials = " ".join(part[0] + "." for part in author.get("given", "").split() if part)
        authors.append(author["family"] + (", " + initials if initials else ""))
    if len(authors) > 20:
        authors = authors[:19] + ["…", authors[-1]]
    year = record.get("year") or record["published"]["date-parts"][0][0]
    title = record["title"][0]
    if record.get("doi") == "10.1145/2939672.2939785":
        title = "XGBoost: A Scalable Tree Boosting System"
    journal = record["journal"][0] if record.get("journal") else ""
    location = str(record.get("volume") or "")
    if record.get("issue"): location += f"({record['issue']})"
    if record.get("page"): location += f", {record['page']}"
    elif record.get("article_number"): location += f", artículo {record['article_number']}"
    url = f"https://doi.org/{record['doi']}" if record.get("doi") else record["url"]
    return f"{'; '.join(authors)} ({year}). {title}. {journal}{', ' + location if location else ''}. {url}"


def blocks(template, catalogue):
    paragraphs = re.split(r"\n\s*\n", template.strip())
    for block in paragraphs:
        match = re.fullmatch(r"\{\{(TABLE|FIGURE)_(\d)\}\}", block.strip())
        if match:
            kind, number = match.group(1), int(match.group(2))
            yield kind.lower(), next(item for item in catalogue["tables" if kind == "TABLE" else "figures"] if item["id"] == number)
        elif block == "{{REFERENCES}}":
            for text in catalogue["reference_texts"]: yield "reference", text
        elif block.startswith("#"):
            mark, text = block.split(" ", 1)
            yield "heading", (len(mark), text)
        else:
            yield "paragraph", block.replace("\n", " ")


def render_markdown(content):
    output = []
    for kind, value in content:
        if kind == "heading": output.append("#" * value[0] + " " + value[1])
        elif kind in {"paragraph", "reference"}: output.append(value)
        elif kind == "table":
            output.append(f"**Tabla {value['id']}. {value['title']}**")
            output.append("| " + " | ".join(value["headers"]) + " |\n| " + " | ".join(["---"] * len(value["headers"])) + " |\n" +
                          "\n".join("| " + " | ".join(row) + " |" for row in value["rows"]))
            output.append("*Nota.* " + value["note"])
        else:
            output.append(f"![Figura {value['id']}. {value['title']}](figuras/{value['png']})")
            output.append(f"**Figura {value['id']}. {value['title']}.** {value['caption']}")
    (ROOT / "articulo_trigo_yieldsat.md").write_text("\n\n".join(output) + "\n", encoding="utf-8")


def doc_paragraph(doc, text, style=None):
    p = doc.add_paragraph(style=style)
    for chunk in re.split(r"(\*\*.*?\*\*)", text):
        run = p.add_run(plain(chunk)); run.bold = chunk.startswith("**")
    return p


def render_docx(content):
    doc = Document(); section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(11)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    header = section.header.paragraphs[0]
    header.add_run("Trigo, evaluación temporal y trazabilidad · Manuscrito de investigación").font.size = Pt(8)
    footer = section.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Página ")
    field = OxmlElement("w:fldSimple"); field.set(qn("w:instr"), "PAGE"); footer._p.append(field)
    no_compression = OxmlElement("w:doNotAutoCompressPictures")
    doc.settings.element.append(no_compression)
    for kind, value in content:
        if kind == "heading": doc.add_heading(value[1], level=0 if value[0] == 1 else min(value[0]-1, 3))
        elif kind in {"paragraph", "reference"}:
            p=doc_paragraph(doc,value)
            if kind == "reference":
                p.paragraph_format.left_indent=Cm(.6); p.paragraph_format.first_line_indent=Cm(-.6)
                for run in p.runs: run.font.size=Pt(9)
        elif kind == "table":
            caption=doc_paragraph(doc,f"**Tabla {value['id']}. {value['title']}**")
            caption.paragraph_format.keep_with_next=True
            table=doc.add_table(rows=1,cols=len(value["headers"])); table.style="Light Shading Accent 1"
            for cell,title in zip(table.rows[0].cells,value["headers"]): cell.text=title
            properties=table.rows[0]._tr.get_or_add_trPr(); repeat=OxmlElement("w:tblHeader"); properties.append(repeat)
            for row in value["rows"]:
                for cell,text in zip(table.add_row().cells,row): cell.text=text
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
                        p.paragraph_format.space_after=Pt(3)
                        for run in p.runs: run.font.size=Pt(8)
            note=doc_paragraph(doc,"Nota. "+value["note"])
            for run in note.runs: run.font.size=Pt(8.5)
        elif kind == "figure":
            caption=doc_paragraph(doc,f"**Figura {value['id']}. {value['title']}**")
            caption.paragraph_format.keep_with_next=True
            p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(ROOT/"figuras"/value["png"]),width=Cm(16.5))
            note=doc_paragraph(doc,value["caption"])
            for run in note.runs: run.font.size=Pt(9)
    doc.core_properties.title="Evaluación temporal del rendimiento de trigo con satélite y suelo"
    doc.core_properties.subject="Piloto retrospectivo de Alemania; seis tablas y seis figuras"
    doc.core_properties.author="Autores por completar · Universidad Nacional de Trujillo"
    assert len(doc.tables)==6 and len(doc.inline_shapes)==6
    doc.save(ROOT/"articulo_trigo_yieldsat.docx")


def render_pdf(content):
    for name, weight in (("Research", "normal"), ("ResearchBold", "bold")):
        path=font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",weight=weight))
        pdfmetrics.registerFont(TTFont(name,path))
    pdfmetrics.registerFontFamily("Research",normal="Research",bold="ResearchBold",italic="Research",boldItalic="ResearchBold")
    styles=getSampleStyleSheet()
    body=ParagraphStyle("ResearchBody",fontName="Research",fontSize=9.3,leading=13,alignment=TA_JUSTIFY,spaceAfter=8)
    small=ParagraphStyle("ResearchSmall",parent=body,fontSize=7.8,leading=10,alignment=TA_LEFT)
    caption=ParagraphStyle("ResearchCaption",parent=small,fontName="ResearchBold",spaceBefore=8,spaceAfter=5)
    heading=ParagraphStyle("ResearchHeading",parent=body,fontName="ResearchBold",fontSize=12,leading=15,spaceBefore=12,keepWithNext=True)
    title=ParagraphStyle("ResearchTitle",parent=heading,fontSize=17,leading=21,spaceBefore=0,spaceAfter=16)
    def text(value):
        escaped=html.escape(value)
        escaped=re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escaped)
        return escaped.replace("`", "")
    story=[]; width=A4[0]-4*28.3465
    for kind,value in content:
        if kind=="heading": story.append(Paragraph(text(value[1]),title if value[0]==1 else heading))
        elif kind in {"paragraph","reference"}: story.append(Paragraph(text(value),small if kind=="reference" else body))
        elif kind=="table":
            story.append(Paragraph(f"Tabla {value['id']}. {text(value['title'])}",caption))
            cells=[[Paragraph(text(cell),small) for cell in row] for row in [value["headers"]]+value["rows"]]
            n=len(value["headers"])
            fractions = [.23]+[(.77/(n-1))]*(n-1)
            if value["id"]==3: fractions=[.17,.06,.55,.10,.12]
            elif value["id"]==2:fractions=[.16,.36,.10,.22,.16]
            elif value["id"]==1:fractions=[.11,.17,.08,.09,.14,.13,.14,.14]
            table=LongTable(cells,colWidths=[width*f for f in fractions],repeatRows=1,hAlign="LEFT")
            table.setStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5eee7")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#d4dfd6")),
                ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)])
            story.extend([table,Spacer(1,5),Paragraph("Nota. "+text(value["note"]),small),Spacer(1,8)])
        elif kind=="figure":
            path=ROOT/"figuras"/value["png"]
            with PILImage.open(path) as image: w,h=image.size
            size=width; height=size*h/w
            story.append(KeepTogether([Paragraph(f"Figura {value['id']}. {text(value['title'])}",caption),
                Image(str(path),width=size,height=height),Paragraph(text(value["caption"]),small),Spacer(1,10)]))
    def footer(canvas,doc):
        canvas.setFont("Research",7);canvas.drawRightString(A4[0]-56,28,f"Universidad Nacional de Trujillo · {doc.page}")
    pdf=SimpleDocTemplate(str(ROOT/"articulo_trigo_yieldsat.pdf"),pagesize=A4,
        rightMargin=56.7,leftMargin=56.7,topMargin=47,bottomMargin=47,
        title="Evaluación temporal del rendimiento de trigo con satélite y suelo",
        author="Autores por completar · Universidad Nacional de Trujillo")
    pdf.build(story,onFirstPage=footer,onLaterPages=footer)


def final_catalogue(catalogue):
    files={}
    candidates=[path for path in ROOT.iterdir() if path.suffix in {".md",".pdf",".docx",".xlsx",".json",".ris",".csv"}
                and path.name not in {"catalogo_base.json","catalogo_publicacion.json","evidencia_modelos.json","manuscrito_base.md"}]
    candidates += list((ROOT/"figuras").glob("*")) + list((ROOT/"tablas").glob("*.csv"))
    for path in candidates:
        if not path.is_file(): continue
        files[path.name]={"path":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
    bundle=ROOT/"figuras_y_tablas_600dpi.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in list((ROOT/"figuras").glob("*"))+list((ROOT/"tablas").glob("*.csv")):
            archive.write(path,path.relative_to(ROOT).as_posix())
        archive.write(ROOT/"tablas_articulo.xlsx","tablas_articulo.xlsx")
    files[bundle.name]={"path":bundle.name,"sha256":hashlib.sha256(bundle.read_bytes()).hexdigest(),"bytes":bundle.stat().st_size}
    catalogue.pop("reference_texts",None)
    catalogue["files"]=files
    catalogue["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    catalogue["editorial_pending"]=["Nombres, ORCID y correspondencia de autores", "Facultad/departamento", "Revista y estilo editorial final", "Declaraciones de financiación, conflictos y términos de difusión"]
    (ROOT/"catalogo_publicacion.json").write_text(json.dumps(catalogue,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    catalogue=json.loads((ROOT/"catalogo_base.json").read_text(encoding="utf-8"))
    references=json.loads((ROOT/"referencias_verificadas.json").read_text(encoding="utf-8"))["references"]
    # Do not assert NeurIPS page metadata that was not retrieved from its primary record.
    references["vaswani2017"].pop("page",None)
    ordered=sorted(references.values(),key=lambda record:record["authors"][0]["family"])
    catalogue["reference_texts"]=[reference_text(record) for record in ordered]
    ris=[]
    for record in ordered:
        year=record.get("year") or record["published"]["date-parts"][0][0]
        ris.extend(["TY  - JOUR","TI  - "+record["title"][0],"PY  - "+str(year)])
        for author in record["authors"]: ris.append("AU  - "+author["family"]+(", "+author["given"] if author.get("given") else ""))
        if record.get("doi"):ris.append("DO  - "+record["doi"])
        ris.extend(["UR  - "+record.get("url", ""),"ER  - ",""])
    (ROOT/"referencias.ris").write_text("\n".join(ris),encoding="utf-8")
    template=(ROOT/"manuscrito_base.md").read_text(encoding="utf-8")
    content=list(blocks(template,catalogue))
    assert sum(kind=="table" for kind,_ in content)==6
    assert sum(kind=="figure" for kind,_ in content)==6
    assert [value["id"] for kind,value in content if kind=="table"]==list(range(1,7))
    render_markdown(content);render_docx(content);render_pdf(content);final_catalogue(catalogue)
    print(f"Manuscript generated: six native tables, six embedded figures, {len(ordered)} references; Markdown, DOCX and PDF",flush=True)


if __name__=="__main__":main()

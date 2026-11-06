from datetime import datetime, timezone
from io import BytesIO
from textwrap import wrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from docx import Document
from openpyxl import Workbook

def report_bytes(kind: str, ext: str, lines: list[str]) -> bytes:
    title = f"Informe {kind} - Predicción agrícola"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [title, f"Generado: {stamp}", *lines]
    buffer = BytesIO()
    if ext == "pdf":
        pdf = canvas.Canvas(buffer, pagesize=A4)
        pdf.setFont("Helvetica", 9)
        y = 800
        for line in rows:
            safe = str(line).encode("latin-1", "replace").decode("latin-1")
            for part in wrap(safe, width=105, break_long_words=True) or [""]:
                if y < 50:
                    pdf.showPage(); pdf.setFont("Helvetica", 9); y = 800
                pdf.drawString(45, y, part)
                y -= 14
            y -= 6
        pdf.save()
    elif ext == "docx":
        doc = Document()
        doc.add_heading(title, 0)
        for line in rows[1:]:
            doc.add_paragraph(str(line))
        doc.save(buffer)
    elif ext == "xlsx":
        book = Workbook()
        sheet = book.active
        sheet.title = kind.capitalize()
        for row in rows:
            sheet.append([str(row)])
        book.save(buffer)
    else:
        raise ValueError(ext)
    return buffer.getvalue()

"""Model-specific scientific exports sharing one verified evaluation payload."""
import json
from io import BytesIO


def render_report(result, ext):
    if ext == "json":
        return json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    from .ml.figures import figure_bytes
    title = f"Evaluación del modelo {result['model']['id']} {result['model']['type']}"
    notes = [f"Dataset: {result['dataset']['name']}. Fuente: {result['dataset']['source']}.",
             f"Licencia: {result['dataset']['license'] or 'no declarada'}.",
             "DEMOSTRACIÓN SINTÉTICA SIN VALIDEZ CIENTÍFICA" if result['dataset']['synthetic'] else "Evaluación sobre datos importados.",
             result["interpretation"], result["importance"]["interpretation"],
             "Residuos positivos indican subestimación; negativos, sobreestimación.",
             "En observado/predicho, la diagonal representa coincidencia perfecta.",
             *result["warnings"]]
    metrics = [[key, str(value), str(result['baseline'].get(key))]
               for key, value in result["metrics"].items()]
    stream = BytesIO()
    if ext == "xlsx":
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        book = Workbook()
        summary = book.active
        summary.title = "Resumen"
        summary.append([title])
        for line in notes:
            summary.append([line])
        summary.append(["Métrica", "Modelo", "Referencia media"])
        for row in metrics:
            summary.append(row)
        for name, rows in (("Predicciones prueba", result["points"]), ("Desglose", result["groups"]),
                           ("Importancia", result["importance"]["values"])):
            sheet = book.create_sheet(name)
            if rows:
                sheet.append(list(rows[0]))
                for row in rows:
                    sheet.append([row[key] for key in rows[0]])
        protocol = book.create_sheet("Protocolo")
        for key, value in {**result["protocol"], "interval": result["interval"], "confidence": result["confidence"]}.items():
            # Excel cells cap strings at 32767 chars; partition long protocol values.
            text = json.dumps(value, ensure_ascii=False)
            for start in range(0, len(text), 30000):
                protocol.append([key if start == 0 else f"{key} continuación", text[start:start + 30000]])
        for sheet in book:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="245B44")
            for cells in sheet.columns:
                sheet.column_dimensions[cells[0].column_letter].width = min(70, max(18, max(len(str(c.value or "")) for c in cells) + 2))
            # Treat all strings as text, including user-supplied names beginning '='.
            for row in sheet:
                for cell in row:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
        book.save(stream)
    elif ext == "docx":
        from docx import Document
        from docx.shared import Inches
        document = Document()
        document.add_heading(title, 0)
        for note in notes:
            document.add_paragraph(note)
        table = document.add_table(rows=1, cols=3)
        table.style = "Light Shading Accent 1"
        for cell, label in zip(table.rows[0].cells, ["Métrica", "Modelo", "Referencia"]):
            cell.text = label
        for row in metrics:
            for cell, value in zip(table.add_row().cells, row):
                cell.text = value
        document.add_heading("Incertidumbre", level=1)
        document.add_paragraph(json.dumps({"confidence": result["confidence"], "interval": result["interval"]}, ensure_ascii=False))
        for kind, caption in (("observed", "Observado y predicho en prueba"), ("residuals", "Residuos en prueba")):
            document.add_heading(caption, level=1)
            document.add_picture(BytesIO(figure_bytes(result, kind)), width=Inches(6))
        document.add_heading("Protocolo reproducible", level=1)
        # Omit large index lists here; complete protocol remains in JSON export.
        for key in ("run_id", "source_sha256", "seed", "validation", "best_parameters", "hyperparameters", "versions"):
            document.add_paragraph(f"{key}: {json.dumps(result['protocol'].get(key), ensure_ascii=False)}")
        document.save(stream)
    elif ext == "pdf":
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from xml.sax.saxutils import escape
        styles = getSampleStyleSheet()
        story = [Paragraph(escape(title), styles["Title"])]
        story += [Paragraph(escape(note), styles["BodyText"]) for note in notes]
        table = Table([["Métrica", "Modelo", "Referencia"], *metrics], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245b44")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey)]))
        story += [Spacer(1, 12), table]
        for kind in ("observed", "residuals"):
            story += [Spacer(1, 12), Image(BytesIO(figure_bytes(result, kind)), width=450, height=289)]
        for key in ("confidence", "interval", "protocol"):
            value = result[key]
            if key == "protocol":
                value = {k: v for k, v in value.items() if k != "split_info"}
            story.append(Paragraph(escape(json.dumps(value, ensure_ascii=False)), styles["BodyText"]))
        SimpleDocTemplate(stream, title=title).build(story)
    else:
        raise ValueError("Unsupported format")
    return stream.getvalue()

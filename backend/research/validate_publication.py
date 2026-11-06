"""Validate delivered manuscript counts, actual image resolution and model evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from docx import Document
from PIL import Image
from pypdf import PdfReader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root=Path("docs/manuscrito")
catalogue=json.loads((root/"catalogo_publicacion.json").read_text(encoding="utf-8"))
evidence=json.loads((root/"evidencia_modelos.json").read_text(encoding="utf-8"))
assert len(catalogue["figures"])==len(catalogue["tables"])==6
for name, entry in catalogue["files"].items():
    path=root/entry["path"]
    assert path.is_file() and path.stat().st_size==entry["bytes"], name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"], name
for figure in catalogue["figures"]:
    for extension in ("png","tiff"):
        with Image.open(root/"figuras"/figure[extension]) as image:
            assert image.size==(figure["width_px"],figure["height_px"])
            assert image.width>=4000
            assert all(abs(float(dpi)-600)<1 for dpi in image.info["dpi"])
doc=Document(root/"articulo_trigo_yieldsat.docx")
assert len(doc.tables)==6 and len(doc.inline_shapes)==6
for table, declared in zip(doc.tables,catalogue["tables"]):
    assert [cell.text for cell in table.rows[0].cells]==declared["headers"]
    assert len(table.rows)-1==len(declared["rows"])
markdown=(root/"articulo_trigo_yieldsat.md").read_text(encoding="utf-8")
assert "{{" not in markdown
for number in range(1,7):
    assert markdown.count(f"**Tabla {number}.")==1
    assert markdown.count(f"**Figura {number}.")==1
pdf=PdfReader(root/"articulo_trigo_yieldsat.pdf")
text="\n".join(page.extract_text() for page in pdf.pages)
assert len(pdf.pages)>10 and "Universidad Nacional de Trujillo" in text
for number in range(1,7):
    assert f"Tabla {number}." in text and f"Figura {number}." in text
frame=pd.read_csv("data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.csv")
indices=[record["row"] for record in evidence["records"] if record["role"]=="test"]
y=frame.iloc[indices].yield_t_ha.to_numpy()
assert len(y)==37
assert abs(frame[frame.campaign<=2019].yield_t_ha.mean()-evidence["baseline_value"])<1e-10
for name,values in evidence["predictions"].items():
    predictions=np.array(values)[indices]
    metrics=evidence["metrics"][name]["test"]
    assert abs(mean_absolute_error(y,predictions)-metrics["mae"])<1e-8
    assert abs(np.sqrt(mean_squared_error(y,predictions))-metrics["rmse"])<1e-8
    assert abs(r2_score(y,predictions)-metrics["r2"])<1e-8
for model in evidence["permutation"].values():
    assert model["evaluation_role"]=="validation" and model["n"]==15
    for group in model["groups"].values():
        assert len(group["replicates"])==30
        assert abs(np.mean(group["replicates"])-group["mean_delta_mae"])<1e-10
with ZipFile(root/"figuras_y_tablas_600dpi.zip") as archive:
    assert len([name for name in archive.namelist() if name.endswith("_600dpi.png")])==6
    assert len([name for name in archive.namelist() if name.endswith("_600dpi.tiff")])==6
result={"figures":6,"tables":6,"native_word_tables":len(doc.tables),"embedded_word_figures":len(doc.inline_shapes),
        "pdf_pages":len(pdf.pages),"image_dpi":600,"verified_files":len(catalogue["files"]),
        "reference_records":17,"model_predictions_recomputed":6,"permutation_groups":7,"permutation_repetitions":30}
Path("data/publication_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))

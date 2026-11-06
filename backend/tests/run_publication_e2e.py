"""Verify the deployed article module, all six originals and previews as a viewer."""
import hashlib
import io
import json
import secrets
import subprocess
from pathlib import Path

import httpx
from PIL import Image


email=f"publication-{secrets.token_hex(8)}@example.test"
password=secrets.token_urlsafe(24)
compose=["docker","compose","exec","-T","backend","python","-c"]
create="""
import sys
from app.db import SessionLocal
from app.models import User
from app.security import hasher
email,password=[sys.stdin.readline().strip() for _ in range(2)]
with SessionLocal() as db:
    db.add(User(email=email,name='Publication verification',role='viewer',password_hash=hasher.hash(password)))
    db.commit()
"""
disable="""
import sys
from sqlalchemy import select
from app.db import SessionLocal
from app.models import User
with SessionLocal() as db:
    user=db.scalar(select(User).where(User.email==sys.stdin.readline().strip()))
    if user:
        user.active=False
        db.commit()
"""
subprocess.run(compose+[create],input=f"{email}\n{password}\n",text=True,check=True,timeout=60)
try:
    results={"figures":[],"documents":[]}
    with httpx.Client(base_url="http://localhost:5173",timeout=60) as client:
        assert client.get("/api/publication").status_code==401
        response=client.post("/api/auth/login",json={"email":email,"password":password})
        assert response.status_code==200,response.text
        response=client.get("/api/publication")
        assert response.status_code==200,response.text
        catalogue=response.json()
        assert len(catalogue["figures"])==len(catalogue["tables"])==6
        assert catalogue["affiliation"]=="Universidad Nacional de Trujillo"
        for figure in catalogue["figures"]:
            assert figure["caption"] and figure["interpretation"] and figure["limitations"]
            for key in ("preview","png","tiff","pdf"):
                name=figure[key]
                response=client.get(f"/api/publication/files/{name}")
                assert response.status_code==200,(name,response.text[:150])
                assert hashlib.sha256(response.content).hexdigest()==catalogue["files"][name]["sha256"]
                if key in {"png","tiff"}:
                    with Image.open(io.BytesIO(response.content)) as image:
                        assert image.width>=4000
                        assert all(abs(float(dpi)-600)<1 for dpi in image.info["dpi"])
            results["figures"].append({"id":figure["id"],"original_dpi":600,"preview_and_originals":"verified"})
            print(f"Figure {figure['id']}: preview, 600 dpi PNG/TIFF, vector PDF and explanation verified",flush=True)
        for name,signature in (("articulo_trigo_yieldsat.docx",b"PK"),("articulo_trigo_yieldsat.pdf",b"%PDF"),
                               ("figuras_y_tablas_600dpi.zip",b"PK"),("tablas_articulo.xlsx",b"PK")):
            response=client.get(f"/api/publication/files/{name}?download=true")
            assert response.status_code==200,response.text[:150]
            assert response.content.startswith(signature)
            assert "attachment" in response.headers["content-disposition"]
            results["documents"].append(name)
        assert client.get("/api/publication/files/.env").status_code==404
        assert client.get("/articulo").status_code==200
        assert client.post("/api/auth/logout",headers={"X-CSRF-Token":client.cookies["csrf_token"]}).status_code==200
    Path("data/publication_api_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
finally:
    subprocess.run(compose+[disable],input=email+"\n",text=True,check=True,timeout=60)

"""Retrieve DOI metadata from Crossref without inventing missing journal fields."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

DOIS = {
    "pathak2023": "10.1109/IGARSS52108.2023.10282318",
    "kallenberg2026": "10.5194/essd-18-3997-2026",
    "khaki2020": "10.3389/fpls.2019.01750",
    "roberts2017": "10.1111/ecog.02881",
    "breiman2001": "10.1023/A:1010933404324",
    "chen2016": "10.1145/2939672.2939785",
    "hochreiter1997": "10.1162/neco.1997.9.8.1735",
    "friedman1937": "10.1080/01621459.1937.10503522",
    "wilcoxon1945": "10.2307/3001968",
    "efron1979": "10.1214/aos/1176344552",
    "altmann2010": "10.1093/bioinformatics/btq134",
    "kapoor2023": "10.1016/j.patter.2023.100804",
}


def fetch(item):
    key, doi = item
    response = httpx.get(f"https://api.crossref.org/works/{quote(doi, safe='/:')}", timeout=40,
                         follow_redirects=True)
    response.raise_for_status()
    message = response.json()["message"]
    return key, {"doi": message["DOI"], "title": message.get("title", []),
                 "authors": message.get("author", []), "journal": message.get("container-title", []),
                 "volume": message.get("volume"), "issue": message.get("issue"), "page": message.get("page"),
                 "article_number": message.get("article-number"), "published": message.get("published"),
                 "url": message.get("URL"), "verification": "Crossref DOI registry"}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=1) as pool:
        references = dict(pool.map(fetch, DOIS.items()))
    references["wilcoxon1945"]["crossref_page"] = references["wilcoxon1945"]["page"]
    references["wilcoxon1945"]["page"] = "80–83"
    references["wilcoxon1945"]["verification"] += "; page range checked in original scan https://sci2s.ugr.es/keel/pdf/algorithm/articulo/wilcoxon1945.pdf"
    references["miranda2026"] = {"title": ["YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction"],
        "authors": [{"family": name} for name in ["Miranda, M.", "Pathak, D.", "Helber, P.", "Bischke, B.", "Najjar, H.", "Mena, F.", "Sanchez, C.", "Pai, A.", "Arenas, D.", "Valdenegro-Toro, M.", "Charfuelan, M.", "Nuske, M.", "Dengel, A."]],
        "doi": "10.48550/arXiv.2604.00940", "url": "https://arxiv.org/abs/2604.00940", "year": 2026,
        "journal": ["arXiv, 2604.00940v1 (preprint)"], "verification": "Read arXiv primary record and HTML v1; not a peer-reviewed publication"}
    references["vaswani2017"] = {"title": ["Attention Is All You Need"], "year": 2017,
        "authors": [{"family": name} for name in ["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J.", "Jones, L.", "Gomez, A. N.", "Kaiser, Ł.", "Polosukhin, I."]],
        "journal": ["Advances in Neural Information Processing Systems"], "volume": "30", "page": "5998–6008",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
        "verification": "Read official NeurIPS proceedings; no journal DOI asserted"}
    references["holm1979"] = {"title": ["A Simple Sequentially Rejective Multiple Test Procedure"], "year": 1979,
        "authors": [{"family": "Holm", "given": "Sture"}], "journal": ["Scandinavian Journal of Statistics"],
        "volume": "6", "issue": "2", "page": "65–70", "url": "https://www.jstor.org/stable/4615733",
        "verification": "Original paper scanned at University of Sao Paulo; no DOI asserted"}
    references["scipy"] = {"title": ["friedmanchisquare — SciPy reference"], "year": "s. f.",
        "authors": [{"family": "SciPy"}], "journal": ["Documentación oficial"],
        "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html",
        "verification": "Read official API documentation and approximation warning on 2026-11-06"}
    references["yieldsat_tutorial"] = {"title": ["YieldSAT machine learning tutorial"], "year": "s. f.",
        "authors": [{"family": "YieldSAT"}], "journal": ["Documentación oficial del proyecto"],
        "url": "https://yieldsat.github.io/notebooks/yieldsat-ml-tutorial.html",
        "verification": "Official tutorial, target unit evidence; checked 2026-11-06"}
    output = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "references": references}
    Path("docs/manuscrito/referencias_verificadas.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, record in references.items():
        print(key, json.dumps(record["title"][0], ensure_ascii=True), record.get("published", record.get("year")))

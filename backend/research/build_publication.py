"""Build six scientific figures at 600 dpi and six tables from frozen evidence.

Run from the repository root using the local research environment.
No fitting, target alteration or invented observations is performed.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from openpyxl import Workbook

ROOT = Path("docs/manuscrito")
FIGURES = ROOT / "figuras"
TABLES = ROOT / "tablas"
NAMES = ["RandomForest", "XGBoost", "LSTM", "CNN-LSTM", "CNN-Transformer"]
LABELS = {"RandomForest": "Random Forest", "TrainingMean": "Media de entrenamiento"}
GROUPS = ["Window_1", "Window_2", "Window_3", "Window_4", "Soil", "DEM", "Context"]
GROUP_LABELS = ["Ventana 1\n0–<60 días", "Ventana 2\n60–<120 días", "Ventana 3\n120–<180 días", "Ventana 4\n180–240 días", "Suelo", "Elevación", "Contexto"]
COLORS = {"train": "#2d7356", "validation": "#dca63b", "test": "#356ea0"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})


def label(name):
    return LABELS.get(name, name)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figure(fig, number, title, caption, explanation, limitation):
    stem = f"figura_{number:02d}"
    png, tiff, pdf, preview = [FIGURES / f"{stem}{suffix}" for suffix in ("_600dpi.png", "_600dpi.tiff", ".pdf", "_vista.png")]
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    with Image.open(png) as image:
        image.save(tiff, compression="tiff_lzw", dpi=(600, 600))
        width, height = image.size
        assert all(abs(dpi - 600) < 1 for dpi in image.info["dpi"])
    return {"id": number, "title": title, "caption": caption, "interpretation": explanation,
        "limitations": limitation, "png": png.name, "tiff": tiff.name, "pdf": pdf.name,
        "preview": preview.name, "dpi": 600, "width_px": width, "height_px": height}


def build():
    FIGURES.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((ROOT / "evidencia_modelos.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path("data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.provenance.json").read_text(encoding="utf-8"))
    source = Path("data/yieldsat_wheat_240d/yieldsat_Germany_wheat_240d.csv")
    assert sha(source) == evidence["dataset_sha256"] == provenance["output_sha256"]
    frame = pd.read_csv(source)
    records = pd.DataFrame(evidence["records"])
    test = records[records.role == "test"]
    test_indices = test.row.to_numpy()
    y = test.yield_t_ha.to_numpy()
    statistics = evidence["statistics"]
    figures = []

    fig, ax = plt.subplots(figsize=(7.2, 5.4)); ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis("off")
    boxes = [(.3, 8.3, 3.9, 1.15, "Fuente autorizada en Colab\nGermany.nc: trigo, 188 registros"),
             (5.5, 8.3, 4.1, 1.15, "Elegibilidad y control temporal\n184 incluidos / 4 excluidos"),
             (.3, 5.7, 3.9, 1.5, "113 entradas por registro\n110 numéricas + 3 categorías\nSin clima ni índices calibrados"),
             (5.5, 5.7, 4.1, 1.5, "Satélite: 12 bandas × 4 ventanas\nMedia + contador por banda\nSuelo (6 propiedades) + DEM"),
             (.3, 3.1, 3.9, 1.45, "Separación cronológica\n2016–2019: 132 entrenamiento\n2020: 15 validación"),
             (5.5, 3.1, 4.1, 1.45, "Cinco modelos congelados\n2021–2022: 37 prueba\nSemilla 42; redes: 30 épocas"),
             (.3, .45, 3.9, 1.45, "Aplicación local en Docker\nReact → FastAPI → PostgreSQL\nArtefactos y trazabilidad SHA-256"),
             (5.5, .45, 4.1, 1.45, "Evaluación exploratoria\nMAE, RMSE, R², Wilcoxon-Holm\nBootstrap y permutación agrupada")]
    for x, yy, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=.12", facecolor="#eef5f0", edgecolor="#39745a"))
        ax.text(x+w/2, yy+h/2, text, ha="center", va="center", fontsize=8.7)
    for start, end in [((4.4,8.9),(5.25,8.9)),((7.55,8.1),(7.55,7.4)),((5.3,6.4),(4.4,6.4)),((2.25,5.45),(2.25,4.8)),((4.4,3.8),(5.25,3.8)),((7.55,2.85),(7.55,2.12)),((5.3,1.2),(4.4,1.2))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#39745a", "lw": 1.4})
    fig.tight_layout(pad=1)
    figures.append(save_figure(fig, 1, "Flujo auditable del experimento y su despliegue",
        "Flujo de procesamiento y aplicación del piloto de trigo de Alemania. El corte óptico es siembra + 240 días; se excluyen cuatro registros con siembra derivada de cosecha. Las flechas indican flujo de trabajo, no relaciones causales. La evaluación principal reserva 2021–2022.",
        "Se distinguen fuente remota, agregación, entrenamiento congelado y aplicación. La existencia de una interfaz funcional no demuestra precisión agronómica.",
        "El perfil usa agregados por registro, sin parches espaciales ni clima integrado. No representa un ensayo prospectivo de campo."))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
    yearly = records.groupby("campaign").size()
    axes[0].bar(yearly.index.astype(str), yearly.values, color=[COLORS[records.loc[records.campaign == year,"role"].iloc[0]] for year in yearly.index])
    axes[0].set(title="a) Registros por campaña", ylabel="Registros"); axes[0].tick_params(axis="x", rotation=55)
    for i, value in enumerate(yearly.values): axes[0].text(i, value+.8, str(value), ha="center", fontsize=7)
    farms = records.groupby(["region", "role"]).size().unstack(fill_value=0).reindex(columns=["train","validation","test"], fill_value=0)
    bottom = np.zeros(len(farms))
    for role, title in [("train","Entrenamiento"),("validation","Validación"),("test","Prueba")]:
        axes[1].bar(np.arange(len(farms)), farms[role], bottom=bottom, color=COLORS[role], label=title)
        bottom += farms[role].to_numpy()
    axes[1].set_xticks(np.arange(len(farms)), [str(x).split("/")[-1].replace("farm","G") for x in farms.index], rotation=45)
    axes[1].set(title="b) Distribución por granja", ylabel="Registros")
    axes[1].legend(fontsize=6, frameon=False, loc="upper right")
    roles = ["train", "validation", "test"]
    box = axes[2].boxplot([records[records.role == role].yield_t_ha for role in roles], patch_artist=True, tick_labels=["Entr.","Val.","Prueba"], widths=.55)
    for patch, role in zip(box["boxes"], roles): patch.set_facecolor(COLORS[role]); patch.set_alpha(.75)
    axes[2].set(title="c) Rendimiento observado", ylabel="t/ha")
    fig.tight_layout(w_pad=1.4)
    figures.append(save_figure(fig, 2, "Composición de la cohorte y desplazamiento del objetivo",
        "Distribución de los 184 registros por campaña (a), granja y partición (b), y rendimiento observado (c). Entrenamiento n=132, validación n=15 y prueba n=37. Las cajas muestran mediana y rango intercuartílico; bigotes de Tukey (1,5 RIC). En prueba, 36 registros son de farm2 y uno de farm6.",
        "La cobertura temporal no implica diversidad espacial. El objetivo medio baja de 9,977 t/ha en entrenamiento a 8,528 t/ha en prueba, un desplazamiento descriptivo del objetivo.",
        "Las granjas no tienen tamaños comparables y solo dos aparecen en prueba. No se acredita identidad física longitudinal de los identificadores ni causalidad del cambio temporal."))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ordered = sorted(NAMES, key=lambda name: evidence["metrics"][name]["test"]["mae"])
    for i, name in enumerate(ordered):
        mae = statistics["models"][name]["test"]["mae"]
        low, high = statistics["models"][name]["mae_95pct_cluster_bootstrap_ci"]
        axes[0].errorbar(mae, i, xerr=[[mae-low],[high-mae]], fmt="o", color="#236849", capsize=3)
    axes[0].axvline(evidence["metrics"]["TrainingMean"]["test"]["mae"], color="#9b5e28", linestyle="--", label="Media de entrenamiento")
    axes[0].set_yticks(range(5), [label(name) for name in ordered]); axes[0].invert_yaxis()
    axes[0].set(xlabel="MAE e IC 95 % exploratorio (t/ha)", title="a) Error absoluto de prueba")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(.5,.005))
    r2 = [evidence["metrics"][name]["test"]["r2"] for name in ordered]
    axes[1].barh(range(5), r2, color="#356ea0"); axes[1].set_yticks(range(5), [""]*5); axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=.8); axes[1].set(xlabel="R²", title="b) Variación no explicada")
    for i, value in enumerate(r2): axes[1].text(value-.04, i, f"{value:.3f}", ha="right", va="center", fontsize=8)
    axes[1].set_xlim(-1.5,.08)
    fig.tight_layout(w_pad=1.5, rect=(0,.1,1,1))
    figures.append(save_figure(fig, 3, "Precisión descriptiva e incertidumbre del MAE",
        "MAE de los cinco modelos en 37 registros de 2021–2022, con intervalos percentiles del 95 % mediante 2.000 remuestreos de identificadores (a), y R² sobre los mismos registros (b). La referencia predice la media del entrenamiento (9,977 t/ha; MAE de prueba 1,500 t/ha). Los intervalos no incorporan dependencia dentro de granja.",
        "CNN-LSTM tiene el menor MAE observado (0,998 t/ha), pero todos los R² son negativos. Mejorar la referencia de entrenamiento no equivale a explicar la variación del conjunto de prueba.",
        "El orden por MAE se observa después de evaluar; no es una selección validada de arquitectura. Estos intervalos son del MAE agregado, no de predicciones individuales."))

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, NAMES+["TrainingMean"]):
        prediction = np.asarray(evidence["predictions"][name])[test_indices]
        for year, color in [("2021", "#347e60"),("2022", "#c17a37")]:
            mask = test.campaign.astype(str).to_numpy() == year
            ax.scatter(y[mask], prediction[mask], s=20, alpha=.8, color=color, label=year, edgecolors="white", linewidths=.3)
        ax.plot([4.5,13],[4.5,13], "--", color="#7e8781", linewidth=.8)
        metric = evidence["metrics"][name]["test"]
        ax.set(title=label(name), xlim=(4.5,13), ylim=(4.5,13))
        ax.text(.04,.96, f"MAE={metric['mae']:.3f}\nR²={metric['r2']:.3f}", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[1,:]: ax.set_xlabel("Observado (t/ha)")
    for ax in axes[:,0]: ax.set_ylabel("Predicho (t/ha)")
    axes[0,0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    figures.append(save_figure(fig, 4, "Concordancia entre rendimientos observados y predichos",
        "Predicciones congeladas de los cinco modelos y la referencia constante para los mismos 37 registros de prueba. Cada punto es un registro, coloreado por campaña; la línea discontinua representa identidad. Los ejes comparten la misma escala para evitar comparaciones visuales engañosas.",
        "La distancia a la diagonal representa error individual. La compresión de las predicciones hacia valores centrales ayuda a explicar por qué un MAE moderado coexiste con R² negativo.",
        "Los puntos no son 37 unidades independientes verificadas. No se dibujan intervalos individuales porque no se estimaron; el color de campaña no demuestra un mecanismo climático."))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    matrix = np.ones((5,5))
    for row in statistics["wilcoxon_holm"]:
        a,b = NAMES.index(row["first"]), NAMES.index(row["second"])
        matrix[a,b] = matrix[b,a] = row["holm_p_value"]
    image = ax.imshow(matrix, cmap="YlGnBu_r", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j,i, "—" if i==j else f"{matrix[i,j]:.4f}"+(" *" if matrix[i,j]<.05 else ""), ha="center",va="center", color="white" if matrix[i,j]<.45 else "black", fontsize=9)
    ax.set_xticks(range(5),[label(n) for n in NAMES],rotation=25,ha="right")
    ax.set_yticks(range(5),[label(n) for n in NAMES])
    fig.colorbar(image,ax=ax,shrink=.85,label="p ajustada por Holm")
    ax.set_title(f"Friedman: χ²={statistics['friedman']['statistic']:.4f}; p={statistics['friedman']['p_value']:.5f}; W={statistics['friedman']['kendall_w']:.5f}", fontsize=9)
    fig.tight_layout()
    figures.append(save_figure(fig, 5, "Contrastes pareados y corrección por multiplicidad",
        "Valores p de diez contrastes Wilcoxon bilaterales sobre errores absolutos por identificador, ajustados por Holm como una sola familia. El asterisco indica p ajustada <0,05. Friedman se calculó sobre la matriz de pérdidas de los mismos 37 identificadores, con aproximación χ² y cuatro grados de libertad.",
        "Solo CNN-LSTM frente a CNN-Transformer cruza el umbral ajustado (p=0,02026). La p global de Friedman (0,03483) no identifica por sí sola qué modelos difieren; W=0,06998 indica escasa separación de rangos.",
        "La dependencia por granja limita los contrastes; con cinco modelos, la aproximación asintótica también exige cautela. Una p alta no demuestra equivalencia ni una p pequeña garantiza utilidad agronómica."))

    fig, axes = plt.subplots(1,2, figsize=(7.2,4.6), gridspec_kw={"width_ratios":[1.35,1]})
    values = np.array([[evidence["permutation"][name]["groups"][group]["mean_delta_mae"] for name in NAMES] for group in GROUPS])
    bound = max(abs(values.min()), abs(values.max()))
    image = axes[0].imshow(values,cmap="RdBu_r",norm=colors.TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect="auto")
    axes[0].set_xticks(range(5), [label(n) for n in NAMES], rotation=60, ha="right", fontsize=7)
    axes[0].set_yticks(range(7), GROUP_LABELS,fontsize=7)
    for i in range(7):
        for j in range(5): axes[0].text(j,i,f"{values[i,j]:.3f}",ha="center",va="center",fontsize=6.8,color="white" if abs(values[i,j])>bound*.6 else "black")
    axes[0].set_title("a) Sensibilidad por grupo", fontsize=9)
    fig.colorbar(image,ax=axes[0],shrink=.7,label="Δ MAE (t/ha)",fraction=.04,pad=.03)
    means = [evidence["permutation"]["CNN-LSTM"]["groups"][g]["mean_delta_mae"] for g in GROUPS]
    sd = [evidence["permutation"]["CNN-LSTM"]["groups"][g]["sd_delta_mae"] for g in GROUPS]
    axes[1].errorbar(means,range(7),xerr=sd,fmt="o",capsize=3,color="#2d7356")
    axes[1].set_yticks(range(7),["V1","V2","V3","V4","Suelo","DEM","Contexto"])
    axes[1].invert_yaxis(); axes[1].axvline(0,color="#777",linestyle="--",linewidth=.8)
    axes[1].set(xlabel="Δ MAE ± DE (t/ha)",title="b) CNN-LSTM: dispersión")
    fig.tight_layout(w_pad=1.2)
    figures.append(save_figure(fig, 6, "Interpretabilidad mediante permutación agrupada en validación",
        "Cambio medio del MAE al permutar conjuntamente cada grupo de entradas en los 15 registros de validación de 2020. Se usan 30 permutaciones con semilla 20261106, iguales para cada modelo/grupo. Panel a: medias de cinco modelos; panel b: media ± desviación estándar de CNN-LSTM. Los grupos V1–V4 incluyen bandas y contadores por ventana; Suelo incluye seis propiedades y sus indicadores; Contexto incluye cultivo, región y campaña.",
        "En CNN-LSTM, el grupo de suelo produce el mayor aumento medio del MAE (0,280 ± 0,085 t/ha). La sensibilidad a las ventanas ópticas es pequeña o negativa en esta muestra. Esto describe dependencia del modelo, no contribución causal de cada propiedad.",
        "Análisis posterior de validación pequeña; las permutaciones pueden romper dependencias entre grupos. Las barras son DE entre permutaciones, no intervalos de confianza. No es SHAP ni una ablación con reentrenamiento; valores negativos no justifican eliminar variables sin otra evaluación."))

    tables = []
    yearly_rows = []
    for year, part in frame.groupby("campaign"):
        role = "Entrenamiento" if year<=2019 else "Validación" if year==2020 else "Prueba"
        yearly_rows.append([str(year),role,str(len(part)),str(part.region.nunique()),fmt(part.yield_t_ha.mean()),fmt(part.yield_t_ha.std()),fmt(part.yield_t_ha.min()),fmt(part.yield_t_ha.max())])
    tables.append({"id":1,"title":"Cohorte elegible por campaña y partición", "headers":["Campaña","Partición","n","Granjas","Media","DE","Mín.","Máx."],"rows":yearly_rows,
        "note":"Rendimiento en t/ha; DE: desviación estándar muestral. Total 184 registros; las granjas se repiten entre campañas. No son 184 parcelas físicas longitudinales verificadas."})
    rows = [[f"Ventana {i+1}","12 medias de banda + 12 contadores", "24", f"{i*60}–{'<'+str((i+1)*60) if i<3 else '240'} días",str(int((frame[f'B02_observations_t{i}']==0).sum()))] for i in range(4)]
    rows += [["Suelo superficial","Arcilla, arena, limo, pH, SOC, N + indicadores","12","Estático, unidades fuente","0"],
             ["Elevación","DEM + indicador de disponibilidad","2","Estático, unidades fuente","0"],
             ["Contexto","Cultivo, región y campaña; one-hot","3","Categorías del registro","No aplica"],
             ["Excluidos","Clima, índices calibrados y parches espaciales","0","Fuera de este perfil","No aplica"]]
    tables.append({"id":2,"title":"Operacionalización de las 113 entradas", "headers":["Grupo","Variables","Entradas","Disponibilidad","Ventanas vacías B02"],"rows":rows,
        "note":"Los intervalos ópticos usan la variable times por píxel; día 240 incluido. Cero + contador cero representa falta de observación, no reflectancia medida. Los siete indicadores estáticos valen 1 en todos los registros; no se eliminaron ni se reentrenó. No se atribuyen unidades físicas nuevas a los predictores."})
    specs = {"RandomForest":"160 árboles; hoja mínima 2; profundidad sin límite; n_jobs=-1",
             "XGBoost":"180 árboles; profundidad 4; lr=0,05; subsample=0,9; colsample=0,9; hist",
             "LSTM":"1 capa LSTM, 32 unidades; último estado + estáticos; MLP 32→1",
             "CNN-LSTM":"Conv1D 24→32, kernel 3, padding 1; LSTM 32; MLP 32→1",
             "CNN-Transformer":"Conv1D 24→32; 1 encoder, 4 cabezas, FFN 64, dropout 0,1; media temporal + MLP"}
    rows = [[label(n),str(evidence["model_ids"][n]),specs[n],"30" if n in NAMES[2:] else "No aplica",fmt(evidence["model_details"][n]["training_seconds"])] for n in NAMES]
    tables.append({"id":3,"title":"Configuración de los modelos congelados", "headers":["Modelo","ID","Configuración","Épocas","Tiempo (s)"],"rows":rows,
        "note":"Redes: entrada 4×24 y 24 estáticos tras codificación; Adam lr=0,003, MSE, minibatch≤64, semilla 42, CPU. Normalización/imputación ajustadas solo en entrenamiento; árboles sin escalado. No hubo Optuna ni parada temprana. Tiempo de una ejecución registrado por el sistema, no benchmark controlado de hardware."})
    rows=[]
    for name in NAMES+["TrainingMean"]:
        m=evidence["metrics"][name]["test"]
        ci=statistics["models"].get(name,{}).get("mae_95pct_cluster_bootstrap_ci")
        rows.append([label(name),fmt(evidence["metrics"][name]["validation"]["mae"]),fmt(m["mae"]),f"[{ci[0]:.3f}; {ci[1]:.3f}]" if ci else "No estimado",fmt(m["rmse"]),fmt(m["r2"]),fmt(m["mape"],2)])
    tables.append({"id":4,"title":"Validación y desempeño final en los mismos 37 registros de prueba", "headers":["Modelo","MAE val.","MAE prueba","IC 95 % MAE","RMSE prueba","R² prueba","MAPE %"],"rows":rows,
        "note":"MAE/RMSE/IC en t/ha. IC exploratorios por identificador (2.000 remuestreos), sin corrección por granja. La referencia usa únicamente la media de entrenamiento. El menor MAE de validación corresponde a CNN-Transformer; el menor MAE de prueba, observado a posteriori, a CNN-LSTM."})
    rows=[[f"{label(x['first'])} / {label(x['second'])}",fmt(x["median_block_mae_difference"],4),fmt(x["p_value"],5),fmt(x["holm_p_value"],5),"Sí" if x["significant_0_05"] else "No"] for x in statistics["wilcoxon_holm"]]
    tables.append({"id":5,"title":"Comparaciones Wilcoxon bilaterales y ajuste de Holm", "headers":["Pareja A / B","Mediana Δ MAE","p original","p Holm","p Holm<0,05"],"rows":rows,
        "note":"Δ=error absoluto A−B en t/ha; negativo favorece A en la mediana. Familia de diez comparaciones. Friedman χ²(4)=10,3568; p≈0,03483; W=0,06998. Valores exploratorios, sujetos a dependencia por granja y aproximación asintótica; no prueban equivalencia ni superioridad universal."})
    rows=[[GROUP_LABELS[i].replace("\n"," ")]+[f"{evidence['permutation'][name]['groups'][group]['mean_delta_mae']:.4f} ± {evidence['permutation'][name]['groups'][group]['sd_delta_mae']:.4f}" for name in NAMES] for i,group in enumerate(GROUPS)]
    tables.append({"id":6,"title":"Sensibilidad agrupada: cambio del MAE en validación", "headers":["Grupo"]+[label(n) for n in NAMES],"rows":rows,
        "note":"Media ± DE de 30 permutaciones, n=15 (2020), t/ha; semilla 20261106. No son IC ni valores p. Un Δ positivo indica deterioro al permutar el grupo; un Δ negativo indica mejora bajo esa perturbación. Interpretabilidad posterior, no causal ni SHAP; todos los modelos permanecen congelados."})
    book=Workbook(); book.remove(book.active)
    for table in tables:
        path=TABLES/f"tabla_{table['id']:02d}.csv"
        with path.open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(table["headers"]);writer.writerows(table["rows"])
        sheet=book.create_sheet(f"Tabla {table['id']}");sheet.append([table["title"]]);sheet.append(table["headers"])
        for row in table["rows"]:sheet.append(row)
        sheet.append([table["note"]]);sheet.freeze_panes="A3"
    book.save(ROOT/"tablas_articulo.xlsx")
    manifest={"title":"Evaluación temporal del rendimiento de trigo con satélite y suelo: piloto reproducible en Alemania",
              "affiliation":"Universidad Nacional de Trujillo", "language":"es", "dataset_id":9,
              "dataset_sha256":evidence["dataset_sha256"],"model_ids":evidence["model_ids"],
              "scientific_status":"Estudio piloto retrospectivo; sin validación externa ni clima integrado",
              "figures":figures,"tables":tables,"files":{}}
    (ROOT/"catalogo_base.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Six figures at 600 dpi and six tables generated",flush=True)
    return manifest


if __name__ == "__main__":
    build()

"""Regression figures without database or web dependencies."""
from io import BytesIO
import numpy as np


def figure_bytes(evaluation, kind="residuals", dpi=600):
    # Direct Figure usage avoids pyplot's shared global state under concurrent requests.
    from matplotlib.figure import Figure
    figure = Figure(figsize=(7, 4.5), layout="constrained")
    if evaluation.get("synthetic"):
        figure.suptitle("Demostración sintética sin validez científica", fontsize=10)
    axis = figure.subplots()
    points = evaluation["points"]
    observed = np.array([p["observed"] for p in points])
    predicted = np.array([p["predicted"] for p in points])
    if kind == "residuals":
        axis.scatter(predicted, observed - predicted, s=12, alpha=.65, color="#25704f")
        axis.axhline(0, color="#a64a30", linestyle="--")
        axis.set(xlabel="Predicho (t/ha)", ylabel="Observado − predicho (t/ha)", title="Residuos en prueba")
    else:
        axis.scatter(observed, predicted, s=12, alpha=.65, color="#25704f")
        limits = [float(min(observed.min(), predicted.min())), float(max(observed.max(), predicted.max()))]
        axis.plot(limits, limits, color="#a64a30", linestyle="--")
        axis.set(xlabel="Observado (t/ha)", ylabel="Predicho (t/ha)", title="Evaluación en prueba")
    axis.grid(alpha=.15)
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=dpi)
    return stream.getvalue()



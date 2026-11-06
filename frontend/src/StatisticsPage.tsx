import { useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowDownToLine, RefreshCw } from 'lucide-react'
import { listOf, modelLabel, number, request, type Dataset, type Experiment } from './api'

const algorithms = ['RandomForest', 'XGBoost', 'LSTM', 'CNN-LSTM', 'CNN-Transformer']
const protocols: Record<string, string> = {
  temporal: 'Temporal por campañas', parcel: 'Por unidad espacial',
  region: 'Por región o granja', campaign: 'Por campaña agrupada',
}
type Metric = { mae: number; rmse: number; r2: number; mape: number | null; n: number }
type Comparison = {
  dataset_id: number; dataset_name: string; source: string; dataset_sha256: string
  validation: string; seed: number; test_observations: number; block_count: number
  bootstrap_repetitions: number; alpha: number; test_region_counts: Record<string, number>
  split_info: { train_rows: number; validation_rows: number; test_rows: number; train_years?: number[]; validation_years?: number[]; test_years?: number[] }
  friedman: { statistic: number; p_value: number; kendall_w: number; degrees_of_freedom: number }
  models: Record<string, { test: Metric; mae_95pct_cluster_bootstrap_ci: [number, number] }>
  model_versions: Record<string, { model_id: number; experiment_id: number; created_at: string }>
  training_details: Record<string, { epochs: number | null; device: string }>
  wilcoxon_holm: { first: string; second: string; statistic: number; median_block_mae_difference: number; p_value: number; holm_p_value: number; significant_0_05: boolean }[]
}
const pText = (value: number) => value < 0.001 ? value.toExponential(3) : number(value, 5)

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="panel statistics-section"><div className="panel-heading"><h2>{title}</h2></div>{children}</section>
}

export default function StatisticsPage() {
  const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: () => request<unknown>('/datasets') })
  const experimentsQuery = useQuery({ queryKey: ['experiments'], queryFn: () => request<unknown>('/experiments') })
  const datasets = listOf<Dataset>(datasetsQuery.data)
  const experiments = listOf<Experiment>(experimentsQuery.data)
  const [chosenDataset, setChosenDataset] = useState('')
  const [chosenProtocol, setChosenProtocol] = useState('')
  const datasetId = chosenDataset || String(datasets.find(d => String(d.source).startsWith('YieldSAT'))?.id || datasets[0]?.id || '')
  const groups = new Map<string, { key: string; validation: string; seed: number; types: Set<string> }>()
  for (const experiment of experiments) {
    const validation = String(experiment.validation || '')
    const seed = Number(experiment.seed)
    if (String(experiment.dataset_id) !== datasetId || experiment.status !== 'completed'
      || !protocols[validation] || !Number.isInteger(seed) || !algorithms.includes(String(experiment.model_type))) continue
    const key = `${validation}:${seed}`
    if (!groups.has(key)) groups.set(key, { key, validation, seed, types: new Set() })
    groups.get(key)!.types.add(String(experiment.model_type))
  }
  const complete = [...groups.values()].filter(group => group.types.size === algorithms.length)
    .sort((a, b) => Number(b.validation === 'temporal') - Number(a.validation === 'temporal'))
  const cohort = complete.find(group => group.key === chosenProtocol) || complete[0]
  const comparison = useQuery({
    queryKey: ['statistics', datasetId, cohort?.validation, cohort?.seed],
    queryFn: () => request<Comparison>(`/statistics/compare?dataset_id=${datasetId}&validation=${cohort!.validation}&seed=${cohort!.seed}`),
    enabled: Boolean(datasetId && cohort), retry: false, staleTime: 0,
  })
  const result = cohort ? comparison.data : undefined
  const ranked = result ? Object.entries(result.models).sort((a, b) => a[1].test.mae - b[1].test.mae) : []
  const significant = result?.wilcoxon_holm.filter(row => row.significant_0_05) || []
  function download() {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url; link.download = `estadistica_dataset_${result.dataset_id}_${result.validation}_seed${result.seed}.json`
    document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url)
  }
  return <>
    <div className="page-header"><div><span className="eyebrow">EVALUACIÓN CIENTÍFICA</span><h1>Pruebas estadísticas</h1><p>Metodología y resultados exploratorios de los cinco modelos sobre los mismos registros de prueba.</p></div>
      <button className="button button-outline" onClick={download} disabled={!result || comparison.isFetching}><ArrowDownToLine size={16} /> Descargar resultados JSON</button>
    </div>
    <Section title="Seleccionar experimento comparable">
      <div className="statistics-filters modal-form">
        <div><label htmlFor="stats-dataset">Dataset</label><select id="stats-dataset" value={datasetId} onChange={event => { setChosenDataset(event.target.value); setChosenProtocol('') }}>
          {!datasets.length && <option value="">Sin datasets</option>}
          {datasets.map(dataset => <option key={dataset.id} value={String(dataset.id)}>{dataset.name} · #{dataset.id}</option>)}
        </select></div>
        <div><label htmlFor="stats-cohort">Evaluación y semilla</label><select id="stats-cohort" value={cohort?.key || ''} disabled={!complete.length} onChange={event => setChosenProtocol(event.target.value)}>
          {!complete.length && <option value="">Sin cinco modelos comparables</option>}
          {complete.map(group => <option key={group.key} value={group.key}>{protocols[group.validation]} · semilla {group.seed}</option>)}
        </select></div>
        <button className="button button-outline" type="button" disabled={!cohort || comparison.isFetching} onClick={() => { comparison.refetch(); experimentsQuery.refetch() }}><RefreshCw size={16} /> Recalcular</button>
      </div>
      <p className="declared-note">Se usa la última versión de cada algoritmo dentro del mismo dataset, partición y semilla. Se comprueba la huella del CSV y el esquema de variables guardado en los artefactos.</p>
      {(datasetsQuery.isLoading || experimentsQuery.isLoading) && <p role="status">Cargando experimentos…</p>}
      {(datasetsQuery.error || experimentsQuery.error) && <p className="subtle-warning" role="alert">{datasetsQuery.error?.message || experimentsQuery.error?.message}</p>}
      {datasetId && !cohort && !experimentsQuery.isLoading && <p className="subtle-warning">Se requieren Random Forest, XGBoost, LSTM, CNN-LSTM y CNN-Transformer completos bajo el mismo protocolo. Entrene los modelos faltantes en Laboratorio.</p>}
      {cohort && comparison.isFetching && <p role="status">Calculando pérdidas pareadas e intervalos…</p>}
      {cohort && comparison.error && <p className="subtle-warning" role="alert">No se pudo completar la comparación: {comparison.error.message}</p>}
    </Section>
    <Section title="Métodos utilizados">
      <div className="statistics-methods">
        <article><h3>Friedman</h3><p>Compara los cinco modelos mediante sus rangos de MAE por bloque. H₀: los errores pareados tienen la misma distribución. Se calcula sobre pérdidas por identificador, nunca sobre cinco métricas globales.</p><p>Se informa χ² corregido por empates, grados de libertad, p aproximada y W de Kendall (entre 0 y 1). Requiere bloques independientes. Con cinco modelos, la aproximación χ² exige cautela.</p><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html" target="_blank" rel="noreferrer">Referencia: SciPy</a></article>
        <article><h3>Wilcoxon pareado</h3><p>Contraste bilateral para cada pareja sobre la diferencia de MAE por bloque. H₀: la distribución de las diferencias es simétrica alrededor de cero. Requiere independencia entre bloques y simetría de las diferencias.</p><p>Se descartan diferencias exactamente nulas; SciPy elige el método de cálculo según tamaño y empates. Una p alta no demuestra equivalencia.</p><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html" target="_blank" rel="noreferrer">Referencia: SciPy</a></article>
        <article><h3>Corrección de Holm</h3><p>Ajusta las diez comparaciones Wilcoxon para controlar el error familiar por comparaciones múltiples. Se evalúa p ajustada &lt; 0,05. Es una corrección, no una prueba adicional de precisión predictiva.</p><a href="https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html" target="_blank" rel="noreferrer">Referencia metodológica: R stats</a></article>
        <article><h3>Bootstrap por bloques</h3><p>Remuestrea identificadores completos con reemplazo, 2.000 veces y con la semilla seleccionada. El intervalo del 95 % utiliza los percentiles 2,5 y 97,5 del MAE ponderado por observación.</p><p>Estima incertidumbre del MAE global; no es un intervalo del rendimiento de una parcela. No corrige dependencia entre identificadores de la misma granja.</p><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html" target="_blank" rel="noreferrer">Referencia: bootstrap percentil; implementación local por bloques</a></article>
      </div>
    </Section>
    {result && <>
      <Section title="Muestra y trazabilidad">
        <p>{result.dataset_name} · {result.source} · {protocols[result.validation]} · semilla {result.seed}.</p>
        <div className="statistics-counts">
          <div><strong>{result.split_info.train_rows}</strong><span>Entrenamiento {result.split_info.train_years?.join(', ')}</span></div>
          <div><strong>{result.split_info.validation_rows}</strong><span>Validación {result.split_info.validation_years?.join(', ')}</span></div>
          <div><strong>{result.test_observations}</strong><span>Prueba {result.split_info.test_years?.join(', ')}</span></div>
          <div><strong>{result.block_count}</strong><span>Identificadores de evaluación</span></div>
        </div>
        <p className="declared-note statistics-hash">SHA-256 del CSV: {result.dataset_sha256}</p>
        <div className="table-scroll"><table><thead><tr><th>Algoritmo</th><th>Modelo</th><th>Experimento</th><th>Épocas</th></tr></thead><tbody>
          {Object.entries(result.model_versions).map(([name, version]) => <tr key={name}><td>{modelLabel(name)}</td><td>#{version.model_id}</td><td>#{version.experiment_id}</td><td>{result.training_details[name]?.epochs ?? 'No aplica'}</td></tr>)}
        </tbody></table></div>
      </Section>
      <Section title="Friedman y tamaño del efecto">
        <div className="statistics-counts"><div><strong>{number(result.friedman.statistic, 4)}</strong><span>χ² ({result.friedman.degrees_of_freedom} gl)</span></div><div><strong>{pText(result.friedman.p_value)}</strong><span>p aproximada</span></div><div><strong>{number(result.friedman.kendall_w, 5)}</strong><span>W de Kendall</span></div></div>
        <p>{result.friedman.p_value < result.alpha ? 'El contraste global cruza el umbral nominal de 0,05; la dependencia entre bloques y la aproximación estadística limitan esta evidencia.' : 'El contraste global no cruza el umbral de 0,05; esto no demuestra que los modelos sean equivalentes.'}</p>
      </Section>
      <Section title="Wilcoxon y corrección de Holm: diez comparaciones">
        <div className="table-scroll"><table><thead><tr><th>Pareja A / B</th><th>Estadístico</th><th>Δ mediana MAE A−B</th><th>p original</th><th>p Holm</th><th>Umbral 0,05</th></tr></thead><tbody>
          {result.wilcoxon_holm.map(row => <tr key={`${row.first}-${row.second}`}><td>{modelLabel(row.first)} / {modelLabel(row.second)}</td><td>{number(row.statistic, 3)}</td><td>{number(row.median_block_mae_difference, 4)}</td><td>{pText(row.p_value)}</td><td>{pText(row.holm_p_value)}</td><td>{row.significant_0_05 ? 'Cruza el umbral' : 'No lo cruza'}</td></tr>)}
        </tbody></table></div>
        <p className="declared-note">Δ en t/ha: un valor negativo favorece A y uno positivo favorece B en la mediana de las diferencias pareadas. {significant.length} de {result.wilcoxon_holm.length} comparaciones cruzan el umbral ajustado; esto no establece superioridad general.</p>
      </Section>
      <Section title="Métricas e intervalos del MAE">
        <p>MAE, RMSE, R² y MAPE son métricas descriptivas; el bootstrap estima el intervalo del MAE con {result.bootstrap_repetitions} remuestreos.</p>
        <div className="table-scroll"><table><thead><tr><th>Modelo</th><th>MAE (t/ha)</th><th>IC 95 % MAE (t/ha)</th><th>RMSE (t/ha)</th><th>R²</th><th>MAPE (%)</th></tr></thead><tbody>
          {ranked.map(([name, values]) => <tr key={name}><td>{modelLabel(name)}</td><td>{number(values.test.mae, 4)}</td><td>[{number(values.mae_95pct_cluster_bootstrap_ci[0], 4)}; {number(values.mae_95pct_cluster_bootstrap_ci[1], 4)}]</td><td>{number(values.test.rmse, 4)}</td><td>{number(values.test.r2, 4)}</td><td>{number(values.test.mape, 3)}</td></tr>)}
        </tbody></table></div>
        {ranked[0] && <p>El menor MAE observado corresponde a {modelLabel(ranked[0][0])}. Este orden describe únicamente la ejecución seleccionada.</p>}
        {ranked.some(([, values]) => values.test.r2 < 0) && <p className="subtle-warning">Hay modelos con R² negativo: su error cuadrático supera el de una constante igual a la media de prueba calculada retrospectivamente. Una diferencia estadística no implica buena precisión.</p>}
      </Section>
      <Section title="Supuestos y límites de interpretación">
        <p className="subtle-warning">La independencia física de los identificadores no está verificada. Las pérdidas de parcelas de una misma granja pueden estar correlacionadas; los valores p y los intervalos son exploratorios.</p>
        <div className="table-scroll"><table><thead><tr><th>Región o granja de prueba</th><th>Registros</th><th>Proporción</th></tr></thead><tbody>{Object.entries(result.test_region_counts).map(([name, count]) => <tr key={name}><td>{name}</td><td>{count}</td><td>{number(count / result.test_observations * 100, 2)} %</td></tr>)}</tbody></table></div>
        <p>Una sola partición y la concentración geográfica limitan las conclusiones. El bootstrap por identificador no sustituye una validación con granjas independientes. Estos resultados no validan transferencia a Perú.</p>
        <p>McNemar y DeLong no se aplicaron porque el objetivo es continuo, sin clasificación ni ROC. Diebold-Mariano no se aplicó porque esta evaluación no define un protocolo de pronósticos y dependencia serial apropiado.</p>
      </Section>
    </>}
  </>
}

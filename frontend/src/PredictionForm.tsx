import { useState, type FormEvent } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowRight } from 'lucide-react'
import { modelLabel, request, type Model, type Prediction } from './api'

type InputRecord = {
  record_id: number; external_id: string; campaign: string; crop: string; region: string
  evaluation_role: 'train' | 'validation' | 'test'
}
type InputList = {
  model_id: number; dataset_name: string; source: string; synthetic: boolean
  feature_count: number; records: InputRecord[]
}
type InputDetail = { model_id: number; record_id: number; features: Record<string, number | string> }
const roleLabels = { train: 'Entrenamiento', validation: 'Validación', test: 'Prueba' }

function parcelLabel(value: string) {
  try {
    const parts: unknown = JSON.parse(value)
    if (Array.isArray(parts)) return parts.map(String).join(' / ')
  } catch { /* Plain identifiers are already displayable. */ }
  return value
}

export default function PredictionForm({ models, canWrite, onPrediction, onSelectionChange }: {
  models: Model[]; canWrite: boolean; onPrediction: (prediction: Prediction) => void
  onSelectionChange: () => void
}) {
  const [modelId, setModelId] = useState('')
  const [campaign, setCampaign] = useState('')
  const [recordId, setRecordId] = useState('')
  const available = models.some(model => String(model.id) === modelId)
  const inputs = useQuery({
    queryKey: ['prediction-inputs', modelId],
    queryFn: () => request<InputList>(`/models/${modelId}/prediction-inputs`),
    enabled: available,
  })
  const records = available ? inputs.data?.records || [] : []
  const selected = records.find(row => String(row.record_id) === recordId && row.campaign === campaign)
  const detail = useQuery({
    queryKey: ['prediction-input-detail', modelId, recordId],
    queryFn: () => request<InputDetail>(`/models/${modelId}/prediction-inputs/${recordId}`),
    enabled: available && !!selected,
  })
  const loaded = available && !!selected && detail.isSuccess && !detail.isFetching
    && detail.data.model_id === Number(modelId) && detail.data.record_id === Number(recordId)
  const mutation = useMutation({
    mutationFn: () => request<Prediction>('/predictions', {
      method: 'POST', body: JSON.stringify({ model_id: Number(modelId), record_id: Number(recordId) }),
    }),
    onSuccess: onPrediction,
  })
  function resetResult() { mutation.reset(); onSelectionChange() }
  function submit(event: FormEvent) {
    event.preventDefault()
    if (canWrite && loaded && !mutation.isPending) mutation.mutate()
  }
  return <form className="prediction-form" onSubmit={submit}>
    <label htmlFor="prediction-model">Modelo</label>
    <select id="prediction-model" value={available ? modelId : ''} disabled={mutation.isPending}
      onChange={event => { setModelId(event.target.value); setCampaign(''); setRecordId(''); resetResult() }}>
      <option value="">Seleccionar modelo</option>
      {models.map(model => <option key={String(model.id)} value={String(model.id)}>{model.name || modelLabel(model.model_type)} #{model.id}</option>)}
    </select>
    {available && inputs.isLoading && <p role="status">Buscando parcelas y campañas…</p>}
    {available && inputs.error && <p className="subtle-warning" role="alert">{inputs.error.message}</p>}
    {available && inputs.data && <>
      <p className="declared-note">Datos disponibles: {inputs.data.dataset_name} · {inputs.data.source}{inputs.data.synthetic ? ' · Demostración sintética' : ''}.</p>
      <label htmlFor="prediction-campaign">Campaña</label>
      <select id="prediction-campaign" value={campaign} disabled={mutation.isPending}
        onChange={event => { setCampaign(event.target.value); setRecordId(''); resetResult() }}>
        <option value="">Seleccionar campaña</option>
        {[...new Set(records.map(row => row.campaign))].sort().reverse().map(year => <option key={year} value={year}>{year}</option>)}
      </select>
      <label htmlFor="prediction-record">Parcela o unidad espacial</label>
      <select id="prediction-record" value={recordId} disabled={!campaign || mutation.isPending}
        onChange={event => { setRecordId(event.target.value); resetResult() }}>
        <option value="">Seleccionar parcela</option>
        {records.filter(row => row.campaign === campaign).map(row => <option key={row.record_id} value={String(row.record_id)}>{parcelLabel(row.external_id)} · {row.region} · {roleLabels[row.evaluation_role]} · registro {row.record_id + 1}</option>)}
      </select>
      {!records.length && <p className="subtle-warning">Este dataset no contiene registros disponibles.</p>}
    </>}
    {selected && <>
      <p className="declared-note">Cultivo: {selected.crop} · Campaña: {selected.campaign} · Conjunto: {roleLabels[selected.evaluation_role]}. Esta consulta usa un registro histórico; no pronostica una campaña nueva.</p>
      {detail.isFetching && <p role="status">Cargando variables de la parcela…</p>}
      {detail.error && <p className="subtle-warning" role="alert">{detail.error.message}</p>}
      {loaded && <>
        <p role="status">Se cargaron automáticamente {Object.keys(detail.data.features).length} variables.</p>
        <details><summary>Consultar variables cargadas</summary>
          <div className="table-scroll"><table><thead><tr><th>Variable</th><th>Valor</th></tr></thead>
            <tbody>{Object.entries(detail.data.features).map(([name, value]) => <tr key={name}><td>{name.replace(/_/g, ' ')}</td><td>{String(value)}</td></tr>)}</tbody>
          </table></div>
        </details>
      </>}
    </>}
    {mutation.error && <p className="subtle-warning" role="alert">{mutation.error.message}</p>}
    <button className="button button-primary predict-button" disabled={!canWrite || !loaded || mutation.isPending}>
      {mutation.isPending ? 'Calculando…' : <>Calcular rendimiento <ArrowRight size={17} /></>}
    </button>
  </form>
}

export const API_BASE = import.meta.env.VITE_API_BASE || '/api'

export type RecordValue = Record<string, unknown>
export type User = { id: number | string; email: string; full_name?: string; role?: string; name?: string; active?: boolean }
export type Dataset = RecordValue & { id: number | string; name?: string; filename?: string; source?: string; status?: string; row_count?: number; created_at?: string; is_demo?: boolean; synthetic?: boolean }
export type Experiment = RecordValue & { id: number | string; dataset_id?: number | string; model_type?: string; status?: string; created_at?: string; metrics?: RecordValue; error?: string }
export type Model = RecordValue & { id: number | string; name?: string; model_type?: string; status?: string; feature_names?: string[]; metrics?: RecordValue; approved?: boolean; dataset_id?: number | string }
export type Prediction = RecordValue & { id: number | string; model_id?: number | string; predicted_yield?: number; prediction_t_ha?: number; created_at?: string; is_demo?: boolean; synthetic?: boolean; warnings?: string[] }
export type Parcel = RecordValue & { id: number | string; dataset_id?: number | string; name?: string; external_id?: string; crop?: string; campaign?: string; latitude?: number; longitude?: number; lat?: number; lon?: number; area_ha?: number; region?: string }

export function latestModelVersions(models: Model[]): Model[] {
  return [...models].sort((a, b) => Number(b.id) - Number(a.id))
    .filter((item, index, sorted) => sorted.findIndex(other => other.model_type === item.model_type) === index)
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const csrf = document.cookie.split('; ').find(part => part.startsWith('csrf_token='))?.split('=')[1]
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: { ...(init.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...(csrf && init.method && init.method !== 'GET' ? { 'X-CSRF-Token': decodeURIComponent(csrf) } : {}), ...init.headers },
  })
  if (!response.ok) {
    let message = `Error ${response.status}`
    try {
      const body = await response.json()
      const detail = body.detail ?? body.message ?? body.error
      if (Array.isArray(detail)) message = detail.map((item: { msg?: string }) => item.msg || String(item)).join('; ')
      else if (detail) message = String(detail)
    } catch { /* keep HTTP status */ }
    throw new Error(message)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export function listOf<T>(value: unknown): T[] {
  if (Array.isArray(value)) return value as T[]
  if (value && typeof value === 'object') {
    const wrapper = value as RecordValue
    for (const key of ['items', 'results', 'data', 'datasets', 'experiments', 'models', 'predictions', 'parcels', 'events']) {
      if (Array.isArray(wrapper[key])) return wrapper[key] as T[]
    }
  }
  return []
}

export function formatDate(value: unknown): string {
  if (!value) return '—'
  const date = new Date(String(value))
  return Number.isNaN(date.getTime()) ? '—' : new Intl.DateTimeFormat('es-PE', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

export function number(value: unknown, digits = 0): string {
  const n = Number(value)
  return value === null || value === undefined || value === '' || !Number.isFinite(n) ? '—' : new Intl.NumberFormat('es-PE', { maximumFractionDigits: digits }).format(n)
}

export function getYield(p: Prediction): number | undefined {
  const value = p.yield_t_ha ?? p.predicted_yield ?? p.prediction_t_ha
  const n = Number(value)
  return value == null || !Number.isFinite(n) ? undefined : n
}

export function isDemo(value: RecordValue): boolean {
  return Boolean(value.is_demo || value.demo || value.is_synthetic || value.synthetic || String(value.source || '').toLowerCase().includes('demo'))
}

export function modelLabel(value: unknown): string {
  const labels: Record<string, string> = { RandomForest: 'Random Forest', XGBoost: 'XGBoost', LSTM: 'LSTM', 'CNN-LSTM': 'CNN-LSTM', 'CNN-Transformer': 'CNN-Transformer', random_forest: 'Random Forest', xgboost: 'XGBoost', lstm: 'LSTM', cnn_lstm: 'CNN-LSTM', cnn_transformer: 'CNN-Transformer' }
  return labels[String(value)] || String(value || 'Modelo')
}

export function testMetric(metrics: RecordValue | undefined, key: string): unknown {
  const test = metrics?.test
  return test && typeof test === 'object' && key in test ? (test as RecordValue)[key] : metrics?.[key]
}

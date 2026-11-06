import { useQuery } from '@tanstack/react-query'
import { ArrowDownToLine, FileText, Image as ImageIcon } from 'lucide-react'
import { API_BASE, request } from './api'

type Figure = { id: number; title: string; caption: string; interpretation: string; limitations: string
  png: string; tiff: string; pdf: string; preview: string; dpi: number; width_px: number; height_px: number }
type ResearchTable = { id: number; title: string; headers: string[]; rows: string[][]; note: string }
type Catalogue = { title: string; affiliation: string; scientific_status: string; generated_at_utc: string
  dataset_id: number; dataset_sha256: string; model_ids: Record<string, number>; editorial_pending: string[]
  figures: Figure[]; tables: ResearchTable[]; files: Record<string, { sha256: string; bytes: number }> }

const url = (filename: string, download = false) => `${API_BASE}/publication/files/${encodeURIComponent(filename)}${download ? '?download=true' : ''}`

export default function PublicationPage() {
  const query = useQuery({ queryKey: ['publication'], queryFn: () => request<Catalogue>('/publication'), retry: false })
  const catalogue = query.data
  return <>
    <div className="page-header"><div><span className="eyebrow">COMUNICACIÓN CIENTÍFICA</span><h1>Artículo y figuras</h1><p>Manuscrito, seis tablas y seis figuras basados en los modelos congelados del piloto de trigo.</p></div></div>
    {query.isLoading && <p role="status">Cargando materiales científicos…</p>}
    {query.error && <p className="error-box" role="alert">{query.error.message}</p>}
    {catalogue && <>
      <section className="panel statistics-section"><h2><ImageIcon size={19} /> Figuras científicas</h2>
        <p>Vista previa ligera. Los originales PNG y TIFF tienen 600 dpi; el PDF es vectorial. Cada figura incluye una explicación y los límites de su interpretación.</p>
        <nav className="publication-index" aria-label="Índice de figuras">{catalogue.figures.map(figure => <a key={figure.id} href={`#figura-${figure.id}`}>Figura {figure.id}</a>)}</nav>
      </section>
      {catalogue.figures.map(figure => <section className="panel statistics-section publication-figure" id={`figura-${figure.id}`} key={figure.id}>
        <h2>Figura {figure.id}. {figure.title}</h2>
        <figure><a href={url(figure.png)} target="_blank" rel="noreferrer" aria-label={`Abrir figura ${figure.id} a 600 dpi`}>
          <img src={url(figure.preview)} loading="lazy" alt={`Figura ${figure.id}: ${figure.title}`} width={Math.round(figure.width_px/4)} height={Math.round(figure.height_px/4)} />
        </a><figcaption>{figure.caption}</figcaption></figure>
        <div className="publication-explanation"><h3>Cómo interpretar la figura</h3><p>{figure.interpretation}</p><h3>Qué no permite concluir</h3><p>{figure.limitations}</p></div>
        <div className="publication-downloads">
          <a className="button button-outline" href={url(figure.png,true)} download><ArrowDownToLine size={15} /> PNG · {figure.dpi} dpi</a>
          <a className="button button-outline" href={url(figure.tiff,true)} download><ArrowDownToLine size={15} /> TIFF · {figure.dpi} dpi</a>
          <a className="button button-outline" href={url(figure.pdf,true)} download><ArrowDownToLine size={15} /> PDF vectorial</a>
        </div><p className="declared-note">Original: {figure.width_px} × {figure.height_px} píxeles · 600 dpi. Las explicaciones son descriptivas y no establecen causalidad.</p>
      </section>)}
      <section className="panel statistics-section"><h2><FileText size={19} /> Tablas del artículo</h2><p>Las mismas seis tablas incluidas en el manuscrito, con notas y unidades.</p></section>
      {catalogue.tables.map(table => <section className="panel statistics-section" key={table.id}>
        <h2>Tabla {table.id}. {table.title}</h2>
        <div className="table-scroll"><table><thead><tr>{table.headers.map(header => <th key={header}>{header}</th>)}</tr></thead><tbody>
          {table.rows.map((row,index) => <tr key={index}>{row.map((cell,column) => <td key={column}>{cell}</td>)}</tr>)}
        </tbody></table></div><p className="declared-note">{table.note}</p>
        <a className="text-link" href={url(`tabla_${String(table.id).padStart(2,'0')}.csv`,true)} download>Descargar CSV</a>
      </section>)}
    </>}
  </>
}

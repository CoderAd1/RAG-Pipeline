import { useState } from 'react'
import { queryAdvanced } from '../services/api'

function AdvancedQuery() {
    const [query, setQuery] = useState('')
    const [topK, setTopK] = useState(10)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handleSubmit = async (e) => {
        e.preventDefault()

        if (!query.trim()) {
            setError('Please enter a query')
            return
        }

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const data = await queryAdvanced(query, topK)
            setResult(data)
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Failed to process query')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="card">
            <h2>🎯 Advanced Query</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
                Multimodal retrieval with tables, images, and enhanced context
            </p>

            <form onSubmit={handleSubmit}>
                <div className="input-group">
                    <label htmlFor="query">Your Question</label>
                    <textarea
                        id="query"
                        className="input textarea"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask about tables, images, or complex document structures..."
                        rows={4}
                    />
                </div>

                <div className="input-group">
                    <label htmlFor="topK">Number of Results (Top K)</label>
                    <input
                        id="topK"
                        type="number"
                        className="input"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        min={1}
                        max={30}
                    />
                </div>

                <button type="submit" className="btn btn-primary" disabled={loading}>
                    {loading ? (
                        <>
                            <span className="loading"></span>
                            <span style={{ marginLeft: '0.5rem' }}>Processing...</span>
                        </>
                    ) : (
                        'Search'
                    )}
                </button>
            </form>

            {error && (
                <div className="alert alert-error mt-3">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {result && (
                <div className="result">
                    <div className="result-header">
                        <h3>Answer</h3>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                            {result.query_time_ms}ms
                        </span>
                    </div>

                    <div className="answer-box">
                        {result.answer}
                    </div>

                    {result.visual_elements && result.visual_elements.length > 0 && (
                        <div className="sources-section">
                            <h3>Visual Elements ({result.visual_elements.length})</h3>
                            {result.visual_elements.map((visual, index) => (
                                <div key={index} className="visual-item">
                                    <div className="visual-header">
                                        <div className="source-meta">
                                            <strong>{visual.element_type.toUpperCase()}</strong>
                                            <span className="badge">Page {visual.page_number}</span>
                                            <span className="badge score-badge">
                                                {(visual.relevance_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <span className="visual-type">{visual.element_type}</span>
                                    </div>

                                    <div className="source-snippet">
                                        <strong>Description:</strong> {visual.description}
                                    </div>

                                    {visual.image_url && (
                                        <div className="visual-image-container">
                                            <img
                                                src={visual.image_url}
                                                alt={visual.description || `${visual.element_type} from page ${visual.page_number}`}
                                                className="visual-image-thumbnail"
                                            />
                                        </div>
                                    )}

                                    {visual.file_path && !visual.image_url && (
                                        <div style={{ marginTop: '0.75rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                                            📎 File: {visual.file_path}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {result.sources && result.sources.length > 0 && (
                        <div className="sources-section mt-3">
                            <h3>Text Sources ({result.sources.length})</h3>
                            {result.sources.map((source, index) => (
                                <div key={index} className="source-item">
                                    <div className="source-header">
                                        <div className="source-meta">
                                            <strong>{source.document_name}</strong>
                                            <span className="badge">Page {source.page_number}</span>
                                            <span className="badge score-badge">
                                                {(source.relevance_score * 100).toFixed(1)}%
                                            </span>
                                            {source.chunk_type && (
                                                <span className="badge">{source.chunk_type}</span>
                                            )}
                                        </div>
                                    </div>
                                    <div className="source-snippet">
                                        {source.text_snippet}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}

export default AdvancedQuery

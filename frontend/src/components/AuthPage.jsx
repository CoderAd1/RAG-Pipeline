import { useState } from 'react'
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

function AuthPage({ onAuth }) {
    const [mode, setMode] = useState('login')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [message, setMessage] = useState(null)

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError(null)
        setMessage(null)
        setLoading(true)
        try {
            const endpoint = mode === 'login' ? `${BASE_URL}/api/v1/auth/login` : `${BASE_URL}/api/v1/auth/signup`
            const { data } = await axios.post(endpoint, { email, password })
            if (data.access_token) {
                localStorage.setItem('rag_token', data.access_token)
                localStorage.setItem('rag_user', JSON.stringify(data.user))
                onAuth(data.user)
            } else {
                setMessage(data.message || 'Check your email to confirm your account.')
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Something went wrong')
        } finally {
            setLoading(false)
        }
    }

    const switchMode = (m) => { setMode(m); setError(null); setMessage(null) }

    return (
        <div className="auth-page">
            {/* Left hero panel */}
            <div className="auth-hero">
                <div className="auth-hero-content">
                    <div className="auth-brand">
                        <div className="auth-brand-icon">
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <span className="auth-brand-name">RAG Pipeline</span>
                    </div>

                    <div className="auth-hero-text">
                        <h1 className="auth-hero-title">Document Intelligence<br />at Your Fingertips</h1>
                        <p className="auth-hero-subtitle">Upload, search, and query your documents using state-of-the-art retrieval-augmented generation.</p>
                    </div>

                    <div className="auth-features">
                        {[
                            { icon: '⚡', label: 'Fast semantic search across all documents' },
                            { icon: '🧠', label: 'Multimodal AI with table & image understanding' },
                            { icon: '🔒', label: 'Secure, private document storage' },
                        ].map(f => (
                            <div key={f.label} className="auth-feature-item">
                                <span className="auth-feature-icon">{f.icon}</span>
                                <span className="auth-feature-label">{f.label}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="auth-hero-glow glow-1" />
                <div className="auth-hero-glow glow-2" />
                <div className="auth-hero-glow glow-3" />
            </div>

            {/* Right form panel */}
            <div className="auth-panel">
                <div className="auth-form-container">
                    <div className="auth-form-header">
                        <h2 className="auth-form-title">
                            {mode === 'login' ? 'Welcome back' : 'Create account'}
                        </h2>
                        <p className="auth-form-subtitle">
                            {mode === 'login'
                                ? 'Sign in to access your documents'
                                : 'Get started with your free account'}
                        </p>
                    </div>

                    <div className="auth-tabs">
                        <button className={`auth-tab ${mode === 'login' ? 'active' : ''}`} onClick={() => switchMode('login')}>
                            Sign In
                        </button>
                        <button className={`auth-tab ${mode === 'signup' ? 'active' : ''}`} onClick={() => switchMode('signup')}>
                            Sign Up
                        </button>
                        <div className={`auth-tab-indicator ${mode === 'signup' ? 'right' : ''}`} />
                    </div>

                    <form onSubmit={handleSubmit} className="auth-form">
                        {error && (
                            <div className="auth-alert auth-alert-error">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/><line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>
                                {error}
                            </div>
                        )}
                        {message && (
                            <div className="auth-alert auth-alert-success">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/><path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                                {message}
                            </div>
                        )}

                        <div className="auth-field">
                            <label className="auth-label">Email address</label>
                            <div className="auth-input-wrap">
                                <svg className="auth-input-icon" width="16" height="16" viewBox="0 0 24 24" fill="none"><rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/><path d="M2 7l10 7 10-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                                <input
                                    type="email"
                                    className="auth-input"
                                    placeholder="you@example.com"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    required
                                    autoFocus
                                />
                            </div>
                        </div>

                        <div className="auth-field">
                            <label className="auth-label">Password</label>
                            <div className="auth-input-wrap">
                                <svg className="auth-input-icon" width="16" height="16" viewBox="0 0 24 24" fill="none"><rect x="3" y="11" width="18" height="11" rx="2" stroke="currentColor" strokeWidth="2"/><path d="M7 11V7a5 5 0 0110 0v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                                <input
                                    type="password"
                                    className="auth-input"
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={e => setPassword(e.target.value)}
                                    required
                                    minLength={6}
                                />
                            </div>
                            {mode === 'signup' && <p className="auth-hint">Minimum 6 characters</p>}
                        </div>

                        <button type="submit" className="auth-submit-btn" disabled={loading}>
                            {loading ? (
                                <>
                                    <span className="auth-spinner" />
                                    {mode === 'login' ? 'Signing in…' : 'Creating account…'}
                                </>
                            ) : (
                                <>
                                    {mode === 'login' ? 'Sign In' : 'Create Account'}
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                                </>
                            )}
                        </button>
                    </form>

                    <p className="auth-switch">
                        {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
                        <button className="auth-switch-btn" onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')}>
                            {mode === 'login' ? 'Sign up' : 'Sign in'}
                        </button>
                    </p>
                </div>
            </div>
        </div>
    )
}

export default AuthPage

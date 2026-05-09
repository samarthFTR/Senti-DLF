import { useState, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sparkles, Loader2, CheckCircle2, MinusCircle, AlertCircle, 
  RefreshCw, UploadCloud, FileText, X, LayoutDashboard, 
  History, PieChart, Settings 
} from 'lucide-react';
import './index.css';

interface Probabilities {
  negative: number;
  neutral: number;
  positive: number;
  mixed: number;
}

interface SentimentResult {
  label: 'negative' | 'neutral' | 'positive' | 'mixed';
  confidence: number;
  probabilities: Probabilities;
}

interface ApiResponse {
  text: string;
  result: SentimentResult;
}

interface AspectInsight {
  aspect: string;
  sentiment: string;
  mention_count: number;
  message: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text');
  const [text, setText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [batchResults, setBatchResults] = useState<ApiResponse[] | null>(null);
  const [batchInsights, setBatchInsights] = useState<AspectInsight[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const stats = useMemo(() => {
    if (!batchResults) return null;
    const total = batchResults.length;
    const positive = batchResults.filter(r => r.result.label === 'positive').length;
    const negative = batchResults.filter(r => r.result.label === 'negative').length;
    return {
      total,
      posPercent: ((positive / total) * 100).toFixed(1),
      negPercent: ((negative / total) * 100).toFixed(1)
    };
  }, [batchResults]);

  const analyzeSentiment = async () => {
    if (activeTab === 'text' && !text.trim()) return;
    if (activeTab === 'file' && !file) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults(null);
    setBatchInsights(null);

    try {
      if (activeTab === 'text') {
        const response = await fetch('http://localhost:8000/api/v1/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text.trim() }),
        });

        if (!response.ok) {
          throw new Error('Failed to analyze sentiment. Is the backend running?');
        }

        const data: ApiResponse = await response.json();
        setResult(data.result);
      } else {
        const formData = new FormData();
        formData.append('file', file!);
        
        const response = await fetch('http://localhost:8000/api/v1/predict/file', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => null);
          throw new Error(errData?.detail || 'Failed to process file.');
        }

        const data = await response.json();
        setBatchResults(data.results);
        setBatchInsights(data.insights);
      }
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'text/plain' || droppedFile.name.endsWith('.txt')) {
        setFile(droppedFile);
      } else {
        setError('Please upload a valid .txt file');
      }
    }
  };

  const getIcon = (label: string) => {
    switch (label) {
      case 'positive': return <CheckCircle2 size={20} />;
      case 'neutral': return <MinusCircle size={20} />;
      case 'negative': return <AlertCircle size={20} />;
      case 'mixed': return <RefreshCw size={20} />;
      default: return null;
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <Sparkles size={28} color="#8E2DE2" />
          <span>Senti-DLF</span>
        </div>
        <nav className="sidebar-nav">
          <button className="nav-item active"><LayoutDashboard size={20} /> Dashboard</button>
          <button className="nav-item"><History size={20} /> History</button>
          <button className="nav-item"><PieChart size={20} /> Analytics</button>
          <button className="nav-item"><Settings size={20} /> Settings</button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <div>
            <h2 className="page-title">Sentiment Dashboard</h2>
            <p className="page-subtitle">Analyze tweets, reviews, and feedback.</p>
          </div>
          <div className="user-profile">
            <div className="avatar">AD</div>
            <span>Admin</span>
          </div>
        </header>

        <div className="dashboard-grid">
          
          {/* Main Input Widget */}
          <div className="widget input-widget">
            <div className="widget-header">
              <h3>Analyze Data</h3>
            </div>
            
            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'text' ? 'active' : ''}`}
                onClick={() => setActiveTab('text')}
              >
                <FileText size={16} /> Text Input
              </button>
              <button 
                className={`tab ${activeTab === 'file' ? 'active' : ''}`}
                onClick={() => setActiveTab('file')}
              >
                <UploadCloud size={16} /> File Upload
              </button>
            </div>

            {activeTab === 'text' ? (
              <div className="input-container">
                <textarea 
                  placeholder="Type a statement here... (e.g. 'I absolutely love the new design!')"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                      analyzeSentiment();
                    }
                  }}
                />
              </div>
            ) : (
              <div 
                className="file-upload-container"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  accept=".txt" 
                  className="hidden" 
                  ref={fileInputRef}
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      setFile(e.target.files[0]);
                    }
                  }}
                />
                {file ? (
                  <div className="file-info" onClick={(e) => e.stopPropagation()}>
                    <FileText size={32} className="file-icon" />
                    <div className="file-details">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
                    </div>
                    <button className="remove-file-btn" onClick={() => setFile(null)}>
                      <X size={20} />
                    </button>
                  </div>
                ) : (
                  <div className="upload-prompt">
                    <UploadCloud size={40} className="upload-icon" />
                    <p>Drag & drop a .txt file, or browse</p>
                  </div>
                )}
              </div>
            )}

            <button 
              className="btn-primary" 
              onClick={analyzeSentiment}
              disabled={loading || (activeTab === 'text' ? text.trim().length === 0 : !file)}
            >
              {loading ? (
                <><Loader2 className="spin" size={18} /> Analyzing...</>
              ) : (
                <><RefreshCw size={18} /> Analyze Sentiment</>
              )}
            </button>

            <AnimatePresence>
              {error && (
                <motion.div 
                  className="error-msg"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Single Result inline */}
            {result && (
              <motion.div 
                className="single-result-box"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className={`result-badge badge-${result.label}`}>
                  {getIcon(result.label)} {result.label}
                </div>
                <div className="prob-mini-bars">
                  {(['positive', 'mixed', 'neutral', 'negative'] as const).map(label => (
                    <div className="prob-mini-row" key={label}>
                      <span>{label}</span>
                      <div className="bar-bg">
                        <div className={`bar-fill fill-${label}`} style={{ width: `${result.probabilities[label] * 100}%` }} />
                      </div>
                      <span>{(result.probabilities[label] * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>

          {/* Stats Row */}
          {stats && (
            <div className="stats-row">
              <div className="stat-card">
                <span className="stat-label">Total Analyzed</span>
                <span className="stat-value">{stats.total}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Positive Sentiment</span>
                <span className="stat-value text-positive">{stats.posPercent}%</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Critical Feedback</span>
                <span className="stat-value text-negative">{stats.negPercent}%</span>
              </div>
            </div>
          )}

          {/* Insights Widget */}
          {batchInsights && batchInsights.length > 0 && (
            <div className="widget insights-widget">
              <div className="widget-header">
                <h3>Key Insights</h3>
              </div>
              <div className="insights-grid">
                {batchInsights.map((insight, idx) => (
                  <div key={idx} className={`insight-card outline-${insight.sentiment}`}>
                    <div className="insight-header">
                      <span className="insight-aspect">{insight.aspect}</span>
                      <span className="insight-count">{insight.mention_count} mentions</span>
                    </div>
                    <p className="insight-message">
                      <span className={`icon-${insight.sentiment}`}>{getIcon(insight.sentiment)}</span>
                      {insight.message}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Batch List Widget */}
          {batchResults && (
            <div className="widget batch-widget">
              <div className="widget-header">
                <h3>Recent Reviews</h3>
              </div>
              <div className="batch-table-container">
                <table className="batch-table">
                  <thead>
                    <tr>
                      <th>Review Text</th>
                      <th>Sentiment</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((res, idx) => (
                      <tr key={idx}>
                        <td className="review-cell">"{res.text}"</td>
                        <td>
                          <div className={`badge-small badge-${res.result.label}`}>
                            {getIcon(res.result.label)} {res.result.label}
                          </div>
                        </td>
                        <td className="conf-cell">{(res.result.confidence * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;

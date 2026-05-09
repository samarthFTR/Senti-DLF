import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Loader2, CheckCircle2, MinusCircle, AlertCircle, RefreshCw, UploadCloud, FileText, X } from 'lucide-react';
import './index.css';

// TypeScript Interfaces
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

function App() {
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text');
  const [text, setText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [batchResults, setBatchResults] = useState<ApiResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeSentiment = async () => {
    if (activeTab === 'text' && !text.trim()) return;
    if (activeTab === 'file' && !file) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setBatchResults(null);

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
      case 'positive': return <CheckCircle2 size={24} />;
      case 'neutral': return <MinusCircle size={24} />;
      case 'negative': return <AlertCircle size={24} />;
      case 'mixed': return <RefreshCw size={24} />;
      default: return null;
    }
  };

  return (
    <div className="app-container">
      <motion.div 
        className="glass-panel"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="header">
          <h1 className="title-glow">
            <Sparkles size={32} color="#8E2DE2" />
            Senti-DLF
          </h1>
          <p className="subtitle">BERT Transformer Fine-Tuned for Twitter Sentiment</p>
        </div>

        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'text' ? 'active' : ''}`}
            onClick={() => setActiveTab('text')}
          >
            <FileText size={18} />
            Text Input
          </button>
          <button 
            className={`tab ${activeTab === 'file' ? 'active' : ''}`}
            onClick={() => setActiveTab('file')}
          >
            <UploadCloud size={18} />
            File Upload
          </button>
        </div>

        {activeTab === 'text' ? (
          <div className="input-container">
            <textarea 
              placeholder="Type a tweet or statement here to analyze its sentiment... (e.g. 'I absolutely love the new design!')"
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
                <button 
                  className="remove-file-btn" 
                  onClick={() => setFile(null)}
                >
                  <X size={20} />
                </button>
              </div>
            ) : (
              <div className="upload-prompt">
                <UploadCloud size={48} className="upload-icon" />
                <p>Drag & drop a .txt file here, or click to browse</p>
                <span className="upload-hint">File should contain reviews separated by blank lines</span>
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
            <>
              <Loader2 className="spin" size={20} />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <RefreshCw size={20} />
              <span>Analyze Sentiment {activeTab === 'text' && '(Ctrl + Enter)'}</span>
            </>
          )}
        </button>

        <AnimatePresence mode="wait">
          {error && (
            <motion.div 
              key="error"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto', marginTop: '1rem' }}
              exit={{ opacity: 0, height: 0 }}
              style={{ color: '#FF0055', textAlign: 'center', fontSize: '0.9rem' }}
            >
              {error}
            </motion.div>
          )}

          {result && (
            <motion.div 
              key="results"
              className="results-container"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
            >
              <div className="primary-result">
                <motion.div 
                  className={`result-badge badge-${result.label}`}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", bounce: 0.5 }}
                >
                  {getIcon(result.label)}
                  {result.label}
                </motion.div>
              </div>

              <div className="prob-container">
                {(['positive', 'mixed', 'neutral', 'negative'] as const).map((label) => (
                  <div className="prob-row" key={label}>
                    <div className="prob-label">{label}</div>
                    <div className="prob-bar-bg">
                      <motion.div 
                        className={`prob-bar-fill fill-${label}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${result.probabilities[label] * 100}%` }}
                        transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                      />
                    </div>
                    <div className="prob-value">
                      {(result.probabilities[label] * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {batchResults && (
            <motion.div 
              key="batch-results"
              className="batch-results-container"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
            >
              <h3 className="batch-title">Analyzed {batchResults.length} Reviews</h3>
              <div className="batch-list">
                {batchResults.map((res, idx) => (
                  <div key={idx} className="batch-item">
                    <div className="batch-item-text">"{res.text}"</div>
                    <div className={`batch-item-badge badge-${res.result.label}`}>
                      {getIcon(res.result.label)}
                      {res.result.label} ({(res.result.confidence * 100).toFixed(1)}%)
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

export default App;

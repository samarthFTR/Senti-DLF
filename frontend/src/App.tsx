import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Loader2, CheckCircle2, MinusCircle, AlertCircle, RefreshCw } from 'lucide-react';
import './index.css';

// TypeScript Interfaces
interface Probabilities {
  negative: number;
  neutral: number;
  positive: number;
}

interface SentimentResult {
  label: 'negative' | 'neutral' | 'positive';
  confidence: number;
  probabilities: Probabilities;
}

interface ApiResponse {
  text: string;
  result: SentimentResult;
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeSentiment = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
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
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const getIcon = (label: string) => {
    switch (label) {
      case 'positive': return <CheckCircle2 size={24} />;
      case 'neutral': return <MinusCircle size={24} />;
      case 'negative': return <AlertCircle size={24} />;
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

        <button 
          className="btn-primary" 
          onClick={analyzeSentiment}
          disabled={loading || text.trim().length === 0}
        >
          {loading ? (
            <>
              <Loader2 className="spin" size={20} />
              <span>Analyzing Vectors...</span>
            </>
          ) : (
            <>
              <RefreshCw size={20} />
              <span>Analyze Sentiment (Ctrl + Enter)</span>
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
                {(['positive', 'neutral', 'negative'] as const).map((label) => (
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
        </AnimatePresence>
      </motion.div>
    </div>
  );
}

export default App;

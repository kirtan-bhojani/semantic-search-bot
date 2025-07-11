import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchPerformed, setSearchPerformed] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults(null);
    setSearchPerformed(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error('Search failed:', err);
      setResults([]);
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <h1>Your smart shopping assistant</h1>
          <p className="tagline">Ask it your way</p>
        </div>
      </header>

      <main className="main-content">
        <div className="results-container">
          {loading ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Searching for the best results...</p>
            </div>
          ) : searchPerformed && results && results.length > 0 ? (
            <div className="results-list">
              <h2 className="results-title">Search Results</h2>
              {results.map((item, index) => (
                <div key={index} className="result-card">
                  <div className="image-placeholder">
                    <span className="camera-icon">📷</span>
                    <span>product image</span>
                  </div>
                  <div className="result-content">
                    <h3 className="result-title">{item.title}</h3>
                    <div 
                      className="result-description" 
                      dangerouslySetInnerHTML={{ __html: item.description }} 
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : searchPerformed ? (
            <div className="no-results">
              <div className="no-results-icon">🔍</div>
              <h3>No results found</h3>
              <p>Try different keywords or a more general search term</p>
            </div>
          ) : (
            <div className="initial-state">
              <div className="search-icon">🔍</div>
              <h3>What are you looking for today?</h3>
              <p>Your results will appear here</p>
            </div>
          )}
        </div>
      </main>

      <div className="search-section">
        <form onSubmit={handleSubmit} className="search-form">
          <div className="search-container">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search the way you talk..."
              className="search-input"
              required
              disabled={loading}
            />
            <button 
              type="submit" 
              className={`search-button ${loading ? 'loading' : ''}`}
              disabled={loading}
            >
              {loading ? (
                <span className="button-loader"></span>
              ) : (
                <span>Search</span>
              )}
            </button>
          </div>
        </form>
      </div>

      {/*<footer className="app-footer">
        <p> {new Date().getFullYear()} Semantic Search Bot. All rights reserved.</p>
      </footer>*/}
    </div>
  );
}

export default App;

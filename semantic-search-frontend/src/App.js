import React, { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResults(null);

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
    <div style={{ padding: 20, maxWidth: 600, margin: 'auto' }}>
      <h1>Semantic Search Bot</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search query"
          style={{ width: '100%', padding: 8, fontSize: 16 }}
          required
        />
        <button type="submit" style={{ marginTop: 10, padding: '8px 16px', fontSize: 16 }}>
          Search
        </button>
      </form>

      <div style={{ marginTop: 30 }}>
        {loading && <p>Loading results...</p>}

        {results && results.length > 0 && (
          <div>
            <h3>Top Results:</h3>
            {results.map((item, index) => (
              <div
                key={index}
                style={{
                  padding: 10,
                  marginBottom: 10,
                  border: '1px solid #ccc',
                  borderRadius: 5,
                }}
              >
                <h4>{item.title}</h4>
                <p>{item.description}</p>
              </div>
            ))}
          </div>
        )}

        {results && results.length === 0 && !loading && <p>No results found.</p>}
      </div>
    </div>
  );
}

export default App;

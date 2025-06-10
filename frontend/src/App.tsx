import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Baseball AI - Over/Under Predictions</h1>
      <p>Repository setup complete. Start building!</p>
      <div>
        <h2>Next Steps:</h2>
        <ol>
          <li>Add your API keys to .env file</li>
          <li>Create and activate virtual environment: <code>python3 -m venv venv && source venv/bin/activate</code></li>
          <li>Install Python dependencies: <code>pip install -r backend/requirements/dev.txt</code></li>
          <li>Install frontend dependencies: <code>cd frontend && npm install</code></li>
          <li>Start development: <code>make run-backend</code> and <code>make run-frontend</code></li>
        </ol>
      </div>
    </div>
  );
}

export default App;

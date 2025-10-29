import React, { useState } from "react";

export default function EWSFGui() {
  const [uploading, setUploading] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [inspected, setInspected] = useState(null);
  const [target, setTarget] = useState("");
  const [features, setFeatures] = useState([]);
  const [modelPath, setModelPath] = useState("");
  const [status, setStatus] = useState("");
  const [predUrl, setPredUrl] = useState("");

  async function uploadCsv(ev) {
    const file = ev.target.files[0];
    if (!file) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch('/upload', { method: 'POST', body: form });
      const j = await res.json();
      setCsvFile(j.filename);
      setStatus("Uploaded: " + j.filename);
      // automatically inspect
      await inspectCsv(j.filename);
    } catch (err) {
      console.error(err);
      setStatus("Upload failed");
    }
    setUploading(false);
  }

  async function inspectCsv(filename) {
    setStatus("Inspecting CSV...");
    const res = await fetch('/inspect', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ filename }) });
    const j = await res.json();
    setInspected(j);
    setStatus('Columns loaded');
    // default: set target to last column
    if (j.columns && j.columns.length > 0) setTarget(j.columns[j.columns.length-1]);
  }

  function toggleFeature(col) {
    if (features.includes(col)) setFeatures(features.filter(c => c !== col));
    else setFeatures([ ...features, col ]);
  }

  async function train() {
    if (!csvFile) { setStatus('Upload CSV first'); return; }
    if (!target) { setStatus('Select a target'); return; }
    if (features.length === 0) { setStatus('Select at least one feature'); return; }
    setStatus('Training...');
    const payload = {
      filename: csvFile,
      target,
      features,
      hyperparams: { n_trees: 90, max_depth: 8, min_leaf: 3, n_subforests: 3 }
    };
    const res = await fetch('/train', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const j = await res.json();
    if (j.status === 'ok') {
      setModelPath(j.model);
      setStatus('Trained. Model saved: ' + j.model);
    } else {
      setStatus('Training failed: ' + (j.error || 'unknown'));
    }
  }

  async function uploadForPredict(ev) {
    const file = ev.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    setStatus('Uploading prediction CSV...');
    const res = await fetch('/upload_predict', { method: 'POST', body: form });
    const j = await res.json();
    if (j.filename) {
      setStatus('Uploaded for prediction: ' + j.filename);
      // run predict
      await predict(j.filename);
    }
  }

  async function predict(filename) {
    if (!modelPath) { setStatus('No model selected'); return; }
    setStatus('Predicting...');
    const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ filename, model: modelPath }) });
    const j = await res.json();
    if (j.status === 'ok') {
      setPredUrl(j.predictions_url);
      setStatus('Predictions ready');
    } else {
      setStatus('Predict failed: ' + (j.error || 'unknown'));
    }
  }

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">EWSF — Upload CSV & Train</h1>
      <div className="mb-4">
        <label className="block mb-1">Upload CSV (training)</label>
        <input type="file" accept=".csv" onChange={uploadCsv} />
        <div className="text-sm text-gray-600 mt-2">{status}</div>
      </div>

      {inspected && (
        <div className="mb-6">
          <h2 className="font-semibold">Columns</h2>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {inspected.columns.map(col => (
              <div key={col} className="border rounded p-2">
                <div className="flex items-center justify-between">
                  <div>{col}</div>
                  <div>
                    <input type="checkbox" checked={features.includes(col)} onChange={()=>toggleFeature(col)} /> Feature
                  </div>
                </div>
                <div className="mt-1 text-xs text-gray-500">Preview: {inspected.preview && inspected.preview.map(r=>r[col]).slice(0,3).join(', ')}</div>
              </div>
            ))}
          </div>

          <div className="mt-3">
            <label>Target column</label>
            <select value={target} onChange={(e)=>setTarget(e.target.value)} className="ml-2">
              {inspected.columns.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>

          <div className="mt-4">
            <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={train}>Train model</button>
            <div className="mt-2 text-sm">Trained model path: {modelPath}</div>
          </div>
        </div>
      )}

      <hr />

      <div className="mt-4">
        <h2 className="font-semibold">Predict on new CSV</h2>
        <input type="file" accept=".csv" onChange={uploadForPredict} />
        {predUrl && <div className="mt-2">Download predictions: <a href={predUrl} target="_blank" rel="noreferrer">{predUrl}</a></div>}
      </div>

    </div>
  );
}

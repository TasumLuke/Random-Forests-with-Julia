// backend/server.js
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const name = Date.now() + '_' + file.originalname;
    cb(null, name);
  }
});
const upload = multer({ storage });

// POST /upload  -> upload CSV for training/inspection
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  return res.json({ filename: req.file.filename });
});

// POST /upload_predict -> upload CSV for prediction
app.post('/upload_predict', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
  return res.json({ filename: req.file.filename });
});

// POST /inspect  -> ask Julia to inspect the CSV and return columns + preview
app.post('/inspect', (req, res) => {
  const { filename } = req.body;
  if (!filename) return res.status(400).json({ error: 'filename required' });
  const fpath = path.join(UPLOAD_DIR, filename);
  const julia = spawn('julia', ['--project=..', path.join('scripts','inspect_csv.jl'), fpath], { cwd: path.join(__dirname,'..') });
  let out = '';
  let err = '';
  julia.stdout.on('data', d=> out += d.toString());
  julia.stderr.on('data', d=> err += d.toString());
  julia.on('close', code => {
    if (code !== 0) return res.status(500).json({ error: 'Julia inspect failed', details: err });
    try {
      const j = JSON.parse(out);
      return res.json(j);
    } catch(e) {
      return res.status(500).json({ error: 'Invalid JSON from Julia', details: out });
    }
  });
});

// POST /train -> instruct Julia to train on an uploaded CSV
app.post('/train', (req, res) => {
  const payload = req.body;
  /* payload = { filename, target, features: [...], hyperparams: {...} } */
  if (!payload.filename || !payload.target || !payload.features) {
    return res.status(400).json({ error: 'filename, target, features required' });
  }
  const fpath = path.join(UPLOAD_DIR, payload.filename);
  const outmodel = `model_${Date.now()}.bin`;
  const outpath = path.join(UPLOAD_DIR, outmodel);

  // build args for Julia script
  const args = [ '--project=..', path.join('scripts','train_api.jl'), fpath, payload.target, payload.features.join(','), outpath,
                 String(payload.hyperparams?.n_trees || 90),
                 String(payload.hyperparams?.max_depth || 8),
                 String(payload.hyperparams?.min_leaf || 3),
                 String(payload.hyperparams?.n_subforests || 3)
               ];
  const julia = spawn('julia', args, { cwd: path.join(__dirname,'..') });
  let out = '', err = '';
  julia.stdout.on('data', d=> out += d.toString());
  julia.stderr.on('data', d=> err += d.toString());
  julia.on('close', code => {
    if (code !== 0) return res.status(500).json({ error: 'Julia train failed', details: err });
    try {
      const j = JSON.parse(out);
      if (j.status === 'ok') {
        return res.json({ status: 'ok', model: outmodel });
      } else {
        return res.status(500).json({ error: 'Julia train returned error', details: j });
      }
    } catch(e) {
      return res.status(500).json({ error: 'Invalid JSON from Julia', details: out });
    }
  });
});

// POST /predict -> run Julia to predict using model and uploaded CSV
app.post('/predict', (req, res) => {
  const { filename, model } = req.body;
  if (!filename || !model) return res.status(400).json({ error: 'filename and model required' });
  const fpath = path.join(UPLOAD_DIR, filename);
  const modelpath = path.join(UPLOAD_DIR, model);
  const outpred = `predictions_${Date.now()}.csv`;
  const outpath = path.join(UPLOAD_DIR, outpred);

  const args = ['--project=..', path.join('scripts','predict_api.jl'), fpath, modelpath, outpath];
  const julia = spawn('julia', args, { cwd: path.join(__dirname,'..') });
  let out = '', err = '';
  julia.stdout.on('data', d=> out += d.toString());
  julia.stderr.on('data', d=> err += d.toString());
  julia.on('close', code => {
    if (code !== 0) return res.status(500).json({ error: 'Julia predict failed', details: err });
    // return URL to download predictions (static file under uploads)
    return res.json({ status: 'ok', predictions_url: `/uploads/${outpred}` });
  });
});

// Serve uploaded files statically
app.use('/uploads', express.static(path.join(__dirname,'uploads')));

// Serve simple index (if you want) or instruct to serve frontend separately
app.get('/', (req,res) => res.send('EWSF backend running. Use the React frontend to interact.'));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));

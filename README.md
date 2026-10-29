# ASD Risk Detector — Full-stack website

This starter package adds a custom HTML/CSS/JavaScript interface and a Flask API while preserving your existing behavioral model files.

## Important safety and model notes

- The questionnaire prompts are simplified demonstration prompts, not a validated clinical questionnaire.
- Yes/No answers are mapped directly to `A1_Score` ... `A10_Score`. Validate this mapping before interpreting any output.
- The audio tool records or accepts WAV audio and displays waveform/Mel-spectrogram visualizations. It does **not** classify audio as ASD/typical.
- Do not use this prototype for diagnosis, treatment, or clinical decisions.
- The API does not save questionnaire submissions or uploaded audio to disk.

## 1. Keep your existing model files

In the project root, create a folder named `models` if it does not exist, and place these existing files inside it:

```text
models/
  behavioral_ann.keras
  behavioral_scaler.joblib
  behavioral_metadata.joblib
```

Do not delete your existing `models`, `results`, or audio dataset folders. This web starter uses the behavioral model only.

## 2. Copy these files into your project

```text
ASD_Risk_Detector/
  server.py
  requirements.txt
  render.yaml
  .gitignore
  website/
    index.html
    styles.css
    app.js
  models/
    behavioral_ann.keras
    behavioral_scaler.joblib
    behavioral_metadata.joblib
```

Copy the provided `server.py`, `requirements.txt`, `render.yaml`, and `.gitignore` into the project root. Create `website` and put the three website files in it. Keep the model files in `models`.

## 3. Use your existing Python environment locally

**Do not install the deployment requirements file into `C:\asdvenv` automatically.** Your existing environment already contains the behavioral model stack, and the deployment pins use Python 3.11-compatible packages.

First check your existing environment:

```powershell
C:\asdvenv\Scripts\python.exe -c "import flask, pandas, joblib, librosa, tensorflow, sklearn; print('All core packages import successfully')"
```

If that command reports a missing package, install only the missing package into `C:\asdvenv` rather than downgrading TensorFlow or replacing the environment.

The `requirements.txt` file is intended for the cloud deployment environment.

## 4. Run locally

```powershell
cd "C:\Users\Jayalaxmi\Downloads\ASD_Risk_Detector_Project\ASD_Risk_Detector"
C:\asdvenv\Scripts\python.exe server.py
```

Open `http://127.0.0.1:5000` in Chrome. Use `http://127.0.0.1:5000/api/status` to check whether the model loaded.

If you see an error, copy the full PowerShell traceback and send it back.

## 5. Prepare for free deployment

Recommended first deployment target: Render Free Web Service. Free hosting is for testing/hobby projects, not clinical or production use. Free services can sleep when idle and the first request after sleep may take around a minute. The free filesystem is ephemeral, so do not save user uploads or submissions there.

Before uploading to GitHub:
1. Make sure the three model files are inside `models`.
2. Never commit passwords, API keys, private user data, or real child recordings.
3. Review the repository contents before making it public. A private repository is preferable if your hosting account supports it.
4. Check whether your model license and dataset terms permit redistribution/hosting.

On Render, create a **Web Service** connected to your Git repository and use:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn server:app --workers 1 --threads 2 --timeout 120`
- Plan: Free

The provided `render.yaml` contains the same settings. Deploy and inspect the logs. TensorFlow can use substantial memory; if the free instance cannot load the model, the service may need a different host or a lighter deployment strategy.

## 6. Browser microphone

Microphone access normally requires HTTPS on the deployed site. It works on `localhost` during local development and on a deployed HTTPS URL after the site is live. The user must grant browser microphone permission.

## Endpoints

- `GET /health` — hosting health check
- `GET /api/status` — model load status
- `POST /api/predict` — behavioral model output
- `POST /api/audio-analysis` — WAV visualization data only

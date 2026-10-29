
import io
import os
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, jsonify, request, send_from_directory


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "website"
MODEL_DIR = BASE_DIR / "models"

# Behavioral questionnaire model
MODEL_PATH = Path(
    os.environ.get(
        "BEHAVIORAL_MODEL_PATH",
        str(MODEL_DIR / "behavioral_ann.keras")
    )
)

SCALER_PATH = Path(
    os.environ.get(
        "BEHAVIORAL_SCALER_PATH",
        str(MODEL_DIR / "behavioral_scaler.joblib")
    )
)

METADATA_PATH = Path(
    os.environ.get(
        "BEHAVIORAL_METADATA_PATH",
        str(MODEL_DIR / "behavioral_metadata.joblib")
    )
)

# Child-cry audio model
AUDIO_MODEL_PATH = Path(
    os.environ.get(
        "AUDIO_MODEL_PATH",
        str(MODEL_DIR / "audio_cnn.keras")
    )
)

AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_SECONDS = 5
AUDIO_N_MELS = 128
AUDIO_N_FFT = 1024
AUDIO_HOP_LENGTH = 256
AUDIO_THRESHOLD = 0.78

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# Cache loaded models so they are not reloaded for every request.
_behavioral_bundle = None
_behavioral_load_error = None

_audio_model = None
_audio_load_error = None


# ============================================================
# 2. BEHAVIORAL MODEL LOADING
# ============================================================

def get_model_bundle():
    global _behavioral_bundle, _behavioral_load_error

    if _behavioral_bundle is not None:
        return _behavioral_bundle

    if _behavioral_load_error:
        raise RuntimeError(_behavioral_load_error)

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        METADATA_PATH
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        _behavioral_load_error = (
            "Behavioral model files are missing: "
            + ", ".join(missing)
        )
        raise RuntimeError(_behavioral_load_error)

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        scaler = joblib.load(SCALER_PATH)
        metadata = joblib.load(METADATA_PATH)

        _behavioral_bundle = (
            model,
            scaler,
            metadata
        )

        return _behavioral_bundle

    except Exception as exc:
        _behavioral_load_error = (
            f"Could not load behavioral model: {exc}"
        )
        raise RuntimeError(_behavioral_load_error) from exc


# ============================================================
# 3. AUDIO MODEL LOADING
# ============================================================

def get_audio_model():
    global _audio_model, _audio_load_error

    if _audio_model is not None:
        return _audio_model

    if _audio_load_error:
        raise RuntimeError(_audio_load_error)

    if not AUDIO_MODEL_PATH.is_file():
        _audio_load_error = (
            f"Audio model not found: {AUDIO_MODEL_PATH}"
        )
        raise RuntimeError(_audio_load_error)

    try:
        model = tf.keras.models.load_model(
            AUDIO_MODEL_PATH,
            compile=False
        )

        expected_shape = tuple(model.input_shape[1:])

        if expected_shape != (128, 313, 1):
            raise ValueError(
                "Unexpected audio model input shape: "
                f"{model.input_shape}. Expected "
                "(None, 128, 313, 1)."
            )

        _audio_model = model
        return _audio_model

    except Exception as exc:
        _audio_load_error = (
            f"Could not load audio model: {exc}"
        )
        raise RuntimeError(_audio_load_error) from exc


# ============================================================
# 4. AUDIO PREPROCESSING
# ============================================================

def audio_to_model_input(raw_bytes):
    """
    Convert an uploaded audio recording into the same
    Mel-spectrogram representation used by the existing
    audio inference script.

    Audio is decoded and processed in memory.
    """

    if not raw_bytes:
        raise ValueError("The uploaded audio file is empty.")

    audio, sr = librosa.load(
        io.BytesIO(raw_bytes),
        sr=AUDIO_SAMPLE_RATE,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("No audio samples were found.")

    audio = audio.astype(np.float32)

    max_length = (
        AUDIO_SAMPLE_RATE * AUDIO_MAX_SECONDS
    )

    # Trim or zero-pad to exactly five seconds.
    if len(audio) > max_length:
        audio = audio[:max_length]

    elif len(audio) < max_length:
        audio = np.pad(
            audio,
            (0, max_length - len(audio))
        )

    # Match the existing training preprocessing.
    audio = audio / (
        np.max(np.abs(audio)) + 1e-8
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=AUDIO_SAMPLE_RATE,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
        n_mels=AUDIO_N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = (
        mel_db - mel_db.mean()
    ) / (mel_db.std() + 1e-8)

    # Shape: (1, 128, 313, 1)
    X = mel_db.astype(
        np.float32
    )[np.newaxis, ..., np.newaxis]

    return X, sr


# ============================================================
# 5. WEBSITE HOME PAGE
# ============================================================

@app.get("/")
def home():
    return send_from_directory(
        WEB_DIR,
        "index.html"
    )


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "ASD Risk Detector"
    })


# ============================================================
# 7. API STATUS
# ============================================================

@app.get("/api/status")
def api_status():
    status = {
        "status": "online",
        "behavioral_model_loaded": False,
        "audio_model_loaded": False
    }

    try:
        model, _, metadata = get_model_bundle()

        status["behavioral_model_loaded"] = True
        status["behavioral_model_name"] = (
            "behavioral_ann.keras"
        )
        status["behavioral_input_count"] = len(
            metadata.get("input_columns", [])
        )

    except Exception as exc:
        status["behavioral_model_error"] = str(exc)

    try:
        audio_model = get_audio_model()

        status["audio_model_loaded"] = True
        status["audio_model_name"] = "audio_cnn.keras"
        status["audio_input_shape"] = list(
            audio_model.input_shape
        )

    except Exception as exc:
        status["audio_model_error"] = str(exc)

    # The server can still serve the website if a model is missing.
    status["status"] = "online"

    return jsonify(status)


# ============================================================
# 8. BEHAVIORAL QUESTIONNAIRE PREDICTION
# ============================================================

@app.post("/api/predict")
def predict():
    try:
        model, scaler, metadata = get_model_bundle()

        body = request.get_json(silent=True) or {}

        columns = metadata.get("input_columns")

        if not isinstance(columns, list) or not columns:
            return jsonify({
                "error": (
                    "Model metadata has no input_columns list."
                )
            }), 500

        required = [
            "A1_Score",
            "A2_Score",
            "A3_Score",
            "A4_Score",
            "A5_Score",
            "A6_Score",
            "A7_Score",
            "A8_Score",
            "A9_Score",
            "A10_Score",
            "age",
            "gender",
            "ethnicity",
            "jaundice",
            "autism",
            "country_of_res",
            "used_app_before",
            "result",
            "age_desc",
            "relation"
        ]

        missing = [
            name
            for name in required
            if name not in body
        ]

        if missing:
            return jsonify({
                "error": (
                    "Missing fields: "
                    + ", ".join(missing)
                )
            }), 400

        missing_columns = [
            name
            for name in columns
            if name not in body
        ]

        if missing_columns:
            return jsonify({
                "error": (
                    "Missing model input fields: "
                    + ", ".join(missing_columns)
                )
            }), 400

        row = pd.DataFrame(
            [[body[name] for name in columns]],
            columns=columns
        )

        transformed = scaler.transform(row)

        raw = model.predict(
            transformed,
            verbose=0
        )

        score = float(
            np.asarray(raw).reshape(-1)[0]
        )

        threshold = float(
            metadata.get("threshold", 0.5)
        )

        return jsonify({
            "success": True,
            "model_output": score,
            "display_percent": round(score * 100, 2),
            "threshold": threshold,
            "flag": (
                "above_configured_threshold"
                if score >= threshold
                else "below_configured_threshold"
            ),
            "notice": (
                "Experimental model output only. This is not "
                "a diagnosis or a validated probability of autism. "
                "The questionnaire mapping and model require "
                "independent validation."
            )
        })

    except RuntimeError as exc:
        return jsonify({
            "error": str(exc)
        }), 503

    except Exception:
        app.logger.exception(
            "Behavioral prediction failed"
        )

        return jsonify({
            "error": (
                "Behavioral prediction failed. "
                "Check the submitted answers and model configuration."
            )
        }), 500


# ============================================================
# 9. AUDIO ANALYSIS AND EXPERIMENTAL MODEL INFERENCE
# ============================================================

@app.post("/api/audio-analysis")
def audio_analysis():
    """
    Accept an audio file from the website and run the existing
    experimental audio model. The uploaded recording is not saved
    by this endpoint.
    """

    uploaded = request.files.get("audio")

    if uploaded is None:
        return jsonify({
            "error": (
                "No audio file received. Please upload or record audio."
            )
        }), 400

    try:
        raw_bytes = uploaded.read()

        if not raw_bytes:
            return jsonify({
                "error": "The audio file is empty."
            }), 400

        model = get_audio_model()

        X, original_sr = audio_to_model_input(
            raw_bytes
        )

        prediction_array = model.predict(
            X,
            verbose=0
        )

        prediction = float(
            np.asarray(prediction_array).reshape(-1)[0]
        )

        if not np.isfinite(prediction):
            raise ValueError(
                "The model returned a non-finite output."
            )

        # Do not label this value as a calibrated ASD probability.
        experimental_flag = (
            "above_experimental_threshold"
            if prediction >= AUDIO_THRESHOLD
            else "below_experimental_threshold"
        )

        return jsonify({
            "success": True,
            "model_name": "audio_cnn.keras",
            "raw_model_output": prediction,
            "display_percent": round(
                prediction * 100, 2
            ),
            "experimental_threshold": AUDIO_THRESHOLD,
            "experimental_flag": experimental_flag,
            "input_shape": list(X.shape),
            "model_sample_rate": AUDIO_SAMPLE_RATE,
            "original_sample_rate_after_decode": original_sr,
            "processed_duration_seconds": AUDIO_MAX_SECONDS,
            "audio_saved": False,
            "notice": (
                "Experimental research output only. This score "
                "is not a calibrated ASD probability, clinical "
                "diagnosis, or substitute for assessment by a "
                "qualified healthcare professional. The model "
                "and threshold require independent validation."
            )
        })

    except RuntimeError as exc:
        return jsonify({
            "error": str(exc)
        }), 503

    except Exception:
        app.logger.exception(
            "Audio prediction failed"
        )

        return jsonify({
            "error": (
                "Could not process this audio file. Try a valid "
                "WAV recording and check the server console."
            )
        }), 400


# ============================================================
# 10. FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": (
            "The audio file is too large. "
            "Please use a file smaller than 15 MB."
        )
    }), 413



# ============================================================
# 11. START THE SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

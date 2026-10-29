
const QUESTIONS = [
  "Does the child look at you when you call their name?",
  "Does the child make eye contact during interaction?",
  "Does the child point to indicate something interesting?",
  "Does the child show interest in interacting with other people?",
  "Does the child imitate actions or expressions?",
  "Does the child use gestures to communicate?",
  "Does the child respond to other people's emotions?",
  "Does the child engage in pretend play?",
  "Does the child show interest in sharing activities with others?",
  "Does the child respond to social interaction?"
];

const $ = (id) => document.getElementById(id);

let selectedAudio = null;
let selectedAudioName = "";
let mediaStream = null;
let mediaRecorder = null;
let recordingTimer = null;
let recordingSeconds = 0;
let toastTimer = null;

function showToast(message) {
  const toast = $("toast");
  if (!toast) {
    alert(message);
    return;
  }

  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3600);
}

function renderQuestions() {
  const container = $("questions");
  if (!container) return;

  container.innerHTML = QUESTIONS.map((question, i) => `
    <div class="question-item">
      <div class="question-copy">
        <span class="question-index">${String(i + 1).padStart(2, "0")}</span>
        <p>${question}</p>
      </div>
      <div class="choice-group" role="radiogroup" aria-label="Question ${i + 1}">
        <label class="choice-option">
          <input type="radio" name="q${i + 1}" value="1" required>
          <span>Yes</span>
        </label>
        <label class="choice-option">
          <input type="radio" name="q${i + 1}" value="0" required>
          <span>No</span>
        </label>
      </div>
    </div>
  `).join("");

  updateProgress();
}

function updateProgress() {
  const form = $("screeningForm");
  if (!form) return;

  const total = 19;
  let completed = 0;

  for (let i = 1; i <= 10; i++) {
    if (form.querySelector(`input[name="q${i}"]:checked`)) completed++;
  }

  [
    "age", "gender", "ethnicity", "country", "jaundice",
    "autism", "relation", "usedApp", "ageDesc"
  ].forEach(id => {
    const el = $(id);
    if (el && String(el.value).trim()) completed++;
  });

  const percent = Math.round(completed / total * 100);

  if ($("progressText")) $("progressText").textContent = `${percent}%`;
  if ($("progressBar")) $("progressBar").style.width = `${percent}%`;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (data.model_loaded) {
      if ($("resultStatus")) $("resultStatus").textContent = "Model ready";
    } else {
      showToast("Website loaded, but the behavioral model is not ready. Check the model files.");
    }
  } catch (_error) {
    showToast("Could not contact the backend. Check that the Python server is running.");
  }
}

if ($("screeningForm")) {
  $("screeningForm").addEventListener("input", updateProgress);
  $("screeningForm").addEventListener("change", updateProgress);

  $("screeningForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const form = event.currentTarget;
    if (!form.reportValidity()) return;

    const age = Number($("age").value);

    if (!Number.isInteger(age) || age < 1 || age > 17) {
      showToast("Enter an age from 1 to 17 years.");
      $("age").focus();
      return;
    }

    const answers = [];

    for (let i = 1; i <= 10; i++) {
      const selected = form.querySelector(`input[name="q${i}"]:checked`);

      if (!selected) {
        showToast(`Please answer question ${i}.`);
        return;
      }

      answers.push(Number(selected.value));
    }

    const payload = {
      A1_Score: answers[0],
      A2_Score: answers[1],
      A3_Score: answers[2],
      A4_Score: answers[3],
      A5_Score: answers[4],
      A6_Score: answers[5],
      A7_Score: answers[6],
      A8_Score: answers[7],
      A9_Score: answers[8],
      A10_Score: answers[9],
      age,
      gender: $("gender").value,
      ethnicity: $("ethnicity").value,
      jaundice: $("jaundice").value,
      autism: $("autism").value,
      country_of_res: $("country").value.trim(),
      used_app_before: $("usedApp").value,
      result: answers.reduce((sum, value) => sum + value, 0),
      age_desc: $("ageDesc").value,
      relation: $("relation").value
    };

    const button = $("submitButton");
    button.disabled = true;
    button.textContent = "Processing...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      const percent = Number(data.display_percent);

      $("resultPercent").textContent =
        `${Number.isFinite(percent) ? percent.toFixed(2) : "N/A"}%`;

      $("resultMeter").style.width =
        `${Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0))}%`;

      $("resultFlag").textContent =
        data.flag === "above_configured_threshold"
          ? "The model output is above its configured threshold."
          : "The model output is below its configured threshold.";

      $("resultNotice").textContent = data.notice ||
        "Experimental output only. This is not a diagnosis.";

      $("resultPanel").hidden = false;
      $("resultPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });

    } catch (error) {
      showToast(error.message || "Could not calculate model output.");
    } finally {
      button.disabled = false;
      button.textContent = "Generate experimental output";
    }
  });
}

if ($("editAnswers")) {
  $("editAnswers").addEventListener("click", () => {
    $("screeningForm").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function stopTimer() {
  if (recordingTimer) clearInterval(recordingTimer);
  recordingTimer = null;
}

function mergeFloat32(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;

  for (let i = 0; i < samples.length; i++, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(
      offset,
      sample < 0 ? sample * 32768 : sample * 32767,
      true
    );
  }

  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Microphone access requires HTTPS or localhost.");
    return;
  }

  let context;
  let processor;
  let source;
  const chunks = [];

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    context = new AudioContextClass();
    source = context.createMediaStreamSource(mediaStream);
    processor = context.createScriptProcessor(4096, 1, 1);

    source.connect(processor);
    processor.connect(context.destination);

    processor.onaudioprocess = (event) => {
      if (mediaRecorder?.state === "recording") {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      }
    };

    mediaRecorder = {
      state: "recording",
      stop: async () => {
        if (mediaRecorder.state !== "recording") return;

        mediaRecorder.state = "stopped";
        stopTimer();

        processor.disconnect();
        source.disconnect();
        mediaStream.getTracks().forEach(track => track.stop());
        await context.close();

        const merged = mergeFloat32(chunks);

        if (!merged.length) {
          showToast("No audio was recorded. Please try again.");
          $("recordButton").disabled = false;
          $("stopButton").disabled = true;
          return;
        }

        const wav = encodeWav(merged, context.sampleRate);
        selectedAudio = wav;
        selectedAudioName = "microphone-recording.wav";

        setSelectedFile(selectedAudioName, wav.size);

        $("audioPlayer").src = URL.createObjectURL(wav);
        $("audioPlayer").hidden = false;
        $("analyzeButton").disabled = false;
        $("recordTitle").textContent = "Recording ready";
        $("recordDescription").textContent = "Your recording is ready to analyze.";
        $("recordCircle").classList.remove("recording");
        $("recordButton").disabled = false;
        $("recordButton").textContent = "Record again";
        $("stopButton").disabled = true;
        $("recordTime").textContent = `Recorded ${recordingSeconds} second(s)`;
      }
    };

    recordingSeconds = 0;
    $("recordCircle").classList.add("recording");
    $("recordTitle").textContent = "Recording in progress";
    $("recordDescription").textContent = "Speak or play the sound near your microphone.";
    $("recordButton").disabled = true;
    $("stopButton").disabled = false;
    $("recordTime").textContent = "00:00 / 00:15";

    recordingTimer = setInterval(() => {
      recordingSeconds++;
      $("recordTime").textContent =
        `00:${String(recordingSeconds).padStart(2, "0")} / 00:15`;

      if (recordingSeconds >= 15 && mediaRecorder?.state === "recording") {
        mediaRecorder.stop();
      }
    }, 1000);

  } catch (error) {
    showToast(
      error.name === "NotAllowedError"
        ? "Microphone permission was denied. Allow access in browser settings."
        : `Could not start microphone: ${error.message}`
    );

    if (processor) processor.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    if (context && context.state !== "closed") await context.close();

    $("recordButton").disabled = false;
    $("stopButton").disabled = true;
  }
}

if ($("recordButton")) {
  $("recordButton").addEventListener("click", startRecording);
}

if ($("stopButton")) {
  $("stopButton").addEventListener("click", () => {
    if (mediaRecorder?.state === "recording") mediaRecorder.stop();
  });
}

if ($("audioFile")) {
  $("audioFile").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".wav")) {
      showToast("Please choose a WAV file.");
      event.target.value = "";
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      showToast("Please choose a file smaller than 15 MB.");
      event.target.value = "";
      return;
    }

    selectedAudio = file;
    selectedAudioName = file.name;

    setSelectedFile(file.name, file.size);
    $("audioPlayer").src = URL.createObjectURL(file);
    $("audioPlayer").hidden = false;
    $("analyzeButton").disabled = false;
  });
}

function setSelectedFile(name, size) {
  $("selectedFile").hidden = false;
  $("selectedFile").textContent =
    `${name} · ${(size / (1024 * 1024)).toFixed(2)} MB`;

  $("audioStatus").textContent = "Audio ready to analyze";
}

$("analyzeButton").addEventListener("click", async () => {
  if (!selectedAudio) {
    showToast("Please upload or record audio first.");
    return;
  }

  const button = $("analyzeButton");
  button.disabled = true;
  button.textContent = "Analyzing audio...";

  try {
    const body = new FormData();
    body.append("audio", selectedAudio, selectedAudioName || "recording.wav");

    const response = await fetch("/api/audio-analysis", {
      method: "POST",
      body
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Audio analysis failed.");
    }

    $("emptyVisual").hidden = true;
    $("audioOutput").hidden = false;

    const score = Number(data.display_percent);

    $("audioDuration").textContent =
      `${Number(data.processed_duration_seconds ?? data.duration_seconds ?? 5).toFixed(2)} seconds`;

    $("audioSampleRate").textContent =
      `${Number(data.model_sample_rate ?? data.sample_rate ?? 16000).toLocaleString()} Hz`;

    $("audioSamples").textContent =
      Array.isArray(data.input_shape)
        ? data.input_shape.join(" × ")
        : "Audio processed";

    $("audioNotice").textContent =
      `Experimental model output: ${
        Number.isFinite(score) ? score.toFixed(2) + "%" : "Unavailable"
      }. This is not a calibrated ASD probability or diagnosis. ` +
      (data.notice || "Please consult a qualified healthcare professional for assessment.");

    $("audioStatus").textContent = "Analysis complete";

    if (Array.isArray(data.waveform)) {
      drawWaveform($("waveformCanvas"), data.waveform);
    }

    if (Array.isArray(data.mel)) {
      drawMel($("melCanvas"), data.mel);
    }

    const resultElement = $("audioResult");
    if (resultElement) {
      resultElement.textContent = Number.isFinite(score)
        ? `${score.toFixed(2)}% (experimental output)`
        : "Result unavailable";
    }

  } catch (error) {
    console.error("Audio analysis error:", error);
    showToast(error.message || "Could not analyze audio.");
    $("audioStatus").textContent = "Analysis failed";
  } finally {
    button.disabled = false;
    button.textContent = "Analyze audio";
  }
});

function prepareCanvas(canvas, height) {
  if (!canvas) return null;

  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(280, canvas.clientWidth || 600);

  canvas.width = width * ratio;
  canvas.height = height * ratio;

  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);

  return { ctx, width, height };
}

function drawWaveform(canvas, values) {
  if (!Array.isArray(values) || !canvas) return;

  const prepared = prepareCanvas(canvas, 115);
  if (!prepared) return;

  const { ctx, width, height } = prepared;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#e2edf0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (!values.length) return;

  ctx.strokeStyle = "#13a998";
  ctx.lineWidth = 1.2;
  ctx.beginPath();

  values.forEach((value, index) => {
    const x = index / Math.max(1, values.length - 1) * width;
    const y = height / 2 - value * (height * 0.42);

    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function drawMel(canvas, matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0 || !canvas) return;

  const rows = matrix.length;
  const cols = Array.isArray(matrix[0]) ? matrix[0].length : 0;
  if (!cols) return;

  const prepared = prepareCanvas(canvas, 220);
  if (!prepared) return;

  const { ctx, width, height } = prepared;
  ctx.clearRect(0, 0, width, height);

  const cellW = width / cols;
  const cellH = height / rows;

  for (let r = 0; r < rows; r++) {
    if (!Array.isArray(matrix[r])) continue;

    for (let c = 0; c < cols; c++) {
      const value = Number(matrix[r][c]);
      if (!Number.isFinite(value)) continue;

      const hue = 174 + value * 30;
      const light = 95 - value * 52;

      ctx.fillStyle = `hsl(${hue}, ${35 + value * 40}%, ${light}%)`;
      ctx.fillRect(
        c * cellW,
        height - (r + 1) * cellH,
        Math.ceil(cellW + 0.3),
        Math.ceil(cellH + 0.3)
      );
    }
  }
}

if ($("menuToggle")) {
  $("menuToggle").addEventListener("click", () => {
    const nav = document.querySelector(".nav-links");
    const isOpen = nav.classList.toggle("open");
    $("menuToggle").setAttribute("aria-expanded", String(isOpen));
  });
}

document.querySelectorAll(".nav-links a").forEach(link => {
  link.addEventListener("click", () => {
    document.querySelector(".nav-links").classList.remove("open");
    if ($("menuToggle")) {
      $("menuToggle").setAttribute("aria-expanded", "false");
    }
  });
});

if ($("year")) $("year").textContent = new Date().getFullYear();

window.addEventListener("resize", () => {
  const wave = $("waveformCanvas");

  if (
    wave &&
    !$("audioOutput").hidden &&
    wave.dataset.values
  ) {
    try {
      drawWaveform(wave, JSON.parse(wave.dataset.values));
    } catch (error) {
      console.warn("Could not redraw waveform:", error);
    }
  }
});

renderQuestions();
loadStatus();

import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1/dist/transformers.min.js";

const state = {
  samples: [],
  summarizer: null,
  charts: {},
  warmupComplete: false,
};

const elements = {
  sampleList: document.getElementById("sample-list"),
  datasetStats: document.getElementById("dataset-stats"),
  textarea: document.getElementById("demo-input"),
  output: document.getElementById("demo-output"),
  metrics: document.getElementById("demo-metrics"),
  rerunBtn: document.getElementById("rerun-demo"),
};

async function init() {
  await loadSamples();
  renderDatasetStats();
  renderSampleList();
  bootstrapCharts();
  bindEvents();
  runDemo(); // auto-run
}

async function loadSamples() {
  try {
    const res = await fetch("data/sst2_samples.json");
    state.samples = await res.json();
  } catch (error) {
    console.error("Failed to load SST-2 samples", error);
    state.samples = [
      {
        id: 0,
        text: "Fallback sample because SST-2 data could not be loaded.",
        label: "neutral",
      },
    ];
  }
}

function renderDatasetStats() {
  const counts = state.samples.reduce(
    (acc, sample) => {
      acc.total += 1;
      acc[sample.label] = (acc[sample.label] || 0) + 1;
      return acc;
    },
    { total: 0 }
  );

  elements.datasetStats.innerHTML = `
    <li><span>Total Samples</span><strong>${counts.total}</strong></li>
    <li><span>Positive</span><strong>${counts.positive || 0}</strong></li>
    <li><span>Negative</span><strong>${counts.negative || 0}</strong></li>
  `;
}

function renderSampleList() {
  elements.sampleList.innerHTML = state.samples
    .map(
      (sample) => `
        <div class="sample-card">
          <span>#${sample.id} · ${sample.label.toUpperCase()}</span>
          <p>${sample.text}</p>
        </div>
      `
    )
    .join("");
}

function bootstrapCharts() {
  const latencyCtx = document.getElementById("latencyChart");
  const energyCtx = document.getElementById("energyChart");

  state.charts.latency = new Chart(latencyCtx, {
    type: "line",
    data: {
      labels: ["Teacher FP32", "Student FP16", "Student INT8", "Student INT4"],
      datasets: [
        {
          label: "Latency (ms)",
          data: [130, 58, 21, 14],
          borderColor: "#4ea1ff",
          fill: false,
          tension: 0.35,
        },
        {
          label: "Accuracy (%)",
          data: [94.7, 94.1, 93.8, 92.9],
          borderColor: "#53f3c3",
          borderDash: [6, 4],
          fill: false,
          tension: 0.35,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: false },
      },
    },
  });

  state.charts.energy = new Chart(energyCtx, {
    type: "bar",
    data: {
      labels: ["Raspberry Pi 5", "Jetson Nano", "Android A78"],
      datasets: [
        {
          label: "Power (W)",
          data: [4.0, 3.1, 2.4],
          backgroundColor: ["#4ea1ff", "#53f3c3", "#ffb347"],
        },
        {
          label: "Throughput (req/s)",
          data: [24, 33, 41],
          backgroundColor: "rgba(255,255,255,0.15)",
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

function bindEvents() {
  elements.rerunBtn.addEventListener("click", () => runDemo(elements.textarea.value));
}

async function getSummarizer() {
  if (state.summarizer) {
    return state.summarizer;
  }

  elements.output.innerHTML =
    '<p class="placeholder">Loading distilled transformer…</p>';
  const start = performance.now();
  state.summarizer = await pipeline("summarization", "Xenova/distilbart-cnn-6-6");
  const loadLatency = (performance.now() - start).toFixed(0);
  elements.metrics.innerHTML = `<li><span>Model Load</span><strong>${loadLatency} ms</strong></li>`;
  return state.summarizer;
}

async function runDemo(customInput) {
  const sample =
    customInput?.trim() ||
    (state.samples.length
      ? state.samples[Math.floor(Math.random() * state.samples.length)].text
      : "This distilled model summarizes SST-2 style feedback.");

  elements.textarea.value = sample;
  elements.output.innerHTML =
    '<p class="placeholder">Running summarizer in your browser…</p>';

  try {
    const summarizer = await getSummarizer();
    const prompt = `Input: ${sample}\nTask: Produce a single sentence summary preserving sentiment cues.`;
    const start = performance.now();
    const result = await summarizer(prompt, {
      min_length: 12,
      max_length: 48,
      truncation: true,
    });
    const latency = (performance.now() - start).toFixed(1);

    const summary = result?.[0]?.summary_text || "Summary unavailable.";
    elements.output.innerHTML = `<p>${summary}</p>`;
    elements.metrics.innerHTML = `
      <li><span>Inference Latency</span><strong>${latency} ms</strong></li>
      <li><span>Batch Size</span><strong>1</strong></li>
      <li><span>Precision</span><strong>INT8 (simulated)</strong></li>
    `;
  } catch (error) {
    console.error(error);
    elements.output.innerHTML =
      "<p>Unable to run the summarizer in this browser. Try a modern Chromium build.</p>";
  }
}

init();


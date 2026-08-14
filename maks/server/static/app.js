const hud = document.getElementById("hud");
const connDot = document.getElementById("conn-dot");
const connLabel = document.getElementById("conn-label");
const stateLabel = document.getElementById("state-label");
const agentLabel = document.getElementById("agent-label");

const transcript = document.getElementById("transcript");
const transcriptUser = document.getElementById("transcript-user");
const transcriptMaks = document.getElementById("transcript-maks");

function setState(state) {
  hud.dataset.state = state;
  stateLabel.textContent = state;
}

function showAgent(name) {
  if (!name) return;
  agentLabel.hidden = false;
  agentLabel.textContent = name.replace(/_/g, " ");
}

function setTranscript(userText, maksText) {
  transcriptUser.textContent = userText ? userText : "";
  transcriptMaks.textContent = maksText ? maksText : "";
  transcript.hidden = !userText && !maksText;
}

// --- Telemetry panels ---
const statEls = {
  cpu: document.getElementById("stat-cpu"),
  ram: document.getElementById("stat-ram"),
  ramDetail: document.getElementById("stat-ram-detail"),
  disk: document.getElementById("stat-disk"),
  diskDetail: document.getElementById("stat-disk-detail"),
  netUp: document.getElementById("stat-net-up"),
  netDown: document.getElementById("stat-net-down"),
  uptime: document.getElementById("stat-uptime"),
  battery: document.getElementById("stat-battery"),
  weather: document.getElementById("stat-weather"),
};
const barEls = {
  cpu: document.getElementById("bar-cpu"),
  ram: document.getElementById("bar-ram"),
  disk: document.getElementById("bar-disk"),
};
const batteryItem = document.getElementById("battery-item");
const batteryDivider = document.getElementById("battery-divider");
const weatherItem = document.getElementById("weather-item");
const weatherDivider = document.getElementById("weather-divider");

function updateStats(data) {
  if (typeof data.cpu_percent === "number") {
    statEls.cpu.textContent = data.cpu_percent.toFixed(0);
    barEls.cpu.style.width = `${Math.min(100, data.cpu_percent)}%`;
  }
  if (typeof data.ram_percent === "number") {
    statEls.ram.textContent = data.ram_percent.toFixed(0);
    barEls.ram.style.width = `${Math.min(100, data.ram_percent)}%`;
    statEls.ramDetail.textContent = `${data.ram_used_gb} / ${data.ram_total_gb} GB`;
  }
  if (typeof data.disk_percent === "number") {
    statEls.disk.textContent = data.disk_percent.toFixed(0);
    barEls.disk.style.width = `${Math.min(100, data.disk_percent)}%`;
    statEls.diskDetail.textContent = `${data.disk_used_gb} / ${data.disk_total_gb} GB`;
  }
  if (typeof data.net_sent_kbps === "number") statEls.netUp.textContent = data.net_sent_kbps.toFixed(1);
  if (typeof data.net_recv_kbps === "number") statEls.netDown.textContent = data.net_recv_kbps.toFixed(1);
  if (data.uptime) statEls.uptime.textContent = data.uptime;

  if (data.battery_percent === null || data.battery_percent === undefined) {
    batteryItem.hidden = true;
    batteryDivider.hidden = true;
  } else {
    batteryItem.hidden = false;
    batteryDivider.hidden = false;
    statEls.battery.textContent = Math.round(data.battery_percent);
  }

  if (data.weather) {
    weatherItem.hidden = false;
    weatherDivider.hidden = false;
    statEls.weather.textContent = data.weather;
  }
}

// --- Live clock (ticks client-side, independent of the stats push) ---
const clockEl = document.getElementById("clock");
function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString([], { hour12: false });
}
tickClock();
setInterval(tickClock, 1000);

function handleEvent(evt) {
  const { type, data } = evt;
  switch (type) {
    case "status":
      if (data && data.state) setState(data.state);
      if (data && data.agent) showAgent(data.agent);
      break;
    case "wake":
      setState("wake");
      setTranscript("", "");
      break;
    case "ready":
      setState("ready");
      break;
    case "listening":
      setState("listening");
      break;
    case "thinking":
      setState("thinking");
      setTranscript(data.text, "");
      break;
    case "handoff":
      setTranscript(transcriptUser.textContent, `Handing this off to Claude: "${data.task}"`);
      break;
    case "speaking":
      setState("speaking");
      break;
    case "reply":
      setTranscript(transcriptUser.textContent, data.text);
      if (data.agent) showAgent(data.agent);
      break;
    case "idle":
      setState("idle");
      break;
    case "stats":
      updateStats(data);
      break;
    default:
      break;
  }
}

let ws;
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connDot.classList.add("ok");
    connLabel.textContent = "connected";
  };
  ws.onclose = () => {
    connDot.classList.remove("ok");
    connLabel.textContent = "reconnecting…";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (msg) => {
    try {
      handleEvent(JSON.parse(msg.data));
    } catch (e) {
      console.error("bad event", e);
    }
  };
}
connect();

// --- Collapsed text-input fallback ---
const inputToggle = document.getElementById("input-toggle");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

inputToggle.addEventListener("click", () => {
  const isHidden = chatForm.hidden;
  chatForm.hidden = !isHidden;
  if (isHidden) chatInput.focus();
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    chatForm.hidden = true;
    chatInput.blur();
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  chatForm.hidden = true;
  try {
    await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err) {
    setTranscript(text, `Failed to send: ${err}`);
  }
});

// --- Settings slide-out (persona editor) ---
const settingsToggle = document.getElementById("settings-toggle");
const settingsClose = document.getElementById("settings-close");
const settingsPanel = document.getElementById("settings-panel");
const settingsScrim = document.getElementById("settings-scrim");
const personaText = document.getElementById("persona-text");
const personaSave = document.getElementById("persona-save");
const personaStatus = document.getElementById("persona-status");

async function loadPersona() {
  const resp = await fetch("/system-prompt");
  const data = await resp.json();
  personaText.value = data.content || "";
}

function openSettings() {
  settingsPanel.classList.add("open");
  settingsScrim.hidden = false;
  loadPersona();
}

function closeSettings() {
  settingsPanel.classList.remove("open");
  settingsScrim.hidden = true;
}

settingsToggle.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsScrim.addEventListener("click", closeSettings);

personaSave.addEventListener("click", async () => {
  await fetch("/system-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: personaText.value }),
  });
  personaStatus.textContent = "Saved.";
  setTimeout(() => (personaStatus.textContent = ""), 2000);
});

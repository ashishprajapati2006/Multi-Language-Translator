/**
 * OmniTranslate AI - Core Application Logic
 * Integrates Web Speech API, Audio Synthesis, History, Comparison Mode, and API client.
 */

// Default language dictionary with flags and locale tags
const LANGUAGES = [
  { code: "en_US", name: "English", native: "English", flag: "🇬🇧", speechLocale: "en-US", custom: true },
  { code: "es_ES", name: "Spanish", native: "Español", flag: "🇪🇸", speechLocale: "es-ES", custom: true },
  { code: "fr_FR", name: "French", native: "Français", flag: "🇫🇷", speechLocale: "fr-FR", custom: false },
  { code: "de_DE", name: "German", native: "Deutsch", flag: "🇩🇪", speechLocale: "de-DE", custom: false },
  { code: "hi_IN", name: "Hindi", native: "हिन्दी", flag: "🇮🇳", speechLocale: "hi-IN", custom: false },
  { code: "ja_JP", name: "Japanese", native: "日本語", flag: "🇯🇵", speechLocale: "ja-JP", custom: false },
  { code: "it_IT", name: "Italian", native: "Italiano", flag: "🇮🇹", speechLocale: "it-IT", custom: false },
  { code: "pt_PT", name: "Portuguese", native: "Português", flag: "🇵🇹", speechLocale: "pt-PT", custom: false },
  { code: "nl_NL", name: "Dutch", native: "Nederlands", flag: "🇳🇱", speechLocale: "nl-NL", custom: false },
  { code: "ar_SA", name: "Arabic", native: "العربية", flag: "🇸🇦", speechLocale: "ar-SA", custom: false, rtl: true }
];

const POPULAR_LANG_CODES = ["en_US", "es_ES", "fr_FR", "de_DE", "hi_IN", "ja_JP"];

// State Management
const state = {
  sourceLang: "en_US",
  targetLang: "es_ES",
  viewMode: "unified",      // 'unified' or 'comparison'
  selectedEngine: "mbart",  // 'mbart' or 'custom'
  isTranslating: false,
  isListening: false,
  isSpeaking: false,
  backendStatus: "checking", // 'online', 'coldstart', 'offline'
  history: []
};

// ==================== INITIALIZATION ====================

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initLanguageSelectors();
  initHistory();
  initEventListeners();
  checkBackendHealth();

  // Periodic health check every 45s
  setInterval(checkBackendHealth, 45000);
});

// ==================== THEME MANAGEMENT ====================

function initTheme() {
  const savedTheme = localStorage.getItem("translator_theme");
  const theme = savedTheme || "dark";
  setTheme(theme);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("translator_theme", theme);
  const themeIcon = document.getElementById("theme-toggle-icon");
  if (themeIcon) {
    themeIcon.innerHTML = theme === "dark"
      ? `<path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>`
      : `<path d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>`;
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  setTheme(current === "dark" ? "light" : "dark");
}

// ==================== LANGUAGE SELECTORS ====================

function initLanguageSelectors() {
  const sourceSelect = document.getElementById("source-lang-select");
  const targetSelect = document.getElementById("target-lang-select");
  const compSourceSelect = document.getElementById("comp-source-select");
  const compTargetSelect = document.getElementById("comp-target-select");

  const populate = (selectElem, selectedCode) => {
    if (!selectElem) return;
    selectElem.innerHTML = "";
    LANGUAGES.forEach(lang => {
      const opt = document.createElement("option");
      opt.value = lang.code;
      opt.textContent = `${lang.flag} ${lang.name}`;
      if (lang.code === selectedCode) opt.selected = true;
      selectElem.appendChild(opt);
    });
  };

  populate(sourceSelect, state.sourceLang);
  populate(targetSelect, state.targetLang);
  populate(compSourceSelect, state.sourceLang);
  populate(compTargetSelect, state.targetLang);

  renderQuickChips("source", state.sourceLang);
  renderQuickChips("target", state.targetLang);
}

function renderQuickChips(type, activeCode) {
  const container = document.getElementById(`${type}-chips-container`);
  if (!container) return;
  container.innerHTML = "";

  POPULAR_LANG_CODES.forEach(code => {
    const lang = LANGUAGES.find(l => l.code === code);
    if (!lang) return;

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `chip ${lang.code === activeCode ? "active" : ""}`;
    chip.textContent = `${lang.flag} ${lang.name}`;
    chip.onclick = () => {
      if (type === "source") setSourceLanguage(lang.code);
      else setTargetLanguage(lang.code);
    };
    container.appendChild(chip);
  });
}

function setSourceLanguage(code) {
  state.sourceLang = code;
  const select = document.getElementById("source-lang-select");
  const compSelect = document.getElementById("comp-source-select");
  if (select) select.value = code;
  if (compSelect) compSelect.value = code;
  renderQuickChips("source", code);
  updateModelCompatibility();
}

function setTargetLanguage(code) {
  state.targetLang = code;
  const select = document.getElementById("target-lang-select");
  const compSelect = document.getElementById("comp-target-select");
  if (select) select.value = code;
  if (compSelect) compSelect.value = code;
  renderQuickChips("target", code);
  updateModelCompatibility();
}

function swapLanguages() {
  const temp = state.sourceLang;
  setSourceLanguage(state.targetLang);
  setTargetLanguage(temp);

  // Also swap text if both areas have content
  const srcInput = document.getElementById("source-text-input");
  const targetOutput = document.getElementById("target-translation-output");
  if (srcInput && targetOutput && targetOutput.textContent.trim() && !targetOutput.classList.contains("empty")) {
    const prevSrc = srcInput.value;
    srcInput.value = targetOutput.textContent.trim();
    targetOutput.textContent = prevSrc;
    updateCounters();
  }

  showToast("Languages swapped", "info");
}

function updateModelCompatibility() {
  const customOption = document.getElementById("engine-custom-opt");
  const isCustomSupported = (state.sourceLang === "en_US" && state.targetLang === "es_ES");

  if (customOption) {
    if (!isCustomSupported) {
      customOption.title = "Custom model only supports English → Spanish";
      if (state.selectedEngine === "custom") {
        setEngine("mbart");
        showToast("Switched to mBART (Custom model supports en_US → es_ES only)", "info");
      }
    } else {
      customOption.title = "Custom Seq2Seq LSTM model ready";
    }
  }

  // Update comparison view Custom banner if not en->es
  const compCustomNote = document.getElementById("comp-custom-warning");
  if (compCustomNote) {
    compCustomNote.style.display = isCustomSupported ? "none" : "block";
  }
}

function setEngine(engine) {
  if (engine === "custom" && !(state.sourceLang === "en_US" && state.targetLang === "es_ES")) {
    showToast("Custom model only supports English → Spanish translation.", "info");
    return;
  }
  state.selectedEngine = engine;
  document.querySelectorAll(".engine-option").forEach(el => {
    el.classList.toggle("active", el.dataset.engine === engine);
  });
}

// ==================== MODE SWITCHING ====================

function setViewMode(mode) {
  state.viewMode = mode;
  document.getElementById("mode-unified-btn")?.classList.toggle("active", mode === "unified");
  document.getElementById("mode-comparison-btn")?.classList.toggle("active", mode === "comparison");

  const unifiedWorkbench = document.getElementById("unified-workbench");
  const compContainer = document.getElementById("comparison-container");
  const enginePill = document.getElementById("engine-selector-pill");

  if (mode === "unified") {
    if (unifiedWorkbench) unifiedWorkbench.style.display = "grid";
    if (compContainer) compContainer.classList.remove("active");
    if (enginePill) enginePill.style.display = "flex";
  } else {
    if (unifiedWorkbench) unifiedWorkbench.style.display = "none";
    if (compContainer) compContainer.classList.add("active");
    if (enginePill) enginePill.style.display = "none";
  }
}

// ==================== TRANSLATION EXECUTION ====================

async function executeTranslation() {
  const srcInput = state.viewMode === "unified"
    ? document.getElementById("source-text-input")
    : document.getElementById("comp-source-text-input");

  const text = srcInput ? srcInput.value.trim() : "";
  if (!text) {
    showToast("Please enter text to translate.", "info");
    srcInput?.focus();
    return;
  }

  if (state.sourceLang === state.targetLang) {
    showToast("Source and target languages are identical.", "info");
    return;
  }

  state.isTranslating = true;
  updateTranslateButtons(true);

  if (state.viewMode === "unified") {
    await runUnifiedTranslation(text);
  } else {
    await runComparisonTranslation(text);
  }

  state.isTranslating = false;
  updateTranslateButtons(false);
}

async function runUnifiedTranslation(text) {
  const outputEl = document.getElementById("target-translation-output");
  const latencyEl = document.getElementById("unified-latency-badge");
  const actionTools = document.getElementById("target-action-tools");

  if (outputEl) {
    outputEl.classList.remove("empty");
    outputEl.innerHTML = `
      <div class="skeleton-line long"></div>
      <div class="skeleton-line medium"></div>
      <div class="skeleton-line short"></div>
    `;
  }
  if (actionTools) actionTools.style.visibility = "hidden";
  if (latencyEl) latencyEl.style.display = "none";

  const startTime = performance.now();

  try {
    let resultText = "";
    let durationMs = 0;

    if (state.selectedEngine === "custom") {
      const res = await API.translateCustom(text, state.sourceLang, state.targetLang);
      resultText = res.translation;
      durationMs = res.duration_ms || Math.round(performance.now() - startTime);
    } else {
      const res = await API.translateMbart(text, state.sourceLang, state.targetLang);
      resultText = res.translation;
      durationMs = res.duration_ms || Math.round(performance.now() - startTime);
    }

    if (outputEl) {
      outputEl.textContent = resultText;
    }
    if (latencyEl) {
      latencyEl.textContent = `⚡ ${durationMs}ms`;
      latencyEl.style.display = "flex";
    }
    if (actionTools) actionTools.style.visibility = "visible";

    saveToHistory({
      text,
      translation: resultText,
      sourceLang: state.sourceLang,
      targetLang: state.targetLang,
      model: state.selectedEngine
    });

  } catch (err) {
    if (outputEl) {
      outputEl.innerHTML = `<span style="color: #ef4444;">❌ Translation Error: ${err.message}</span>`;
    }
    showToast(err.message, "error");
  }
}

async function runComparisonTranslation(text) {
  const customOut = document.getElementById("comp-custom-output");
  const mbartOut = document.getElementById("comp-mbart-output");
  const customLat = document.getElementById("comp-custom-latency");
  const mbartLat = document.getElementById("comp-mbart-latency");

  const skeletonHtml = `
    <div class="skeleton-line long"></div>
    <div class="skeleton-line medium"></div>
  `;

  if (customOut) customOut.innerHTML = skeletonHtml;
  if (mbartOut) mbartOut.innerHTML = skeletonHtml;

  try {
    const res = await API.translate({
      text,
      sourceLang: state.sourceLang,
      targetLang: state.targetLang,
      modelType: "both"
    });

    const trans = res.translations || {};

    // Custom result
    if (trans.custom) {
      if (trans.custom.error) {
        customOut.innerHTML = `<span style="color: #ef4444;">${trans.custom.error}</span>`;
      } else {
        customOut.textContent = trans.custom.translation;
        if (customLat) customLat.textContent = `⚡ ${trans.custom.duration_ms}ms`;
      }
    } else {
      customOut.innerHTML = `<span style="color: var(--text-subtle); font-style: italic;">Not available for this language pair (supports en_US → es_ES only)</span>`;
      if (customLat) customLat.textContent = `--`;
    }

    // mBART result
    if (trans.mbart) {
      if (trans.mbart.error) {
        mbartOut.innerHTML = `<span style="color: #ef4444;">${trans.mbart.error}</span>`;
      } else {
        mbartOut.textContent = trans.mbart.translation;
        if (mbartLat) mbartLat.textContent = `⚡ ${trans.mbart.duration_ms}ms`;
      }
    }

    saveToHistory({
      text,
      translation: trans.mbart?.translation || trans.custom?.translation || "",
      sourceLang: state.sourceLang,
      targetLang: state.targetLang,
      model: "comparison"
    });

  } catch (err) {
    if (customOut) customOut.textContent = "Error";
    if (mbartOut) mbartOut.textContent = "Error";
    showToast(err.message, "error");
  }
}

function updateTranslateButtons(loading) {
  const btns = [
    document.getElementById("btn-translate-main"),
    document.getElementById("btn-translate-comp")
  ];
  btns.forEach(btn => {
    if (!btn) return;
    btn.disabled = loading;
    if (loading) {
      btn.innerHTML = `<div class="spinner"></div> Translating with AI...`;
    } else {
      btn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <path d="M5 12h14M12 5l7 7-7 7"/>
        </svg>
        Translate Now
      `;
    }
  });
}

// ==================== WEB SPEECH API (VOICE INPUT & OUTPUT) ====================

function toggleSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast("Speech Recognition is not supported in this browser. Please use Chrome or Edge.", "info");
    return;
  }

  const micBtn = document.getElementById("btn-voice-input");
  const textarea = state.viewMode === "unified"
    ? document.getElementById("source-text-input")
    : document.getElementById("comp-source-text-input");

  if (state.isListening) {
    if (window.activeRecognition) window.activeRecognition.stop();
    return;
  }

  const langObj = LANGUAGES.find(l => l.code === state.sourceLang);
  const recognition = new SpeechRecognition();
  recognition.lang = langObj?.speechLocale || "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;

  recognition.onstart = () => {
    state.isListening = true;
    micBtn?.classList.add("listening");
    showToast(`Listening in ${langObj?.name || "English"}...`, "info");
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    if (textarea) {
      textarea.value = transcript;
      updateCounters();
    }
  };

  recognition.onerror = (e) => {
    console.error("Speech Recognition Error:", e);
    showToast(`Microphone error: ${e.error}`, "error");
    state.isListening = false;
    micBtn?.classList.remove("listening");
  };

  recognition.onend = () => {
    state.isListening = false;
    micBtn?.classList.remove("listening");
    if (textarea && textarea.value.trim().length > 0) {
      executeTranslation();
    }
  };

  window.activeRecognition = recognition;
  recognition.start();
}

function speakText(elementId, langCode) {
  if (!("speechSynthesis" in window)) {
    showToast("Text-to-Speech not supported in your browser.", "info");
    return;
  }

  const el = document.getElementById(elementId);
  const text = el?.textContent?.trim() || el?.value?.trim();
  if (!text) return;

  window.speechSynthesis.cancel();

  const langObj = LANGUAGES.find(l => l.code === langCode);
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = langObj?.speechLocale || "en-US";
  utterance.rate = 0.95;

  const speakBtn = document.getElementById("btn-speak-target");
  speakBtn?.classList.add("speaking");

  utterance.onend = () => speakBtn?.classList.remove("speaking");
  utterance.onerror = () => speakBtn?.classList.remove("speaking");

  window.speechSynthesis.speak(utterance);
}

// ==================== CLIPBOARD & UTILITIES ====================

function copyText(elementId) {
  const el = document.getElementById(elementId);
  const text = el ? (el.value !== undefined ? el.value : el.textContent) : "";
  if (!text || text.trim() === "Translation will appear here...") {
    showToast("No text to copy.", "info");
    return;
  }

  navigator.clipboard.writeText(text.trim()).then(() => {
    showToast("Copied to clipboard!", "success");
  }).catch(() => {
    showToast("Failed to copy.", "error");
  });
}

function clearSourceText() {
  const src = document.getElementById("source-text-input");
  const compSrc = document.getElementById("comp-source-text-input");
  const out = document.getElementById("target-translation-output");
  const lat = document.getElementById("unified-latency-badge");

  if (src) src.value = "";
  if (compSrc) compSrc.value = "";
  if (out) {
    out.classList.add("empty");
    out.textContent = "Translation will appear here...";
  }
  if (lat) lat.style.display = "none";

  updateCounters();
}

function loadSample(phrase) {
  const src = state.viewMode === "unified"
    ? document.getElementById("source-text-input")
    : document.getElementById("comp-source-text-input");
  if (src) {
    src.value = phrase;
    updateCounters();
    executeTranslation();
  }
}

function updateCounters() {
  const src = document.getElementById("source-text-input");
  const text = src ? src.value : "";
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  const counterEl = document.getElementById("source-char-counter");
  if (counterEl) counterEl.textContent = `${chars} chars • ${words} words`;
}

// ==================== TRANSLATION HISTORY ====================

function initHistory() {
  try {
    const raw = localStorage.getItem("translator_history");
    state.history = raw ? JSON.parse(raw) : [];
  } catch {
    state.history = [];
  }
}

function saveToHistory(item) {
  if (!item.text || !item.translation) return;
  const entry = {
    ...item,
    id: Date.now(),
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };
  state.history.unshift(entry);
  if (state.history.length > 30) state.history.pop();
  try {
    localStorage.setItem("translator_history", JSON.stringify(state.history));
  } catch (e) {
    console.error("Failed to save history:", e);
  }
}

function toggleHistoryDrawer(open) {
  const drawer = document.getElementById("history-drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  if (open) {
    renderHistoryItems();
    drawer?.classList.add("active");
    backdrop?.classList.add("active");
  } else {
    drawer?.classList.remove("active");
    backdrop?.classList.remove("active");
  }
}

function renderHistoryItems() {
  const container = document.getElementById("history-items-container");
  if (!container) return;
  container.innerHTML = "";

  if (state.history.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; color: var(--text-subtle); padding: 40px 20px;">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 12px; opacity: 0.5;">
          <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
        <p>No translation history yet.</p>
      </div>
    `;
    return;
  }

  state.history.forEach(item => {
    const srcLang = LANGUAGES.find(l => l.code === item.sourceLang);
    const tgtLang = LANGUAGES.find(l => l.code === item.targetLang);

    const div = document.createElement("div");
    div.className = "history-item";
    div.innerHTML = `
      <div class="history-meta">
        <span>${srcLang?.flag || ""} ${srcLang?.name || item.sourceLang} → ${tgtLang?.flag || ""} ${tgtLang?.name || item.targetLang}</span>
        <span>${item.timestamp}</span>
      </div>
      <div class="history-source">${escapeHtml(item.text)}</div>
      <div class="history-translation">${escapeHtml(item.translation)}</div>
    `;
    div.onclick = () => {
      setSourceLanguage(item.sourceLang);
      setTargetLanguage(item.targetLang);
      const srcInput = document.getElementById("source-text-input");
      if (srcInput) srcInput.value = item.text;
      const targetOut = document.getElementById("target-translation-output");
      if (targetOut) {
        targetOut.classList.remove("empty");
        targetOut.textContent = item.translation;
      }
      updateCounters();
      toggleHistoryDrawer(false);
    };
    container.appendChild(div);
  });
}

function clearAllHistory() {
  state.history = [];
  localStorage.removeItem("translator_history");
  renderHistoryItems();
  showToast("Translation history cleared", "info");
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

// ==================== HEALTH & SETTINGS MODAL ====================

async function checkBackendHealth() {
  const pill = document.getElementById("backend-status-pill");
  const dot = document.getElementById("backend-status-dot");
  const text = document.getElementById("backend-status-text");

  const result = await API.checkHealth();
  if (result.ok) {
    state.backendStatus = "online";
    if (dot) {
      dot.className = "status-dot online";
    }
    if (text) {
      text.textContent = `Backend Live • ${result.pingMs}ms`;
    }
    if (pill) {
      pill.title = `Connected to ${window.CONFIG.getApiBaseUrl()}`;
    }
  } else {
    // Check if cold-starting or offline
    state.backendStatus = "coldstart";
    if (dot) {
      dot.className = "status-dot coldstart";
    }
    if (text) {
      text.textContent = "Backend Standby";
    }
    if (pill) {
      pill.title = `Backend standby or unreachable: ${result.error}`;
    }
  }
}

function openSettingsModal() {
  const modal = document.getElementById("settings-modal");
  const input = document.getElementById("backend-url-input");
  if (input) input.value = window.CONFIG.getApiBaseUrl();
  modal?.classList.add("active");
}

function closeSettingsModal() {
  document.getElementById("settings-modal")?.classList.remove("active");
}

async function testBackendUrl() {
  const input = document.getElementById("backend-url-input");
  const testBadge = document.getElementById("backend-test-result");
  const url = input?.value.trim();

  if (testBadge) {
    testBadge.style.display = "block";
    testBadge.innerHTML = `<span style="color: var(--accent-amber);">Pinging backend...</span>`;
  }

  window.CONFIG.setApiBaseUrl(url);
  const result = await API.checkHealth();

  if (testBadge) {
    if (result.ok) {
      testBadge.innerHTML = `<span style="color: var(--accent-emerald);">Connected! Latency: ${result.pingMs}ms</span>`;
      checkBackendHealth();
    } else {
      testBadge.innerHTML = `<span style="color: #ef4444;">Failed to connect: ${result.error}</span>`;
    }
  }
}

function saveSettings() {
  const input = document.getElementById("backend-url-input");
  window.CONFIG.setApiBaseUrl(input?.value.trim());
  showToast("Backend configuration saved", "success");
  closeSettingsModal();
  checkBackendHealth();
}

// ==================== TOAST NOTIFICATIONS ====================

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const iconMap = {
    success: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>`,
    error: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>`,
    info: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>`
  };

  toast.innerHTML = `
    ${iconMap[type] || iconMap.info}
    <span>${message}</span>
  `;

  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));

  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 350);
  }, 3500);
}

// ==================== EVENT LISTENERS & SHORTCUTS ====================

function initEventListeners() {
  // Source inputs character counters
  document.getElementById("source-text-input")?.addEventListener("input", updateCounters);
  document.getElementById("comp-source-text-input")?.addEventListener("input", updateCounters);

  // Keyboard shortcut: Ctrl + Enter / Cmd + Enter to translate
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      executeTranslation();
    }
    if (e.key === "Escape") {
      closeSettingsModal();
      toggleHistoryDrawer(false);
    }
  });

  // Language selectors change event
  document.getElementById("source-lang-select")?.addEventListener("change", (e) => {
    setSourceLanguage(e.target.value);
  });
  document.getElementById("target-lang-select")?.addEventListener("change", (e) => {
    setTargetLanguage(e.target.value);
  });
  document.getElementById("comp-source-select")?.addEventListener("change", (e) => {
    setSourceLanguage(e.target.value);
  });
  document.getElementById("comp-target-select")?.addEventListener("change", (e) => {
    setTargetLanguage(e.target.value);
  });
}

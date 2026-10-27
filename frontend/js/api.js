/**
 * API Service Client
 * Handles HTTP communication with the Flask / Render Backend
 */
const API = {
  /**
   * Helper fetch with timeout
   */
  async request(endpoint, options = {}, timeoutMs = 90000) {
    const baseUrl = window.CONFIG.getApiBaseUrl();
    const url = `${baseUrl}${endpoint}`;

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          ...(options.headers || {})
        },
        signal: controller.signal
      });

      clearTimeout(id);
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return data;
    } catch (err) {
      clearTimeout(id);
      if (err.name === "AbortError") {
        throw new Error("Request timed out. The backend server might be starting up from cold standby (can take 30-60s on Render free tier). Please retry.");
      }
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        throw new Error(`Cannot reach backend at ${baseUrl}. Please ensure the server is active or update the URL in Settings.`);
      }
      throw err;
    }
  },

  /**
   * Health Check / Ping
   */
  async checkHealth() {
    const startTime = performance.now();
    try {
      const data = await this.request("/health", { method: "GET" }, 15000);
      const pingMs = Math.round(performance.now() - startTime);
      return { ok: true, data, pingMs };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  },

  /**
   * Fetch Supported Languages
   */
  async getLanguages() {
    return await this.request("/api/languages", { method: "GET" });
  },

  /**
   * Translate using Unified API
   */
  async translate({ text, sourceLang, targetLang, modelType = "both" }) {
    return await this.request("/api/translate", {
      method: "POST",
      body: JSON.stringify({
        text,
        source_lang: sourceLang,
        target_lang: targetLang,
        model_type: modelType
      })
    });
  },

  /**
   * Translate using Custom Seq2Seq model
   */
  async translateCustom(text, sourceLang = "en_US", targetLang = "es_ES") {
    return await this.request("/api/translate/custom", {
      method: "POST",
      body: JSON.stringify({
        text,
        source_lang: sourceLang,
        target_lang: targetLang
      })
    });
  },

  /**
   * Translate using mBART model
   */
  async translateMbart(text, sourceLang, targetLang) {
    return await this.request("/api/translate/mbart", {
      method: "POST",
      body: JSON.stringify({
        text,
        source_lang: sourceLang,
        target_lang: targetLang
      })
    });
  }
};

window.API = API;

/**
 * Global Configuration for Translator Frontend
 * Allows dynamic configuration of the backend API endpoint (e.g. Render Web Service URL).
 */
const CONFIG = {
  // Default fallback URLs: if running on localhost, use local backend port 5000;
  // otherwise, default to Render URL if provided, or empty string for manual configuration.
  DEFAULT_LOCAL_URL: "http://localhost:5000",
  DEFAULT_RENDER_URL: "https://multi-language-translator-1-ik9b.onrender.com",
  STORAGE_KEY: "translator_backend_url",

  /**
   * Returns the currently configured Backend API Base URL.
   */
  getApiBaseUrl() {
    const saved = localStorage.getItem(this.STORAGE_KEY);
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, "");
    }
    // Auto-detect local development
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return this.DEFAULT_LOCAL_URL;
    }
    // Production default (Render)
    return this.DEFAULT_RENDER_URL;
  },

  /**
   * Save a custom backend URL to localStorage
   */
  setApiBaseUrl(url) {
    if (!url) {
      localStorage.removeItem(this.STORAGE_KEY);
    } else {
      const cleanUrl = url.trim().replace(/\/+$/, "");
      localStorage.setItem(this.STORAGE_KEY, cleanUrl);
    }
  },

  /**
   * Reset backend URL to automatic detection
   */
  resetApiBaseUrl() {
    localStorage.removeItem(this.STORAGE_KEY);
  }
};

window.CONFIG = CONFIG;

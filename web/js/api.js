/* api.js — thin fetch wrapper for /api/*. Exposes window.API. */
(function () {
  "use strict";

  async function request(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (_) {
      throw { code: "network", message: "Tidak dapat terhubung ke server. Periksa koneksi Anda." };
    }
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      const err = body && body.error
        ? body.error
        : { code: "internal", message: "Maaf, terjadi kendala teknis. Silakan coba lagi." };
      throw err;
    }
    return body;
  }

  window.API = {
    getMeta: function () {
      return request("api/meta");
    },
    getHealth: function () {
      return request("api/health");
    },
    postChat: function (message, sessionId, mode) {
      return request("api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message, session_id: sessionId, mode: mode }),
      });
    },
  };
})();

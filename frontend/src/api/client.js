// Backend REST + WebSocket istemcisi.
// Geliştirmede Vite proxy göreli yolları backend'e (8000) yönlendirir.
// Yayında (Cloudflare Pages vb.) backend farklı bir adreste (tünel)
// çalışır: adres localStorage'da tutulur, boşsa aynı origin kullanılır.

// URL'de ?backend=https://... varsa otomatik kaydet (tek tıkla kurulum).
// Örn: https://site/?backend=https://xxxx.trycloudflare.com
try {
  const qs = new URLSearchParams(location.search);
  const fromUrl = qs.get("backend");
  if (fromUrl) {
    localStorage.setItem("backend_url", fromUrl.trim().replace(/\/+$/, ""));
    qs.delete("backend");
    const clean =
      location.pathname + (qs.toString() ? `?${qs}` : "") + location.hash;
    history.replaceState(null, "", clean);
  }
} catch {
  /* localStorage kapalıysa sessizce geç */
}

export const backendBase = () =>
  (localStorage.getItem("backend_url") || "").replace(/\/+$/, "");

export const setBackendBase = (url) => {
  const clean = (url || "").trim().replace(/\/+$/, "");
  if (clean) localStorage.setItem("backend_url", clean);
  else localStorage.removeItem("backend_url");
};

// Snapshot görselleri gibi statik dosya yolları için tam URL üret
export const assetUrl = (path) => (path ? backendBase() + path : path);

const F = (path, opts) => fetch(backendBase() + path, opts);

const j = async (res) => {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.status === 204 ? null : res.json();
};

export const api = {
  health: () => F("/api/health").then(j),
  modelStatus: () => F("/api/reports/model-status").then(j),

  // Kameralar
  listCameras: () => F("/api/cameras").then(j),
  createCamera: (body) =>
    F("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j),
  updateCamera: (id, body) =>
    F(`/api/cameras/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(j),
  deleteCamera: (id) =>
    F(`/api/cameras/${id}`, { method: "DELETE" }).then(j),

  // Olaylar & uyarılar
  listEvents: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return F(`/api/events?${q}`).then(j);
  },
  listAlerts: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return F(`/api/alerts?${q}`).then(j);
  },
  ackAlert: (id, by = "operatör") =>
    F(`/api/alerts/${id}/ack`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ acknowledged_by: by }),
    }).then(j),

  // Raporlar
  summary: () => F("/api/reports/summary").then(j),
  report: (hours = 24) => F(`/api/reports?hours=${hours}`).then(j),

  // Video analizi
  analyzeVideo: (file, cameraId) => {
    const fd = new FormData();
    fd.append("file", file);
    const url = cameraId ? `/api/analyze?camera_id=${cameraId}` : "/api/analyze";
    return F(url, { method: "POST", body: fd }).then(j);
  },
};

// Canlı akış WebSocket URL'i — backend adresi ayarlıysa onu kullan
export const streamUrl = (cameraId) => {
  const base = backendBase();
  if (base) return base.replace(/^http/, "ws") + `/ws/stream/${cameraId}`;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/stream/${cameraId}`;
};

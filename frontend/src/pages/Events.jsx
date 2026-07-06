import React, { useEffect, useState } from "react";
import { api, assetUrl } from "../api/client";
import { Badge, LABEL_STYLES, Spinner, formatTime } from "../components/ui";

const FILTERS = [
  { key: "", label: "Tümü" },
  { key: "attacker", label: "Saldırgan" },
  { key: "suspicious", label: "Şüpheli" },
  { key: "normal", label: "Normal" },
];

export default function Events() {
  const [events, setEvents] = useState([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [preview, setPreview] = useState(null); // modal için seçili olay

  const load = async () => {
    const params = { limit: 200 };
    if (filter) params.label = filter;
    setEvents(await api.listEvents(params));
    setLoading(false);
  };

  useEffect(() => {
    setLoading(true);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  return (
    <div className="p-8 space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Olay Geçmişi</h1>
        <p className="text-slate-400 text-sm">
          Kaydedilen tüm tespit olayları (veri kaydı &amp; raporlama)
        </p>
      </header>

      <div className="flex gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setFilter(f.key)}
            className={`btn text-sm ${
              filter === f.key ? "bg-accent text-white" : "bg-ink-700 text-slate-300"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {loading ? (
        <Spinner />
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-ink-700/50 text-slate-400 text-xs uppercase">
              <tr>
                <th className="text-left px-4 py-3">Zaman</th>
                <th className="text-left px-4 py-3">Etiket</th>
                <th className="text-left px-4 py-3">Eylem</th>
                <th className="text-left px-4 py-3">Skor</th>
                <th className="text-left px-4 py-3">Kişi</th>
                <th className="text-left px-4 py-3">Görüntü</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {events.map((e) => {
                const ls = LABEL_STYLES[e.label] || LABEL_STYLES.normal;
                return (
                  <tr key={e.id} className="hover:bg-ink-700/30">
                    <td className="px-4 py-3 text-slate-400">{formatTime(e.timestamp)}</td>
                    <td className="px-4 py-3">
                      <Badge className={ls.cls}>{ls.text}</Badge>
                    </td>
                    <td className="px-4 py-3">{e.action || "—"}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-ink-600 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-threat"
                            style={{ width: `${e.threat_score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-400">
                          {Math.round(e.threat_score * 100)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">{e.person_count}</td>
                    <td className="px-4 py-3">
                      {e.snapshot_path ? (
                        <button onClick={() => setPreview(e)} className="focus:outline-none">
                          <img
                            src={assetUrl(e.snapshot_path)}
                            alt=""
                            className="h-9 w-16 object-cover rounded border border-white/10 hover:border-accent/60 transition-colors cursor-pointer"
                          />
                        </button>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {events.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                    Olay bulunamadı.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Snapshot Önizleme Modalı */}
      {preview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setPreview(null)}
        >
          <div
            className="relative bg-ink-800 rounded-2xl border border-white/10 shadow-2xl max-w-4xl w-full mx-4 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Üst bilgi çubuğu */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/5">
              <div className="flex items-center gap-3">
                <Badge className={(LABEL_STYLES[preview.label] || LABEL_STYLES.normal).cls}>
                  {(LABEL_STYLES[preview.label] || LABEL_STYLES.normal).text}
                </Badge>
                <span className="text-sm text-slate-400">{formatTime(preview.timestamp)}</span>
                <span className="text-sm text-slate-500">
                  Tehdit: %{Math.round(preview.threat_score * 100)} · {preview.person_count} kişi
                </span>
              </div>
              <button
                onClick={() => setPreview(null)}
                className="text-slate-400 hover:text-white text-xl transition-colors px-2"
              >
                ✕
              </button>
            </div>
            {/* Snapshot görüntüsü */}
            <div className="flex items-center justify-center bg-black/40 p-2">
              <img
                src={assetUrl(preview.snapshot_path)}
                alt="Olay snapshot'ı"
                className="max-h-[75vh] w-auto rounded"
              />
            </div>
            {/* Alt bilgi */}
            {preview.action && (
              <div className="px-5 py-2 border-t border-white/5 text-xs text-slate-500">
                Eylem: {preview.action}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

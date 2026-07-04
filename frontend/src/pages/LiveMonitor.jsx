import React, { useEffect, useState } from "react";
import { api } from "../api/client";
import { useStream } from "../components/useStream";
import { Badge, LABEL_STYLES, SEVERITY_STYLES, formatTime } from "../components/ui";

const STATUS_TEXT = {
  idle: "Hazır",
  connecting: "Bağlanıyor…",
  open: "Canlı",
  closed: "Bağlantı kapandı",
  error: "Hata / kaynak açılamadı",
};

export default function LiveMonitor() {
  const [cameras, setCameras] = useState([]);
  const [cameraId, setCameraId] = useState(null);
  const { frame, result, status, alerts } = useStream(cameraId, cameraId != null);

  useEffect(() => {
    api.listCameras().then((c) => {
      setCameras(c);
      if (c.length) setCameraId(c[0].id);
    });
  }, []);

  const persons = result?.persons || [];
  const attackerNow = result?.has_attacker;

  return (
    <div className="p-8 space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Canlı İzleme</h1>
          <p className="text-slate-400 text-sm">
            Gerçek zamanlı saldırgan tespiti (R-CNN → OpenPose → LSTM+Attention)
          </p>
        </div>
        <select
          value={cameraId ?? ""}
          onChange={(e) => setCameraId(Number(e.target.value))}
          className="bg-ink-700 border border-white/10 rounded-lg px-4 py-2 text-sm"
        >
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} — {c.location}
            </option>
          ))}
        </select>
      </header>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Video alanı */}
        <div className="lg:col-span-2">
          <div
            className={`card overflow-hidden ${
              attackerNow ? "ring-2 ring-threat animate-pulseRing" : ""
            }`}
          >
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/5">
              <div className="flex items-center gap-2 text-sm">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    status === "open" ? "bg-safe animate-pulse" : "bg-slate-600"
                  }`}
                />
                {STATUS_TEXT[status]}
              </div>
              {attackerNow && (
                <Badge className="bg-threat text-white animate-pulse">
                  ⚠ SALDIRGAN TESPİT EDİLDİ
                </Badge>
              )}
            </div>
            <div className="relative bg-black aspect-video grid place-items-center">
              {frame ? (
                <img
                  src={`data:image/jpeg;base64,${frame}`}
                  alt="canlı akış"
                  className="w-full h-full object-contain"
                />
              ) : (
                <p className="text-slate-600 text-sm">Akış bekleniyor…</p>
              )}
            </div>
          </div>

          {/* Tespit edilen kişiler */}
          <div className="card p-4 mt-4">
            <h3 className="text-sm font-semibold mb-3">
              Tespit Edilen Kişiler ({persons.length})
            </h3>
            <div className="grid sm:grid-cols-2 gap-2">
              {persons.map((p) => {
                const ls = LABEL_STYLES[p.label] || LABEL_STYLES.normal;
                return (
                  <div
                    key={p.track_id}
                    className="flex items-center justify-between bg-ink-700 rounded-lg px-3 py-2"
                  >
                    <span className="text-sm">
                      <span className="text-slate-500">#{p.track_id}</span>{" "}
                      {p.action || "—"}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">
                        {Math.round(p.threat_score * 100)}%
                      </span>
                      <Badge className={ls.cls}>{ls.text}</Badge>
                    </div>
                  </div>
                );
              })}
              {persons.length === 0 && (
                <p className="text-sm text-slate-500">Kişi tespit edilmedi.</p>
              )}
            </div>
          </div>
        </div>

        {/* Canlı uyarı akışı */}
        <div className="card p-5 h-fit">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            🔔 Anlık Uyarı Akışı
          </h3>
          <div className="space-y-3 max-h-[560px] overflow-y-auto pr-1">
            {alerts.length === 0 && (
              <p className="text-sm text-slate-500">
                Bu oturumda henüz uyarı yok. Saldırgan davranış algılandığında
                burada görünecek.
              </p>
            )}
            {alerts.map((a) => (
              <div
                key={a.id + "-" + a._t}
                className="border-l-2 border-threat bg-ink-700/60 rounded-r-lg px-3 py-2"
              >
                <div className="flex items-center justify-between">
                  <Badge className={SEVERITY_STYLES[a.severity]}>{a.severity}</Badge>
                  <span className="text-[11px] text-slate-500">
                    {formatTime(a.created_at)}
                  </span>
                </div>
                <p className="text-sm text-slate-300 mt-1">{a.message}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

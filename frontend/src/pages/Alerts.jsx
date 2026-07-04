import React, { useEffect, useState } from "react";
import { api, assetUrl } from "../api/client";
import { Badge, SEVERITY_STYLES, Spinner, formatTime } from "../components/ui";

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [onlyUnack, setOnlyUnack] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    const data = await api.listAlerts({ only_unack: onlyUnack, limit: 100 });
    setAlerts(data);
    setLoading(false);
  };

  useEffect(() => {
    setLoading(true);
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onlyUnack]);

  const ack = async (id) => {
    await api.ackAlert(id);
    load();
  };

  return (
    <div className="p-8 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Uyarılar</h1>
          <p className="text-slate-400 text-sm">
            Saldırgan tespiti sonrası tetiklenen uyarı sinyalleri
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={onlyUnack}
            onChange={(e) => setOnlyUnack(e.target.checked)}
            className="accent-accent"
          />
          Sadece bekleyenler
        </label>
      </header>

      {loading ? (
        <Spinner />
      ) : (
        <div className="card divide-y divide-white/5">
          {alerts.length === 0 && (
            <p className="p-6 text-sm text-slate-500">Uyarı bulunamadı.</p>
          )}
          {alerts.map((a) => (
            <div key={a.id} className="flex items-center gap-4 p-4">
              <Badge className={SEVERITY_STYLES[a.severity]}>{a.severity}</Badge>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-200">{a.message}</p>
                <p className="text-[11px] text-slate-500">
                  {formatTime(a.created_at)}
                  {a.event && ` • eylem: ${a.event.action || "—"} • ${a.event.person_count} kişi`}
                </p>
              </div>
              {a.event?.snapshot_path && (
                <a href={assetUrl(a.event.snapshot_path)} target="_blank" rel="noreferrer">
                  <img
                    src={assetUrl(a.event.snapshot_path)}
                    alt="snapshot"
                    className="h-12 w-20 object-cover rounded-md border border-white/10"
                  />
                </a>
              )}
              {a.acknowledged ? (
                <Badge className="bg-safe/20 text-safe">
                  ✓ {a.acknowledged_by || "onaylandı"}
                </Badge>
              ) : (
                <button onClick={() => ack(a.id)} className="btn-ghost text-sm">
                  Onayla
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

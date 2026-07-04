import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { api } from "../api/client";
import { Badge, SEVERITY_STYLES, Spinner, StatCard, formatTime } from "../components/ui";

export default function Dashboard() {
  const [report, setReport] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    const [rep, al] = await Promise.all([
      api.report(24),
      api.listAlerts({ limit: 6 }),
    ]);
    setReport(rep);
    setAlerts(al);
    setLoading(false);
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (loading) return <div className="p-8"><Spinner /></div>;
  const s = report.summary;

  return (
    <div className="p-8 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Genel Bakış</h1>
          <p className="text-slate-400 text-sm">
            Kapalı alan güvenlik izleme — anlık durum
          </p>
        </div>
        <Link to="/live" className="btn-primary">◉ Canlı İzlemeye Geç</Link>
      </header>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Toplam Olay" value={s.total_events} icon="≣" />
        <StatCard
          label="Saldırgan Olayı"
          value={s.attacker_events}
          accent="text-threat"
          icon="⚠"
        />
        <StatCard
          label="Bekleyen Uyarı"
          value={s.unacknowledged_alerts}
          accent="text-warn"
          sub={`${s.total_alerts} toplam uyarı`}
          icon="🔔"
        />
        <StatCard
          label="Ort. Tehdit Skoru"
          value={`${Math.round(s.avg_threat_score * 100)}%`}
          accent="text-accent"
          sub={`${s.active_cameras} aktif kamera`}
          icon="📈"
        />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="card p-5 lg:col-span-2">
          <h2 className="font-semibold mb-4">Son 24 Saat — Olay Zaman Çizelgesi</h2>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={report.timeline}>
              <defs>
                <linearGradient id="att" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="norm" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2740" />
              <XAxis dataKey="label" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: "#0f1522",
                  border: "1px solid #1e2740",
                  borderRadius: 12,
                }}
              />
              <Area
                type="monotone"
                dataKey="attacker"
                name="Saldırgan"
                stroke="#ef4444"
                fill="url(#att)"
              />
              <Area
                type="monotone"
                dataKey="suspicious"
                name="Şüpheli"
                stroke="#f59e0b"
                fill="transparent"
              />
              <Area
                type="monotone"
                dataKey="normal"
                name="Normal"
                stroke="#22c55e"
                fill="url(#norm)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold">Son Uyarılar</h2>
            <Link to="/alerts" className="text-xs text-accent hover:underline">
              Tümü →
            </Link>
          </div>
          <div className="space-y-3">
            {alerts.length === 0 && (
              <p className="text-sm text-slate-500">Henüz uyarı yok.</p>
            )}
            {alerts.map((a) => (
              <div key={a.id} className="flex gap-3 items-start">
                <Badge className={SEVERITY_STYLES[a.severity]}>{a.severity}</Badge>
                <div className="min-w-0">
                  <p className="text-sm text-slate-300 truncate">{a.message}</p>
                  <p className="text-[11px] text-slate-500">{formatTime(a.created_at)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Küçük, yeniden kullanılabilir UI parçaları.
import React from "react";

export const LABEL_STYLES = {
  attacker: { text: "Saldırgan", cls: "bg-threat/20 text-threat" },
  suspicious: { text: "Şüpheli", cls: "bg-warn/20 text-warn" },
  normal: { text: "Normal", cls: "bg-safe/20 text-safe" },
};

export const SEVERITY_STYLES = {
  critical: "bg-red-500/20 text-red-400",
  high: "bg-threat/20 text-threat",
  medium: "bg-warn/20 text-warn",
  low: "bg-blue-500/20 text-blue-300",
};

export function Badge({ children, className = "" }) {
  return <span className={`badge ${className}`}>{children}</span>;
}

export function StatCard({ label, value, sub, accent = "text-slate-100", icon }) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-wider text-slate-400">{label}</p>
          <p className={`mt-2 text-3xl font-bold ${accent}`}>{value}</p>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        {icon && <div className="text-2xl opacity-60">{icon}</div>}
      </div>
    </div>
  );
}

export function Spinner({ label = "Yükleniyor..." }) {
  return (
    <div className="flex items-center gap-3 text-slate-400 text-sm py-8 justify-center">
      <span className="h-4 w-4 rounded-full border-2 border-slate-600 border-t-accent animate-spin" />
      {label}
    </div>
  );
}

export function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("tr-TR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

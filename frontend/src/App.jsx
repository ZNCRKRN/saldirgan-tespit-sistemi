import React, { Suspense, lazy, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api, backendBase, setBackendBase } from "./api/client";

// Sayfalar lazy yüklenir: her sayfa ayrı bir parça (chunk) olur,
// kullanıcı sadece açtığı sayfanın kodunu indirir.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const LiveMonitor = lazy(() => import("./pages/LiveMonitor"));
const Alerts = lazy(() => import("./pages/Alerts"));
const Events = lazy(() => import("./pages/Events"));
const Reports = lazy(() => import("./pages/Reports"));
const Cameras = lazy(() => import("./pages/Cameras"));
const VideoAnalysis = lazy(() => import("./pages/VideoAnalysis"));

const NAV = [
  { to: "/dashboard", label: "Genel Bakış", icon: "▦" },
  { to: "/live", label: "Canlı İzleme", icon: "◉" },
  { to: "/alerts", label: "Uyarılar", icon: "⚠" },
  { to: "/events", label: "Olay Geçmişi", icon: "≣" },
  { to: "/reports", label: "Raporlar", icon: "📊" },
  { to: "/analysis", label: "Video Analizi", icon: "⤴" },
  { to: "/cameras", label: "Kameralar", icon: "🎥" },
];

function Sidebar({ model }) {
  return (
    <aside className="w-64 shrink-0 bg-ink-800 border-r border-white/5 flex flex-col">
      <div className="px-6 py-5 border-b border-white/5">
        <div className="flex items-center gap-2">
          <span className="h-9 w-9 rounded-xl bg-accent/20 text-accent grid place-items-center text-lg">
            ⛨
          </span>
          <div>
            <h1 className="font-bold leading-tight">Saldırgan Tespit</h1>
            <p className="text-[11px] text-slate-500">Derin Öğrenme Sistemi</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-slate-400 hover:bg-ink-700 hover:text-slate-200"
              }`
            }
          >
            <span className="w-5 text-center">{n.icon}</span>
            {n.label}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-4 border-t border-white/5 text-[11px] text-slate-500">
        <p className="mb-1 font-semibold text-slate-400">Model Durumu</p>
        {model ? (
          <>
            <p className="truncate">⚙ {model.behavior_classifier}</p>
            <p className={model.using_real_model ? "text-safe" : "text-warn"}>
              {model.using_real_model
                ? "● Eğitilmiş model aktif"
                : "● Yedek (mock) modda"}
            </p>
            {model.using_real_model && (
              <p className="text-slate-500 mt-0.5">
                {model.device === "cuda" ? "⚡ GPU (CUDA)" : "CPU"}
                {model.test_accuracy != null
                  ? ` · test %${(model.test_accuracy * 100).toFixed(1)}`
                  : model.val_accuracy != null
                    ? ` · doğruluk %${(model.val_accuracy * 100).toFixed(1)}`
                    : ""}
              </p>
            )}
          </>
        ) : (
          <p>Bağlanılıyor…</p>
        )}
        <button
          onClick={() => {
            const cur = backendBase();
            const url = window.prompt(
              "Backend sunucu adresi (boş bırak = aynı adres):\n" +
                "Örn: https://xxxx.trycloudflare.com",
              cur
            );
            if (url === null) return; // vazgeçildi
            setBackendBase(url);
            location.reload();
          }}
          className="mt-2 text-slate-600 hover:text-slate-300 transition-colors"
          title="Yayınlanan arayüzü kendi bilgisayarındaki backend'e bağla"
        >
          ⚙ Sunucu: {backendBase() || "aynı adres"}
        </button>
        <p className="mt-1 text-slate-600">TÜBİTAK 2209-A</p>
      </div>
    </aside>
  );
}

export default function App() {
  const [model, setModel] = useState(null);

  useEffect(() => {
    api.modelStatus().then(setModel).catch(() => {});
  }, []);

  return (
    <div className="flex h-full">
      <Sidebar model={model} />
      <main className="flex-1 overflow-y-auto">
        <Suspense
          fallback={
            <div className="p-8 text-sm text-slate-500">Yükleniyor…</div>
          }
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/live" element={<LiveMonitor />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/events" element={<Events />} />
            <Route path="/reports" element={<Reports model={model} />} />
            <Route path="/analysis" element={<VideoAnalysis />} />
            <Route path="/cameras" element={<Cameras />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}

import React, { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  ReferenceLine,
} from "recharts";
import { api } from "../api/client";
import { Badge } from "../components/ui";

const LABEL_COLORS = {
  attacker: "#ef4444",
  suspicious: "#f59e0b",
  normal: "#22c55e",
};

export default function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.analyzeVideo(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      <header>
        <h1 className="text-2xl font-bold">Video Analizi</h1>
        <p className="text-slate-400 text-sm">
          Kayıtlı bir videoyu yükleyin; sistem kare kare saldırgan tespiti yapar
          (test-doğrulama senaryoları için).
        </p>
      </header>

      <div className="card p-8">
        <label className="block border-2 border-dashed border-white/10 rounded-xl p-10 text-center cursor-pointer hover:border-accent/50 transition-colors">
          <input
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <div className="text-4xl mb-2">⤴</div>
          <p className="text-sm text-slate-300">
            {file ? file.name : "Video dosyası seçmek için tıklayın"}
          </p>
          <p className="text-xs text-slate-500 mt-1">MP4, AVI, MOV…</p>
        </label>

        <button
          onClick={run}
          disabled={!file || busy}
          className="btn-primary w-full mt-4"
        >
          {busy ? "Analiz ediliyor… (bu işlem sürebilir)" : "Analizi Başlat"}
        </button>

        {error && (
          <p className="mt-4 text-sm text-threat bg-threat/10 rounded-lg px-4 py-3">
            Hata: {error}
          </p>
        )}

        {result && (
          <div className="mt-6 space-y-4">
            <div
              className={`rounded-xl px-5 py-4 ${
                result.attacker_frames
                  ? "bg-threat/15 border border-threat/30"
                  : "bg-safe/15 border border-safe/30"
              }`}
            >
              <p className="text-lg font-bold">
                {result.attacker_frames ? "⚠ " : "✓ "}
                {result.verdict}
              </p>
              <p className="text-sm text-slate-400">{result.filename}</p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <Metric k="İşlenen Kare" v={result.processed_frames} />
              <Metric k="Saldırgan Kare" v={result.attacker_frames} />
              <Metric k="Maks. Tehdit" v={`${Math.round(result.max_threat_score * 100)}%`} />
              <Metric k="Oluşan Uyarı" v={result.alerts_created} />
            </div>

            {/* Pencere-pencere skor grafiği */}
            {result.windows && result.windows.length > 0 && (
              <div className="card p-5 mt-4">
                <h3 className="font-semibold mb-1">Pencere Bazlı Tehdit Skoru</h3>
                <p className="text-xs text-slate-500 mb-4">
                  Her çubuk ~5 saniyelik bir analiz penceresini temsil eder.
                  Kırmızı çizgi alarm eşiğini (%80) gösterir.
                </p>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={result.windows}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2740" />
                    <XAxis dataKey="window" stroke="#64748b" fontSize={11} label={{ value: "Pencere", position: "insideBottom", offset: -2, fontSize: 10, fill: "#64748b" }} />
                    <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <Tooltip
                      contentStyle={{
                        background: "#0f1522",
                        border: "1px solid #1e2740",
                        borderRadius: 12,
                      }}
                      formatter={(v) => [`${(v * 100).toFixed(1)}%`, "Skor"]}
                      labelFormatter={(l) => `Pencere ${l}`}
                    />
                    <ReferenceLine y={0.8} stroke="#ef4444" strokeDasharray="5 3" strokeWidth={1.5} label={{ value: "Eşik %80", position: "right", fontSize: 10, fill: "#ef4444" }} />
                    <Bar dataKey="score" name="Tehdit Skoru" radius={[4, 4, 0, 0]}>
                      {result.windows.map((w, i) => (
                        <Cell key={i} fill={LABEL_COLORS[w.label] || "#64748b"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ k, v }) {
  return (
    <div className="bg-ink-700 rounded-lg py-3">
      <p className="text-2xl font-bold">{v}</p>
      <p className="text-xs text-slate-500">{k}</p>
    </div>
  );
}


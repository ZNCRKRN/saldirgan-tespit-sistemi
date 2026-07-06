import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { api, backendBase } from "../api/client";
import { Spinner, StatCard } from "../components/ui";

const SEV_COLORS = {
  critical: "#dc2626",
  high: "#ef4444",
  medium: "#f59e0b",
  low: "#3b82f6",
};

export default function Reports({ model }) {
  const [report, setReport] = useState(null);
  const [hours, setHours] = useState(24);

  useEffect(() => {
    api.report(hours).then(setReport);
  }, [hours]);

  if (!report) return <div className="p-8"><Spinner /></div>;
  const s = report.summary;

  return (
    <div className="p-8 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Raporlar</h1>
          <p className="text-slate-400 text-sm">
            Sistem performansı ve tehdit istatistikleri
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={hours}
            onChange={(e) => setHours(Number(e.target.value))}
            className="bg-ink-700 border border-white/10 rounded-lg px-4 py-2 text-sm"
          >
            <option value={6}>Son 6 saat</option>
            <option value={24}>Son 24 saat</option>
            <option value={72}>Son 3 gün</option>
            <option value={168}>Son 7 gün</option>
          </select>
          <a
            href={`${backendBase()}/api/reports/export?hours=${hours}`}
            download
            className="bg-accent/15 text-accent border border-accent/30 rounded-lg px-4 py-2 text-sm hover:bg-accent/25 transition-colors"
            title="Yazdırılabilir HTML raporu indir (tarayıcıdan PDF'e çevrilebilir)"
          >
            ⬇ Raporu İndir
          </a>
        </div>
      </header>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Toplam Olay" value={s.total_events} />
        <StatCard label="Saldırgan Olayı" value={s.attacker_events} accent="text-threat" />
        <StatCard label="Toplam Uyarı" value={s.total_alerts} accent="text-warn" />
        <StatCard
          label="Ort. Tehdit"
          value={`${Math.round(s.avg_threat_score * 100)}%`}
          accent="text-accent"
        />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="card p-5 lg:col-span-2">
          <h2 className="font-semibold mb-4">Olay Dağılımı (Zaman)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={report.timeline}>
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
              <Legend />
              <Bar dataKey="attacker" name="Saldırgan" stackId="a" fill="#ef4444" />
              <Bar dataKey="suspicious" name="Şüpheli" stackId="a" fill="#f59e0b" />
              <Bar dataKey="normal" name="Normal" stackId="a" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-5">
          <h2 className="font-semibold mb-4">Uyarı Önem Dağılımı</h2>
          {report.severity_breakdown.length === 0 ? (
            <p className="text-sm text-slate-500">Henüz uyarı yok.</p>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={report.severity_breakdown}
                  dataKey="count"
                  nameKey="severity"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={3}
                >
                  {report.severity_breakdown.map((e) => (
                    <Cell key={e.severity} fill={SEV_COLORS[e.severity] || "#64748b"} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f1522",
                    border: "1px solid #1e2740",
                    borderRadius: 12,
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {model?.test_accuracy != null && (
        <div className="card p-5">
          <h2 className="font-semibold mb-1">
            Model Performansı — Bağımsız Test Kümesi
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Eğitimde hiç görülmemiş 300 video (150 şiddet / 150 normal)
            üzerinde ölçülmüştür.
          </p>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <StatCard
              label="Test Doğruluğu"
              value={`${(model.test_accuracy * 100).toFixed(1)}%`}
              accent="text-safe"
            />
            {model.test_f1 != null && (
              <StatCard label="F1 Skoru" value={model.test_f1.toFixed(3)} />
            )}
            {model.test_auc != null && (
              <StatCard label="AUC" value={model.test_auc.toFixed(3)} />
            )}
            {model.val_accuracy != null && (
              <StatCard
                label="Doğrulama Doğruluğu"
                value={`${(model.val_accuracy * 100).toFixed(1)}%`}
              />
            )}
          </div>
          {model.class_report && (
            <div className="grid lg:grid-cols-2 gap-6">
              <table className="w-full text-sm h-fit">
                <thead>
                  <tr className="text-left text-[11px] text-slate-500 border-b border-white/5">
                    <th className="py-2">Sınıf</th>
                    <th>Kesinlik (Precision)</th>
                    <th>Duyarlılık (Recall)</th>
                    <th>F1</th>
                    <th>Örnek</th>
                  </tr>
                </thead>
                <tbody>
                  {["NonViolence", "Violence"].map((cls) => {
                    const r = model.class_report[cls];
                    if (!r) return null;
                    return (
                      <tr key={cls} className="border-b border-white/5">
                        <td className="py-2">
                          {cls === "Violence" ? "Şiddet" : "Normal"}
                        </td>
                        <td>{(r.precision * 100).toFixed(1)}%</td>
                        <td>{(r.recall * 100).toFixed(1)}%</td>
                        <td>{(r["f1-score"] * 100).toFixed(1)}%</td>
                        <td>{r.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <ConfusionMatrix report={model.class_report} />
            </div>
          )}
        </div>
      )}

      {model && (
        <div className="card p-5">
          <h2 className="font-semibold mb-3">Model & Pipeline Bilgisi</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
            <Info k="Pipeline" v={model.pipeline} />
            <Info k="Kişi Tespiti (R-CNN)" v={model.person_detector} />
            <Info k="Poz (OpenPose)" v={model.pose_estimator} />
            <Info k="Davranış (LSTM+Attention)" v={model.behavior_classifier} />
            <Info
              k="Eğitilmiş Model"
              v={model.using_real_model ? "Aktif" : "Yedek (mock)"}
            />
            <Info k="Tehdit Eşiği" v={`${Math.round(model.threat_threshold * 100)}%`} />
            <Info
              k="İşlem Birimi"
              v={model.device === "cuda" ? "GPU (CUDA) ⚡" : "CPU"}
            />
            <Info
              k="Kare Penceresi"
              v={model.frame_window ? `${model.frame_window} kare` : "—"}
            />
            {model.val_accuracy != null && (
              <Info
                k="Doğrulama Doğruluğu"
                v={`${(model.val_accuracy * 100).toFixed(1)}%`}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Info({ k, v }) {
  return (
    <div className="bg-ink-700 rounded-lg px-3 py-2">
      <p className="text-[11px] text-slate-500">{k}</p>
      <p className="text-slate-200">{v}</p>
    </div>
  );
}

function ConfusionMatrix({ report }) {
  const nv = report?.NonViolence;
  const vi = report?.Violence;
  if (!nv || !vi) return null;

  // TP/TN/FP/FN hesaplama (recall = TP/(TP+FN), precision = TP/(TP+FP), support = TP+FN)
  const TP = Math.round(vi.recall * vi.support);       // Doğru Şiddet
  const FN = vi.support - TP;                            // Kaçırılan Şiddet
  const TN = Math.round(nv.recall * nv.support);        // Doğru Normal
  const FP = nv.support - TN;                            // Yanlış Alarm
  const total = TP + TN + FP + FN;

  const cells = [
    { val: TN, label: "TN", desc: "Doğru Normal", bg: "bg-safe/25", text: "text-safe" },
    { val: FP, label: "FP", desc: "Yanlış Alarm", bg: "bg-warn/25", text: "text-warn" },
    { val: FN, label: "FN", desc: "Kaçırılan", bg: "bg-orange-500/25", text: "text-orange-400" },
    { val: TP, label: "TP", desc: "Doğru Şiddet", bg: "bg-accent/25", text: "text-accent" },
  ];

  return (
    <div>
      <p className="text-[11px] text-slate-500 mb-2 font-semibold uppercase tracking-wider">
        Karışıklık Matrisi (Confusion Matrix)
      </p>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-0.5 text-center text-sm max-w-xs">
        {/* Üst başlıklar */}
        <div />
        <div className="text-[10px] text-slate-500 py-1">Tahmin: Normal</div>
        <div className="text-[10px] text-slate-500 py-1">Tahmin: Şiddet</div>
        {/* Satır 1: Gerçek Normal */}
        <div className="text-[10px] text-slate-500 flex items-center justify-end pr-2 writing-mode-vertical"
             style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
          Gerçek: Normal
        </div>
        {[cells[0], cells[1]].map((c) => (
          <div key={c.label} className={`${c.bg} rounded-lg py-4 px-2`}>
            <p className={`text-2xl font-bold ${c.text}`}>{c.val}</p>
            <p className="text-[10px] text-slate-400">{c.label} — {c.desc}</p>
          </div>
        ))}
        {/* Satır 2: Gerçek Şiddet */}
        <div className="text-[10px] text-slate-500 flex items-center justify-end pr-2"
             style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
          Gerçek: Şiddet
        </div>
        {[cells[2], cells[3]].map((c) => (
          <div key={c.label} className={`${c.bg} rounded-lg py-4 px-2`}>
            <p className={`text-2xl font-bold ${c.text}`}>{c.val}</p>
            <p className="text-[10px] text-slate-400">{c.label} — {c.desc}</p>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-600 mt-2">
        Toplam: {total} video · Doğruluk: {((TP + TN) / total * 100).toFixed(1)}%
      </p>
    </div>
  );
}

import React, { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { Badge, Spinner } from "../components/ui";
import CameraConnectGuide from "../components/CameraConnectGuide";

const EMPTY = { name: "", location: "", source: "demo", zone: "", is_active: true };

export default function Cameras() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState(EMPTY);
  const [saving, setSaving] = useState(false);
  const sourceRef = useRef(null);

  // Rehberdeki "Forma Uygula" → kaynak alanını doldur + odaklan + vurgula
  const applySource = (src) => {
    setForm((prev) => ({ ...prev, source: src }));
    requestAnimationFrame(() => {
      const el = sourceRef.current;
      if (el) {
        el.focus();
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  };

  const load = async () => {
    setCameras(await api.listCameras());
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    await api.createCamera(form);
    setForm(EMPTY);
    setSaving(false);
    load();
  };

  const toggle = async (c) => {
    await api.updateCamera(c.id, { is_active: !c.is_active });
    load();
  };

  const remove = async (id) => {
    if (!confirm("Kamera silinsin mi?")) return;
    await api.deleteCamera(id);
    load();
  };

  return (
    <div className="p-8 space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Kameralar</h1>
        <p className="text-slate-400 text-sm">
          Kamera kaynakları — webcam indexi, RTSP/HTTP URL veya "demo"
        </p>
      </header>

      <CameraConnectGuide onApply={applySource} />

      <form onSubmit={submit} className="card p-5 grid sm:grid-cols-5 gap-3 items-end">
        <Field label="Ad">
          <input
            required
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="input"
            placeholder="Giriş Holü - Açı 1"
          />
        </Field>
        <Field label="Konum">
          <input
            value={form.location}
            onChange={(e) => setForm({ ...form, location: e.target.value })}
            className="input"
            placeholder="Zemin Kat"
          />
        </Field>
        <Field label="Kaynak">
          <input
            ref={sourceRef}
            value={form.source}
            onChange={(e) => setForm({ ...form, source: e.target.value })}
            className="input"
            placeholder="demo / 0 / rtsp://…"
          />
        </Field>
        <Field label="Bölge (çok açılı füzyon)">
          <input
            value={form.zone}
            onChange={(e) => setForm({ ...form, zone: e.target.value })}
            className="input"
            placeholder="ör: giris-holu (opsiyonel)"
            title="Aynı alanı gören kameralara aynı bölge adını verin: bölgedeki en yüksek tehdit skoru tüm açılara yansıtılır"
          />
        </Field>
        <button disabled={saving} className="btn-primary">
          {saving ? "Ekleniyor…" : "+ Kamera Ekle"}
        </button>
      </form>

      {loading ? (
        <Spinner />
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cameras.map((c) => (
            <div key={c.id} className="card p-5">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{c.name}</h3>
                  <p className="text-xs text-slate-500">
                    {c.location || "—"}
                    {c.zone ? ` · bölge: ${c.zone}` : ""}
                  </p>
                </div>
                <Badge
                  className={c.is_active ? "bg-safe/20 text-safe" : "bg-slate-600/30 text-slate-400"}
                >
                  {c.is_active ? "Aktif" : "Pasif"}
                </Badge>
              </div>
              <p className="mt-3 text-xs text-slate-400 font-mono bg-ink-700 rounded px-2 py-1 truncate">
                {c.source}
              </p>
              <div className="mt-4 flex gap-2">
                <button onClick={() => toggle(c)} className="btn-ghost text-sm flex-1">
                  {c.is_active ? "Devre dışı" : "Etkinleştir"}
                </button>
                <button
                  onClick={() => remove(c.id)}
                  className="btn text-sm bg-threat/20 text-threat hover:bg-threat/30"
                >
                  Sil
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <style>{`.input{width:100%;background:#161d2e;border:1px solid rgba(255,255,255,.1);border-radius:.5rem;padding:.5rem .75rem;font-size:.875rem;color:#e2e8f0}.input:focus{outline:none;border-color:#3b82f6}`}</style>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="text-xs text-slate-400 mb-1 block">{label}</span>
      {children}
    </label>
  );
}

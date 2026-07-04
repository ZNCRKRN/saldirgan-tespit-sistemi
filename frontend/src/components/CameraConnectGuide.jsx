import React, { useState } from "react";

/**
 * Gerçek kamera bağlama rehberi + RTSP adres oluşturucu.
 * "Sitenin güvenlik ekibi kendi kameralarını bağlıyor" senaryosuna göre,
 * teknik olmayan bir kullanıcının da anlayacağı şekilde hazırlanmıştır.
 *
 * onApply(source): oluşturulan kaynak metnini üstteki "Kamera Ekle" formuna yazar.
 */

// Yaygın markaların RTSP adres şablonları. {ip}{user}{pass}{port} doldurulur.
// main = ana akış (yüksek çözünürlük), sub = alt akış (düşük, akıcı/performanslı).
const BRANDS = {
  hikvision: {
    label: "Hikvision",
    main: "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/101",
    sub: "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/102",
    port: 554,
  },
  dahua: {
    label: "Dahua / Imou",
    main: "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
    sub: "rtsp://{user}:{pass}@{ip}:{port}/cam/realmonitor?channel=1&subtype=1",
    port: 554,
  },
  reolink: {
    label: "Reolink",
    main: "rtsp://{user}:{pass}@{ip}:{port}/h264Preview_01_main",
    sub: "rtsp://{user}:{pass}@{ip}:{port}/h264Preview_01_sub",
    port: 554,
  },
  tplink: {
    label: "TP-Link Tapo / VIGI",
    main: "rtsp://{user}:{pass}@{ip}:{port}/stream1",
    sub: "rtsp://{user}:{pass}@{ip}:{port}/stream2",
    port: 554,
  },
  axis: {
    label: "Axis",
    main: "rtsp://{user}:{pass}@{ip}/axis-media/media.amp",
    sub: "rtsp://{user}:{pass}@{ip}/axis-media/media.amp?resolution=640x480",
    port: 554,
  },
  onvif: {
    label: "Genel / ONVIF",
    main: "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/101",
    sub: "rtsp://{user}:{pass}@{ip}:{port}/Streaming/Channels/102",
    port: 554,
  },
};

const SOURCE_TYPES = [
  {
    icon: "🎥",
    title: "IP / Güvenlik kamerası (en yaygın)",
    desc: "Ağ üzerinden RTSP adresiyle bağlanır. Aşağıdaki oluşturucuyu kullanın.",
    example: "rtsp://kullanici:sifre@192.168.1.64:554/Streaming/Channels/101",
  },
  {
    icon: "🌐",
    title: "HTTP / MJPEG akışı",
    desc: "Bazı kameralar tarayıcıdan görülebilen bir video bağlantısı sunar.",
    example: "http://192.168.1.50:8080/video",
  },
  {
    icon: "🔌",
    title: "USB / dizüstü kamerası",
    desc: "Sunucu bilgisayara takılı kamera. 0 = ilk kamera, 1 = ikinci…",
    example: "0",
  },
  {
    icon: "🧪",
    title: "Demo (kamerasız test)",
    desc: "Donanım olmadan sistemi denemek için sentetik sahne.",
    example: "demo",
  },
];

function fill(tpl, { ip, user, pass, port }) {
  return tpl
    .replaceAll("{ip}", ip || "192.168.1.64")
    .replaceAll("{user}", user || "admin")
    .replaceAll("{pass}", pass || "sifre")
    .replaceAll("{port}", port || 554);
}

export default function CameraConnectGuide({ onApply }) {
  const [open, setOpen] = useState(true);
  const [brand, setBrand] = useState("hikvision");
  const [stream, setStream] = useState("sub");
  const [f, setF] = useState({ ip: "", user: "admin", pass: "", port: 554 });

  const b = BRANDS[brand];
  const url = fill(b[stream], { ...f, port: f.port || b.port });

  const copy = () => navigator.clipboard?.writeText(url).catch(() => {});

  return (
    <div className="card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-white/5 transition-colors"
      >
        <span className="flex items-center gap-2 font-semibold">
          <span className="text-accent">📡</span> Gerçek kameranızı nasıl bağlarsınız?
          <span className="text-xs font-normal text-slate-500">
            (RTSP / IP kamera kurulum rehberi)
          </span>
        </span>
        <span className="text-slate-400 text-sm">{open ? "▲ gizle" : "▼ göster"}</span>
      </button>

      {open && (
        <div className="px-5 pb-6 pt-1 space-y-6">
          {/* 1. Kaynak türleri */}
          <div>
            <p className="text-sm text-slate-300 mb-3">
              Aşağıdaki <b>"Kaynak"</b> alanına kameranızın bağlantı adresini yazın.
              Kamera türünüze göre dört yol vardır:
            </p>
            <div className="grid sm:grid-cols-2 gap-3">
              {SOURCE_TYPES.map((s) => (
                <div key={s.title} className="bg-ink-700 rounded-lg p-3">
                  <p className="font-medium text-sm flex items-center gap-2">
                    <span>{s.icon}</span> {s.title}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">{s.desc}</p>
                  <code
                    onClick={() => onApply?.(s.example)}
                    title="Forma uygulamak için tıklayın"
                    className="mt-2 block text-[11px] font-mono text-accent/90 bg-black/30 rounded px-2 py-1 cursor-pointer hover:bg-black/50 truncate"
                  >
                    {s.example}
                  </code>
                </div>
              ))}
            </div>
          </div>

          {/* 2. RTSP adres oluşturucu */}
          <div className="bg-ink-700/60 rounded-xl p-4 border border-accent/15">
            <p className="font-semibold text-sm mb-1">🔧 RTSP adres oluşturucu</p>
            <p className="text-xs text-slate-400 mb-4">
              Kameranızın markasını seçin, IP ve giriş bilgilerini girin — adres
              otomatik oluşur. <b>Forma Uygula</b> ile yukarıdaki kayıt formuna aktarın.
            </p>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
              <L label="Marka">
                <select
                  value={brand}
                  onChange={(e) => setBrand(e.target.value)}
                  className="input"
                >
                  {Object.entries(BRANDS).map(([k, v]) => (
                    <option key={k} value={k}>{v.label}</option>
                  ))}
                </select>
              </L>
              <L label="Kamera IP adresi">
                <input
                  className="input"
                  placeholder="192.168.1.64"
                  value={f.ip}
                  onChange={(e) => setF({ ...f, ip: e.target.value })}
                />
              </L>
              <L label="Kullanıcı adı">
                <input
                  className="input"
                  placeholder="admin"
                  value={f.user}
                  onChange={(e) => setF({ ...f, user: e.target.value })}
                />
              </L>
              <L label="Şifre">
                <input
                  className="input"
                  type="text"
                  placeholder="kamera şifresi"
                  value={f.pass}
                  onChange={(e) => setF({ ...f, pass: e.target.value })}
                />
              </L>
            </div>

            <div className="flex items-center gap-4 mt-3">
              <span className="text-xs text-slate-400">Akış kalitesi:</span>
              <label className="text-xs flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  checked={stream === "sub"}
                  onChange={() => setStream("sub")}
                />
                Alt akış <span className="text-slate-500">(önerilir — daha akıcı)</span>
              </label>
              <label className="text-xs flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  checked={stream === "main"}
                  onChange={() => setStream("main")}
                />
                Ana akış <span className="text-slate-500">(yüksek çözünürlük)</span>
              </label>
            </div>

            <div className="mt-4">
              <span className="text-xs text-slate-400">Oluşan adres:</span>
              <div className="mt-1 flex gap-2">
                <code className="flex-1 text-[12px] font-mono text-safe bg-black/40 rounded px-3 py-2 truncate">
                  {url}
                </code>
                <button onClick={copy} className="btn-ghost text-xs px-3">Kopyala</button>
                <button onClick={() => onApply?.(url)} className="btn-primary text-xs px-3">
                  Forma Uygula
                </button>
              </div>
            </div>
          </div>

          {/* 3. Kontrol listesi */}
          <div className="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-xs text-slate-300">
            <p className="sm:col-span-2 font-semibold text-slate-200">
              ✅ Bağlanmadan önce kontrol listesi
            </p>
            <Check>
              Kamera ile bu sistemin çalıştığı sunucu <b>aynı ağda</b> olmalı (veya
              kameraya erişilebilen bir ağ/VPN üzerinden).
            </Check>
            <Check>
              Kameranın <b>IP adresini</b> ve <b>kullanıcı/şifresini</b> kamera
              yöneticinizden veya kayıt cihazı (NVR) arayüzünden öğrenin.
            </Check>
            <Check>
              RTSP genelde <b>554</b> portunu kullanır; kapalıysa kamera ayarlarından
              RTSP'yi etkinleştirin.
            </Check>
            <Check>
              Performans için <b>alt akışı (sub-stream)</b> tercih edin — analiz için
              fazlasıyla yeterli, sunucuyu yormaz.
            </Check>
            <Check>
              Birden çok açı için her kamerayı <b>ayrı kayıt</b> olarak ekleyin
              (örn. "Giriş Holü - Açı 1", "Açı 2").
            </Check>
            <Check>
              Adresi test etmek için VLC → "Ağ akışı aç" ile aynı RTSP linkini
              deneyebilirsiniz; VLC'de açılıyorsa burada da çalışır.
            </Check>
          </div>

          <p className="text-[11px] text-slate-500 border-t border-white/5 pt-3">
            Not: İnternet üzerinden uzaktan erişim için yönlendiricide port
            yönlendirme/DDNS gerekebilir. Güvenlik için kameraları mümkünse yerel
            ağda tutun.
          </p>
        </div>
      )}
    </div>
  );
}

function L({ label, children }) {
  return (
    <label className="block">
      <span className="text-xs text-slate-400 mb-1 block">{label}</span>
      {children}
    </label>
  );
}

function Check({ children }) {
  return (
    <p className="flex gap-2">
      <span className="text-safe shrink-0">✓</span>
      <span>{children}</span>
    </p>
  );
}

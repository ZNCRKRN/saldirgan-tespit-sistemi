"""Raporlama endpoint'leri (form bölüm 2.6 "Veri Kaydı ve Raporlama")."""
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from .. import crud, models, schemas
from ..config import settings
from ..database import get_db
from ..ml.pipeline import get_pipeline

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/summary", response_model=schemas.StatsSummary)
def summary(db: Session = Depends(get_db)):
    return crud.summary_stats(db)


@router.get("", response_model=schemas.ReportResponse)
def full_report(
    hours: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)
):
    return {
        "summary": crud.summary_stats(db),
        "timeline": crud.timeline(db, hours=hours),
        "severity_breakdown": crud.severity_breakdown(db),
    }


@router.get("/model-status", response_model=schemas.ModelStatus)
def model_status():
    return get_pipeline().status()


# ── Rapor dışa aktarma (form 2.6.3: "PDF, HTML vb. rapor formatları") ──
_SEV_TR = {"critical": "Kritik", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}
_LBL_TR = {"attacker": "Saldırgan", "suspicious": "Şüpheli", "normal": "Normal"}


@router.get("/export")
def export_report(
    hours: int = Query(24, ge=1, le=168), db: Session = Depends(get_db)
):
    """Kendi kendine yeten HTML raporu üretir (indirilebilir).

    Tarayıcıdan "Yazdır → PDF olarak kaydet" ile PDF'e çevrilebilir;
    stil yazdırma için optimize edilmiştir.
    """
    s = crud.summary_stats(db)
    tl = crud.timeline(db, hours=hours)
    sev = crud.severity_breakdown(db)
    status = get_pipeline().status()
    events = (
        db.query(models.Event)
        .order_by(models.Event.timestamp.desc())
        .limit(50)
        .all()
    )
    now = datetime.now()

    def esc(x) -> str:
        return (str(x).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    tl_rows = "".join(
        f"<tr><td>{esc(b['label'])}</td><td>{b['attacker']}</td>"
        f"<td>{b['suspicious']}</td><td>{b['normal']}</td></tr>"
        for b in tl
    )
    sev_rows = "".join(
        f"<tr><td>{_SEV_TR.get(r['severity'], r['severity'])}</td>"
        f"<td>{r['count']}</td></tr>"
        for r in sev
    ) or "<tr><td colspan=2>Uyarı yok</td></tr>"
    ev_rows = "".join(
        f"<tr><td>{e.timestamp.strftime('%d.%m.%Y %H:%M:%S')}</td>"
        f"<td>{e.camera_id or '—'}</td>"
        f"<td>{_LBL_TR.get(e.label, e.label)}</td>"
        f"<td>%{round(e.threat_score * 100)}</td>"
        f"<td>{esc(e.action or '—')}</td><td>{e.person_count}</td></tr>"
        for e in events
    ) or "<tr><td colspan=6>Kayıtlı olay yok</td></tr>"

    perf = ""
    if status.get("test_accuracy") is not None:
        cr = status.get("class_report") or {}
        cr_rows = "".join(
            f"<tr><td>{'Şiddet' if c == 'Violence' else 'Normal'}</td>"
            f"<td>%{cr[c]['precision'] * 100:.1f}</td>"
            f"<td>%{cr[c]['recall'] * 100:.1f}</td>"
            f"<td>%{cr[c]['f1-score'] * 100:.1f}</td>"
            f"<td>{int(cr[c]['support'])}</td></tr>"
            for c in ("NonViolence", "Violence") if c in cr
        )
        perf = f"""
  <h2>Model Performansı — Bağımsız Test Kümesi</h2>
  <div class="cards">
    <div class="card"><b>%{status['test_accuracy'] * 100:.1f}</b><span>Test Doğruluğu</span></div>
    <div class="card"><b>{status.get('test_f1', 0):.3f}</b><span>F1 Skoru</span></div>
    <div class="card"><b>{status.get('test_auc', 0):.3f}</b><span>AUC</span></div>
  </div>
  <table><tr><th>Sınıf</th><th>Kesinlik</th><th>Duyarlılık</th><th>F1</th><th>Örnek</th></tr>{cr_rows}</table>"""

        # Karışıklık matrisi (TP/TN/FP/FN) hesapla ve HTML'e ekle
        vi = cr.get("Violence", {})
        nv = cr.get("NonViolence", {})
        if vi and nv:
            tp = round(vi.get("recall", 0) * vi.get("support", 0))
            fn = int(vi.get("support", 0)) - tp
            tn = round(nv.get("recall", 0) * nv.get("support", 0))
            fp = int(nv.get("support", 0)) - tn
            total_cm = tp + tn + fp + fn
            perf += f"""
  <h3 style="margin-top:20px">Karışıklık Matrisi (Confusion Matrix)</h3>
  <table class="cm" style="max-width:400px;text-align:center">
    <tr><th></th><th>Tahmin: Normal</th><th>Tahmin: Şiddet</th></tr>
    <tr><th style="text-align:right;padding-right:8px">Gerçek: Normal</th>
        <td style="background:#dcfce7;color:#166534"><b>{tn}</b><br><small>TN — Doğru Normal</small></td>
        <td style="background:#fef9c3;color:#854d0e"><b>{fp}</b><br><small>FP — Yanlış Alarm</small></td></tr>
    <tr><th style="text-align:right;padding-right:8px">Gerçek: Şiddet</th>
        <td style="background:#ffedd5;color:#9a3412"><b>{fn}</b><br><small>FN — Kaçırılan</small></td>
        <td style="background:#dbeafe;color:#1e40af"><b>{tp}</b><br><small>TP — Doğru Şiddet</small></td></tr>
  </table>
  <p style="font-size:11px;color:#94a3b8">Toplam: {total_cm} video · Doğruluk: %{(tp + tn) / max(total_cm, 1) * 100:.1f}</p>"""

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Saldırgan Tespit Raporu — {now.strftime('%d.%m.%Y %H:%M')}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px auto; max-width: 900px;
         color: #1a2433; }}
  h1 {{ border-bottom: 3px solid #3b82f6; padding-bottom: 8px; }}
  h2 {{ margin-top: 28px; color: #1e3a8a; }}
  .meta {{ color: #64748b; font-size: 13px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
  .card {{ border: 1px solid #dbe3ef; border-radius: 10px; padding: 12px 20px;
          min-width: 130px; text-align: center; }}
  .card b {{ display: block; font-size: 22px; }}
  .card span {{ font-size: 12px; color: #64748b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #dbe3ef; padding: 6px 10px; text-align: left; }}
  th {{ background: #eef2ff; }}
  footer {{ margin-top: 32px; color: #94a3b8; font-size: 12px;
           border-top: 1px solid #dbe3ef; padding-top: 8px; }}
  @media print {{ body {{ margin: 10mm; }} .card {{ break-inside: avoid; }} }}
</style></head><body>
<h1>Saldırgan Tespit Sistemi — Otomatik Rapor</h1>
<p class="meta">Oluşturulma: {now.strftime('%d.%m.%Y %H:%M:%S')} ·
Kapsam: son {hours} saat · {esc(settings.app_name)} v{esc(settings.app_version)}</p>

<h2>Özet</h2>
<div class="cards">
  <div class="card"><b>{s['total_events']}</b><span>Toplam Olay</span></div>
  <div class="card"><b>{s['attacker_events']}</b><span>Saldırgan Olayı</span></div>
  <div class="card"><b>{s['total_alerts']}</b><span>Toplam Uyarı</span></div>
  <div class="card"><b>%{round(s['avg_threat_score'] * 100)}</b><span>Ortalama Tehdit</span></div>
</div>
{perf}
<h2>Olay Dağılımı (Zaman)</h2>
<table><tr><th>Zaman</th><th>Saldırgan</th><th>Şüpheli</th><th>Normal</th></tr>{tl_rows}</table>

<h2>Uyarı Önem Dağılımı</h2>
<table><tr><th>Önem</th><th>Adet</th></tr>{sev_rows}</table>

<h2>Son Olaylar (en yeni 50)</h2>
<table><tr><th>Zaman</th><th>Kamera</th><th>Etiket</th><th>Tehdit</th>
<th>Eylem</th><th>Kişi</th></tr>{ev_rows}</table>

<h2>Model &amp; Pipeline</h2>
<table>
<tr><th>Pipeline</th><td>{esc(status['pipeline'])}</td></tr>
<tr><th>Kişi Tespiti</th><td>{esc(status['person_detector'])}</td></tr>
<tr><th>Davranış Modeli</th><td>{esc(status['behavior_classifier'])}</td></tr>
<tr><th>İşlem Birimi</th><td>{'GPU (CUDA)' if status['device'] == 'cuda' else 'CPU'}</td></tr>
<tr><th>Tehdit Eşiği</th><td>%{round(status['threat_threshold'] * 100)}</td></tr>
</table>
<footer>TÜBİTAK 2209-A — Kapalı Alanlarda Derin Öğrenme Tabanlı Saldırgan Tespiti.
Bu rapor sistem tarafından otomatik üretilmiştir. PDF için: Yazdır → PDF olarak kaydet.</footer>
</body></html>"""

    fname = f"saldirgan-tespit-raporu-{now.strftime('%Y%m%d-%H%M')}.html"
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )

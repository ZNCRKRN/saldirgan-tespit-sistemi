import { useEffect, useRef, useState } from "react";
import { streamUrl } from "../api/client";

// Bir kameraya WebSocket ile bağlanır; her kareyi (jpeg + tespit sonucu)
// ve gelen uyarıları döndürür.
export function useStream(cameraId, enabled = true) {
  const [frame, setFrame] = useState(null); // base64 jpeg
  const [result, setResult] = useState(null); // tespit sonucu
  const [status, setStatus] = useState("idle"); // idle|connecting|open|closed|error
  const [alerts, setAlerts] = useState([]); // bu oturumda gelen uyarılar
  const wsRef = useRef(null);

  useEffect(() => {
    if (!enabled || cameraId == null) return;
    setStatus("connecting");
    setAlerts([]);
    const ws = new WebSocket(streamUrl(cameraId));
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.error) {
        setStatus("error");
        return;
      }
      if (msg.image) setFrame(msg.image);
      if (msg.result) setResult(msg.result);
      if (msg.alert) {
        setAlerts((prev) => [{ ...msg.alert, _t: Date.now() }, ...prev].slice(0, 30));
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [cameraId, enabled]);

  return { frame, result, status, alerts };
}

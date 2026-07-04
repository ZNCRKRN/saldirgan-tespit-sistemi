import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Paketi mantıklı parçalara böl: değişmeyen kütüphaneler ayrı
        // dosyada kalır -> tarayıcı önbelleği verimli, ilk açılış hızlı.
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      // REST ve statik snapshot istekleri backend'e yönlendirilir
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/snapshots": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});

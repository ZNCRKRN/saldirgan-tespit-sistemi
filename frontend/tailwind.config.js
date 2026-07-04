/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0a0e17",
          800: "#0f1522",
          700: "#161d2e",
          600: "#1e2740",
        },
        accent: "#3b82f6",
        threat: "#ef4444",
        warn: "#f59e0b",
        safe: "#22c55e",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
      },
      keyframes: {
        pulseRing: {
          "0%": { boxShadow: "0 0 0 0 rgba(239,68,68,0.5)" },
          "70%": { boxShadow: "0 0 0 12px rgba(239,68,68,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(239,68,68,0)" },
        },
      },
      animation: {
        pulseRing: "pulseRing 1.6s infinite",
      },
    },
  },
  plugins: [],
};

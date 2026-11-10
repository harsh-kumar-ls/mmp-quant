import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

// The React UI (10 Nov 2026) is a client of the same FastAPI server as the
// original UI: `mmpquant serve` must be running on :8000. Every number shown
// comes from that API — this app adds no scoring behaviour.
export default defineConfig(({ command }) => ({
  // built files are served by `mmpquant serve` at /react/ (no Node needed to view)
  base: command === "build" ? "/react/" : "/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
}));

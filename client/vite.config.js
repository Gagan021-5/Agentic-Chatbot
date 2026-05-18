import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:3001",
        changeOrigin: true,
        proxyTimeout: 120000,  // 120s — LLM calls can be slow
        timeout: 120000
      }
    }
  }
});

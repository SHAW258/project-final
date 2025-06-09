import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import path from "path"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    host: true, // Allow external connections
    open: true,
    cors: true,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          ui: ["@radix-ui/react-alert-dialog", "@radix-ui/react-label", "@radix-ui/react-slot", "@radix-ui/react-tabs"],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["react", "react-dom", "lucide-react"],
  },
})
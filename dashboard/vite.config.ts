/// <reference types="vitest/config" />
import { readFileSync } from 'node:fs'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { parse } from 'yaml'

const config = parse(readFileSync(new URL('../configs/default.yaml', import.meta.url), 'utf8'))
const server = `${config.server.host}:${process.env.CT_SERVER_PORT ?? config.server.port}`

export default defineConfig({
  plugins: [react()],
  define: {
    __DEFAULT_SPEED__: JSON.stringify(config.server.default_speed),
    __CALIBRATION_SECONDS__: JSON.stringify(config.profile.calibration_seconds),
    __VT_CONSECUTIVE_V__: JSON.stringify(config.decision.vt_consecutive_v),
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': `http://${server}`,
      '/ws': { target: `ws://${server}`, ws: true },
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
})

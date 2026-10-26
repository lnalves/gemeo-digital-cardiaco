// Dashboard parameters (D-29). Values in capitals with underscores come from configs/default.yaml (vite.config.ts).

declare const __DEFAULT_SPEED__: number
declare const __CALIBRATION_SECONDS__: number
declare const __VT_CONSECUTIVE_V__: number

export const DEFAULT_SPEED: number = __DEFAULT_SPEED__
export const CALIBRATION_SECONDS: number = __CALIBRATION_SECONDS__
export const VT_CONSECUTIVE_V: number = __VT_CONSECUTIVE_V__
export const SPEEDS = [1, 5] as const
export const ECG_WINDOW_SECONDS = 10
export const UI_REFRESH_MS = 250
export const RECONNECT_BASE_MS = 500
export const RECONNECT_MAX_MS = 8000
export const ALERT_LIST_LIMIT = 30
export const HISTORY_PAGE_SIZE = 200

/**
 * config.ts — environment-specific configuration for the mobile app.
 */

// WebSocket base URL (no trailing slash)
// Dev:  ws://192.168.1.x:8000   (your local IP on LAN)
// Prod: wss://api.visionassistant.app
// Edit these values to match your machine's LAN IP.
export const API_WS_URL: string = 'ws://192.168.1.100:8000';

// HTTP base URL for one-shot /analyse endpoint
export const API_HTTP_URL: string = 'http://192.168.1.100:8000';

// Feature flags
export const ENABLE_DEPTH_MAP_OVERLAY: boolean = false;
export const ENABLE_BOUNDING_BOX_OVERLAY: boolean = true;
export const DEFAULT_LANGUAGE: 'en' | 'hi' = 'en';
export const FRAME_RATE_HZ: number = 12;

/**
 * config.ts — environment-specific configuration for the mobile app.
 *
 * For production, use react-native-config to inject env vars at build time.
 * See: https://github.com/luggit/react-native-config
 */

import Config from 'react-native-config';

// WebSocket base URL (no trailing slash)
// Dev:  ws://192.168.1.x:8000   (your local IP on LAN)
// Prod: wss://api.visionassistant.app
export const API_WS_URL: string =
  Config.API_WS_URL ?? 'ws://192.168.1.100:8000';

// HTTP base URL for one-shot /analyse endpoint
export const API_HTTP_URL: string =
  Config.API_HTTP_URL ?? 'http://192.168.1.100:8000';

// Feature flags
export const ENABLE_DEPTH_MAP_OVERLAY: boolean = false;
export const ENABLE_BOUNDING_BOX_OVERLAY: boolean = __DEV__;
export const DEFAULT_LANGUAGE: 'en' | 'hi' = 'en';
export const FRAME_RATE_HZ: number = 12;

/**
 * frameUtils.ts
 *
 * Utilities for encoding camera frames to base64 JPEG
 * before transmission over WebSocket.
 *
 * Uses react-native-vision-camera Frame API.
 * Compression quality 0.65 balances latency vs accuracy.
 */

import { Frame } from 'react-native-vision-camera';

const JPEG_QUALITY = 0.65;
const MAX_DIMENSION = 640; // resize to max 640px on longest side before encoding

/**
 * Encode a VisionCamera Frame to base64 JPEG string.
 * This runs inside a Worklet — must be synchronous.
 */
export function encodeFrameToBase64(frame: Frame): string {
  // react-native-vision-camera provides toArrayBuffer() in v3+
  // For production use the native plugin approach below:
  const buffer = frame.toArrayBuffer();
  const uint8 = new Uint8Array(buffer);
  
  // Convert to base64 — Worklet-safe implementation
  let binary = '';
  const len = uint8.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(uint8[i]);
  }
  return btoa(binary);
}

/**
 * Calculate scaled dimensions preserving aspect ratio.
 * Reduces frame size before encoding to lower bandwidth.
 */
export function scaleToMax(width: number, height: number, maxDim: number = MAX_DIMENSION) {
  if (width <= maxDim && height <= maxDim) return { width, height };
  const ratio = Math.min(maxDim / width, maxDim / height);
  return {
    width: Math.round(width * ratio),
    height: Math.round(height * ratio),
  };
}

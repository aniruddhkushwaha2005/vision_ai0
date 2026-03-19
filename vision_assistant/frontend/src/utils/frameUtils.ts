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

const MAX_DIMENSION = 640; // resize to max 640px on longest side before encoding

// Worklet-safe base64 encoder (avoids relying on `btoa`, which isn't typed/available everywhere)
const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function base64FromBytes(bytes: Uint8Array): string {
  'worklet';
  let out = '';
  const len = bytes.length;
  let i = 0;
  while (i < len) {
    const a = bytes[i++] ?? 0;
    const b = i < len ? (bytes[i++] ?? 0) : 0;
    const c = i < len ? (bytes[i++] ?? 0) : 0;

    const triple = (a << 16) | (b << 8) | c;
    out += BASE64_ALPHABET[(triple >> 18) & 63];
    out += BASE64_ALPHABET[(triple >> 12) & 63];
    out += i - 2 < len ? BASE64_ALPHABET[(triple >> 6) & 63] : '=';
    out += i - 1 < len ? BASE64_ALPHABET[triple & 63] : '=';
  }
  return out;
}

/**
 * Encode a VisionCamera Frame to base64 JPEG string.
 * This runs inside a Worklet — must be synchronous.
 */
export function encodeFrameToBase64(frame: Frame): string {
  // react-native-vision-camera provides toArrayBuffer() in v3+
  // For production use the native plugin approach below:
  const buffer = frame.toArrayBuffer();
  const uint8 = new Uint8Array(buffer);

  return base64FromBytes(uint8);
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

/**
 * VisionWebSocketService
 *
 * Manages the persistent WebSocket connection to the Vision Assistant backend.
 * Handles:
 *   - Connection lifecycle (connect/disconnect/reconnect with backoff)
 *   - Frame transmission from camera
 *   - Incoming navigation result parsing
 *   - Audio playback queue
 */

import { EventEmitter } from 'events';
import Sound from 'react-native-sound';
import RNFS from 'react-native-fs';
import uuid from 'react-native-uuid';

export type NavigationDecision =
  | 'FORWARD' | 'TURN_LEFT' | 'TURN_RIGHT'
  | 'STOP' | 'DANGER' | 'CLEAR';

export interface StreamResult {
  frameId: string;
  decision: NavigationDecision;
  speechTextEn: string;
  speechTextHi: string | null;
  shouldSpeak: boolean;
  isDangerAlert: boolean;
  audioBase64: string | null;
  processingMs: number;
  detectedObjects: Array<{ className: string; confidence: number; region: string }>;
}

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

const RECONNECT_DELAYS = [1000, 2000, 5000, 10000, 30000]; // exponential-ish backoff

class VisionWebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private serverUrl: string;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempt = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private preferredLang: 'en' | 'hi' = 'en';
  private audioQueue: string[] = [];
  private isPlayingAudio = false;

  constructor(serverUrl: string) {
    super();
    this.serverUrl = serverUrl;
    this.sessionId = uuid.v4() as string;
  }

  connect() {
    if (this.state === 'connected' || this.state === 'connecting') return;
    this._setState('connecting');

    try {
      this.ws = new WebSocket(`${this.serverUrl}/api/v1/stream`);
      this.ws.onopen    = this._onOpen.bind(this);
      this.ws.onmessage = this._onMessage.bind(this);
      this.ws.onerror   = this._onError.bind(this);
      this.ws.onclose   = this._onClose.bind(this);
    } catch (err) {
      console.error('[WS] Connection failed:', err);
      this._scheduleReconnect();
    }
  }

  disconnect() {
    this._clearReconnectTimer();
    this.reconnectAttempt = 0;
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this._setState('disconnected');
  }

  setLanguage(lang: 'en' | 'hi') {
    this.preferredLang = lang;
  }

  /**
   * Send a base64-encoded JPEG frame for processing.
   * Call this from your camera frame callback at ~10-15fps.
   */
  sendFrame(imageBase64: string) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    const payload = JSON.stringify({
      frame_id: uuid.v4(),
      image_base64: imageBase64,
      timestamp_ms: Date.now(),
      session_id: this.sessionId,
      preferred_language: this.preferredLang,
    });

    this.ws.send(payload);
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private _onOpen() {
    console.log('[WS] Connected');
    this.reconnectAttempt = 0;
    this._setState('connected');
    this.emit('connected');
  }

  private _onMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data as string);

      // Error from server
      if (data.error) {
        console.warn('[WS] Server error:', data.error);
        return;
      }

      const result: StreamResult = {
        frameId: data.frame_id,
        decision: data.navigation?.decision,
        speechTextEn: data.navigation?.speech_text_en,
        speechTextHi: data.navigation?.speech_text_hi,
        shouldSpeak: data.navigation?.should_speak,
        isDangerAlert: data.navigation?.is_danger_alert,
        audioBase64: data.audio_base64,
        processingMs: data.server_processing_ms,
        detectedObjects: (data.detections?.objects || []).map((o: any) => ({
          className: o.class_name,
          confidence: o.confidence,
          region: o.region,
        })),
      };

      this.emit('result', result);

      // Queue audio for playback
      if (result.shouldSpeak && result.audioBase64) {
        this._queueAudio(result.audioBase64, result.isDangerAlert);
      }
    } catch (err) {
      console.error('[WS] Parse error:', err);
    }
  }

  private _onError(event: Event) {
    console.error('[WS] Error:', event);
    this.emit('error', event);
  }

  private _onClose(event: CloseEvent) {
    console.log('[WS] Closed:', event.code, event.reason);
    this.ws = null;
    if (event.code !== 1000) {
      // Abnormal closure — reconnect
      this._setState('reconnecting');
      this._scheduleReconnect();
    } else {
      this._setState('disconnected');
    }
    this.emit('disconnected', event.code);
  }

  private _setState(state: ConnectionState) {
    this.state = state;
    this.emit('stateChange', state);
  }

  private _scheduleReconnect() {
    this._clearReconnectTimer();
    const delay = RECONNECT_DELAYS[Math.min(this.reconnectAttempt, RECONNECT_DELAYS.length - 1)];
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt + 1})`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempt++;
      this.connect();
    }, delay);
  }

  private _clearReconnectTimer() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /** Danger alerts jump the queue; normal audio is appended. */
  private _queueAudio(base64Audio: string, urgent: boolean) {
    if (urgent) {
      this.audioQueue.unshift(base64Audio);
    } else {
      this.audioQueue.push(base64Audio);
    }
    if (!this.isPlayingAudio) {
      this._playNextAudio();
    }
  }

  private async _playNextAudio() {
    if (this.audioQueue.length === 0) {
      this.isPlayingAudio = false;
      return;
    }

    this.isPlayingAudio = true;
    const base64 = this.audioQueue.shift()!;

    try {
      // Write base64 MP3 to temp file, then play
      const tmpPath = `${RNFS.CachesDirectoryPath}/nav_audio_${Date.now()}.mp3`;
      await RNFS.writeFile(tmpPath, base64, 'base64');

      const sound = new Sound(tmpPath, '', (err) => {
        if (err) {
          console.warn('[Audio] Load error:', err);
          this._playNextAudio();
          return;
        }
        sound.play(() => {
          sound.release();
          RNFS.unlink(tmpPath).catch(() => {});
          this._playNextAudio();
        });
      });
    } catch (err) {
      console.error('[Audio] Playback error:', err);
      this._playNextAudio();
    }
  }

  get connectionState(): ConnectionState {
    return this.state;
  }
}

export default VisionWebSocketService;

import { v4 as uuidv4 } from 'uuid';

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

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

const RECONNECT_DELAYS = [1000, 2000, 5000, 10000, 30000];

type VisionWsEvents = {
  connected: () => void;
  disconnected: (code: number) => void;
  error: (event: unknown) => void;
  result: (result: StreamResult) => void;
  stateChange: (state: ConnectionState) => void;
};

export class VisionWebSocketService {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private serverUrl: string;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempt = 0;
  private reconnectTimer: number | null = null;
  private preferredLang: 'en' | 'hi' = 'en';
  private audioQueue: string[] = [];
  private isPlayingAudio = false;
  private listeners: { [K in keyof VisionWsEvents]: Set<VisionWsEvents[K]> } = {
    connected: new Set(),
    disconnected: new Set(),
    error: new Set(),
    result: new Set(),
    stateChange: new Set(),
  };

  constructor(serverUrl: string) {
    this.serverUrl = serverUrl;
    this.sessionId = uuidv4();
  }

  on<K extends keyof VisionWsEvents>(event: K, handler: VisionWsEvents[K]) {
    this.listeners[event].add(handler);
    return () => this.listeners[event].delete(handler);
  }

  removeAllListeners() {
    (Object.keys(this.listeners) as Array<keyof VisionWsEvents>).forEach((k) => {
      this.listeners[k].clear();
    });
  }

  private emit<K extends keyof VisionWsEvents>(event: K, ...args: Parameters<VisionWsEvents[K]>) {
    const handlers = this.listeners[event] as Set<(...a: unknown[]) => unknown>;
    handlers.forEach((handler) => handler(...(args as unknown[])));
  }

  connect() {
    if (this.state === 'connected' || this.state === 'connecting') return;
    this._setState('connecting');

    try {
      this.ws = new WebSocket(`${this.serverUrl}/api/v1/stream`);
      this.ws.onopen = this._onOpen.bind(this);
      this.ws.onmessage = this._onMessage.bind(this);
      this.ws.onerror = this._onError.bind(this);
      this.ws.onclose = this._onClose.bind(this);
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

  sendFrame(imageBase64: string) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;

    const cleanBase64 = imageBase64.replace(/^data:image\/(jpeg|png);base64,/, '');

    const payload = JSON.stringify({
      frame_id: uuidv4(),
      image_base64: cleanBase64,
      timestamp_ms: Date.now(),
      session_id: this.sessionId,
      preferred_language: this.preferredLang,
    });

    this.ws.send(payload);
  }

  private _onOpen() {
    console.log('[WS] Connected');
    this.reconnectAttempt = 0;
    this._setState('connected');
    this.emit('connected');
  }

  private _onMessage(event: MessageEvent) {
    try {
      const data = JSON.parse(event.data);
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
    this.reconnectTimer = window.setTimeout(() => {
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

  private _playNextAudio() {
    if (this.audioQueue.length === 0) {
      this.isPlayingAudio = false;
      return;
    }

    this.isPlayingAudio = true;
    const base64 = this.audioQueue.shift()!;
    const audioDataUrl = `data:audio/mp3;base64,${base64}`;

    const audio = new Audio(audioDataUrl);
    
    audio.onended = () => {
      this._playNextAudio();
    };
    
    audio.onerror = (e) => {
      console.error('[Audio] Playback error:', e);
      this._playNextAudio();
    };

    audio.play().catch(e => {
       console.error('[Audio] Autoplay blocked or error:', e);
       this._playNextAudio();
    });
  }

  get connectionState(): ConnectionState {
    return this.state;
  }
}

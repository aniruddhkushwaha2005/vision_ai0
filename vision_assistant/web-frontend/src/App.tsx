import React, { useEffect, useRef, useState, useCallback } from 'react';
import { VisionCamera } from './components/VisionCamera';
import { VisionWebSocketService, StreamResult, ConnectionState, NavigationDecision } from './services/VisionWebSocketService';
import { Play, Square, Languages, AlertTriangle } from 'lucide-react';
import './index.css';

const API_WS_URL = 'ws://localhost:8000';

const DECISION_DISPLAY: Record<NavigationDecision, { icon: string; label: string; color: string }> = {
  FORWARD:    { icon: '↑',  label: 'Move Forward',  color: '#22C55E' },
  TURN_LEFT:  { icon: '←',  label: 'Turn Left',     color: '#3B82F6' },
  TURN_RIGHT: { icon: '→',  label: 'Turn Right',    color: '#3B82F6' },
  STOP:       { icon: '■',  label: 'STOP',          color: '#EF4444' },
  DANGER:     { icon: '⚠',  label: 'DANGER!',       color: '#DC2626' },
  CLEAR:      { icon: '✓',  label: 'All Clear',     color: '#22C55E' },
};

export default function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [lastResult, setLastResult] = useState<StreamResult | null>(null);
  const [language, setLanguage] = useState<'en' | 'hi'>('en');
  const [fps, setFps] = useState(0);

  const wsRef = useRef<VisionWebSocketService | null>(null);
  const fpsCounter = useRef({ count: 0, last: Date.now() });

  useEffect(() => {
    const ws = new VisionWebSocketService(API_WS_URL);
    wsRef.current = ws;

    ws.on('stateChange', setConnectionState);
    ws.on('result', (result) => {
      setLastResult(result);
      
      fpsCounter.current.count++;
      const now = Date.now();
      if (now - fpsCounter.current.last >= 1000) {
        setFps(fpsCounter.current.count);
        fpsCounter.current = { count: 0, last: now };
      }
    });

    ws.connect();

    return () => {
      ws.disconnect();
      ws.removeAllListeners();
    };
  }, []);

  const toggleLanguage = useCallback(() => {
    const next = language === 'en' ? 'hi' : 'en';
    setLanguage(next);
    wsRef.current?.setLanguage(next);
  }, [language]);

  const handleFrame = useCallback((base64: string) => {
    wsRef.current?.sendFrame(base64);
  }, []);

  const toggleConnection = () => {
    if (connectionState === 'disconnected') {
      wsRef.current?.connect();
    } else {
      wsRef.current?.disconnect();
    }
  };

  const display = lastResult ? DECISION_DISPLAY[lastResult.decision] : DECISION_DISPLAY.CLEAR;
  const isDanger = lastResult?.isDangerAlert ?? false;

  return (
    <div className={`app-container ${isDanger ? 'danger-mode' : ''}`}>
      <VisionCamera onFrame={handleFrame} isActive={true} fps={12} />

      {isDanger && (
        <div className="danger-overlay">
          <AlertTriangle size={64} className="danger-icon-large pulse" />
        </div>
      )}

      <div className="top-bar glass-panel">
        <div className={`conn-dot ${connectionState === 'connected' ? 'connected' : 'disconnected'}`} />
        <span className="status-text">{connectionState.toUpperCase()}</span>
        <span className="fps-text">{fps} FPS</span>
      </div>

      <div className={`nav-card glass-panel ${isDanger ? 'nav-card-danger' : ''}`}>
        <div className="nav-icon" style={{ color: display.color }}>{display.icon}</div>
        <div className="nav-label" style={{ color: display.color }}>{display.label}</div>
        {lastResult && (
          <div className="speech-text">
            {language === 'hi' ? (lastResult.speechTextHi ?? lastResult.speechTextEn) : lastResult.speechTextEn}
          </div>
        )}
      </div>

      {lastResult && lastResult.detectedObjects.length > 0 && (
        <div className="objects-bar">
          {lastResult.detectedObjects.slice(0, 4).map((obj, i) => (
            <div key={i} className="object-pill glass-panel">
              {obj.className} · {obj.region}
            </div>
          ))}
        </div>
      )}

      <div className="bottom-bar">
        <button className="ctrl-btn glass-panel" onClick={toggleLanguage} aria-label="Toggle Language">
          <Languages size={24} />
          <span>{language === 'en' ? 'हिं' : 'EN'}</span>
        </button>

        <button 
          className={`ctrl-btn primary glass-panel ${connectionState === 'connected' ? 'active' : ''}`}
          onClick={toggleConnection}
        >
          {connectionState === 'connected' ? <Square size={24} /> : <Play size={24} />}
        </button>
      </div>
    </div>
  );
}

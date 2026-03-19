/**
 * NavigationScreen
 *
 * Primary user-facing screen. Streams camera frames to the backend
 * and renders real-time navigation feedback overlaid on the camera view.
 *
 * UX principles for accessibility:
 *  - Large, high-contrast visual indicators
 *  - Vibration haptics on direction changes
 *  - Voice (TTS audio) is primary feedback channel
 *  - All interactive elements have large touch targets (≥ 48×48dp)
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Alert,
  Dimensions,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  Vibration,
  View,
} from 'react-native';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';
import { useIsFocused } from '@react-navigation/native';

import VisionWebSocketService, {
  NavigationDecision,
  StreamResult,
} from '../services/VisionWebSocketService';
import { encodeFrameToBase64 } from '../utils/frameUtils';
import { API_WS_URL } from '../config';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// ── Direction arrow characters ─────────────────────────────────────────────
const DECISION_DISPLAY: Record<NavigationDecision, { icon: string; label: string; color: string }> = {
  FORWARD:    { icon: '↑',  label: 'Move Forward',  color: '#22C55E' },
  TURN_LEFT:  { icon: '←',  label: 'Turn Left',     color: '#3B82F6' },
  TURN_RIGHT: { icon: '→',  label: 'Turn Right',    color: '#3B82F6' },
  STOP:       { icon: '■',  label: 'STOP',          color: '#EF4444' },
  DANGER:     { icon: '⚠',  label: 'DANGER!',       color: '#DC2626' },
  CLEAR:      { icon: '✓',  label: 'All Clear',     color: '#22C55E' },
};

const VIBRATION_PATTERNS: Record<NavigationDecision, number[]> = {
  FORWARD:    [0, 50],
  TURN_LEFT:  [0, 100, 50, 100],
  TURN_RIGHT: [0, 100, 50, 100, 50, 100],
  STOP:       [0, 400],
  DANGER:     [0, 200, 100, 200, 100, 200],
  CLEAR:      [0, 50],
};

const FRAME_RATE_HZ = 12;   // frames per second sent to backend

export default function NavigationScreen() {
  const isFocused = useIsFocused();
  const device = useCameraDevice('back');
  const wsRef = useRef<VisionWebSocketService | null>(null);

  const [connectionState, setConnectionState] = useState('disconnected');
  const [lastResult, setLastResult] = useState<StreamResult | null>(null);
  const [language, setLanguage] = useState<'en' | 'hi'>('en');
  const [fps, setFps] = useState(0);

  const lastFrameSentAt = useRef<number>(0);
  const fpsCounter = useRef({ count: 0, last: Date.now() });

  // ── WebSocket lifecycle ───────────────────────────────────────────────────
  useEffect(() => {
    const ws = new VisionWebSocketService(API_WS_URL);
    wsRef.current = ws;

    ws.on('stateChange', setConnectionState);
    ws.on('result', (result: StreamResult) => {
      setLastResult(result);
      if (result.isDangerAlert || result.decision !== lastResult?.decision) {
        Vibration.vibrate(VIBRATION_PATTERNS[result.decision] || [0, 100]);
      }
      // Track FPS
      fpsCounter.current.count++;
      const now = Date.now();
      if (now - fpsCounter.current.last >= 1000) {
        setFps(fpsCounter.current.count);
        fpsCounter.current = { count: 0, last: now };
      }
    });
    ws.on('error', () => {});

    ws.connect();

    return () => {
      ws.disconnect();
      ws.removeAllListeners();
      wsRef.current = null;
    };
  }, []);

  const toggleLanguage = useCallback(() => {
    const next = language === 'en' ? 'hi' : 'en';
    setLanguage(next);
    wsRef.current?.setLanguage(next);
  }, [language]);

  // ── Frame processor (runs on camera thread via Worklet) ──────────────────
  const sendFrame = useCallback((base64: string) => {
    wsRef.current?.sendFrame(base64);
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    const now = Date.now();
    if (now - lastFrameSentAt.value < 1000 / FRAME_RATE_HZ) return;
    lastFrameSentAt.value = now;

    try {
      const base64 = encodeFrameToBase64(frame);
      runOnJS(sendFrame)(base64);
    } catch {
      // skip frame on encode error
    }
  }, [sendFrame]);

  // ── Render ────────────────────────────────────────────────────────────────
  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>Camera not available</Text>
      </View>
    );
  }

  const display = lastResult
    ? DECISION_DISPLAY[lastResult.decision]
    : DECISION_DISPLAY.CLEAR;

  const isDanger = lastResult?.isDangerAlert ?? false;

  return (
    <View style={styles.container}>
      {/* Camera feed */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused}
        frameProcessor={frameProcessor}
        frameProcessorFps={FRAME_RATE_HZ}
        photo={false}
        video={false}
      />

      {/* Danger overlay */}
      {isDanger && (
        <View style={styles.dangerOverlay} accessibilityLabel="DANGER! Stop immediately!" />
      )}

      {/* Top status bar */}
      <View style={styles.topBar}>
        <View style={[styles.connDot, { backgroundColor: connectionState === 'connected' ? '#22C55E' : '#EF4444' }]} />
        <Text style={styles.statusText}>{connectionState.toUpperCase()}</Text>
        <Text style={styles.fpsText}>{fps} FPS</Text>
      </View>

      {/* Navigation indicator (centre) */}
      <View
        style={[styles.navCard, isDanger && styles.navCardDanger]}
        accessibilityLiveRegion="polite"
        accessibilityLabel={display.label}
      >
        <Text style={[styles.navIcon, { color: display.color }]}>{display.icon}</Text>
        <Text style={[styles.navLabel, { color: display.color }]}>{display.label}</Text>
        {lastResult && (
          <Text style={styles.speechText}>
            {language === 'hi' ? lastResult.speechTextHi ?? lastResult.speechTextEn : lastResult.speechTextEn}
          </Text>
        )}
      </View>

      {/* Detected objects strip */}
      {lastResult && lastResult.detectedObjects.length > 0 && (
        <View style={styles.objectsBar}>
          {lastResult.detectedObjects.slice(0, 4).map((obj, i) => (
            <View key={i} style={styles.objectPill}>
              <Text style={styles.objectText}>
                {obj.className} · {obj.region}
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* Bottom controls */}
      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={styles.ctrlBtn}
          onPress={toggleLanguage}
          accessibilityLabel={`Switch to ${language === 'en' ? 'Hindi' : 'English'}`}
          accessibilityRole="button"
        >
          <Text style={styles.ctrlBtnText}>{language === 'en' ? 'हिं' : 'EN'}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.ctrlBtn, styles.ctrlBtnPrimary]}
          onPress={() => {
            if (connectionState === 'disconnected') {
              wsRef.current?.connect();
            } else {
              wsRef.current?.disconnect();
            }
          }}
          accessibilityLabel={connectionState === 'connected' ? 'Disconnect' : 'Connect'}
          accessibilityRole="button"
        >
          <Text style={styles.ctrlBtnText}>
            {connectionState === 'connected' ? '⏹' : '▶'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#000' },
  errorText: { color: '#fff', fontSize: 18 },

  dangerOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(220, 38, 38, 0.35)',
    borderWidth: 6,
    borderColor: '#DC2626',
  },

  topBar: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 56 : 32,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  connDot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { color: '#fff', fontSize: 13, fontWeight: '500', flex: 1 },
  fpsText: { color: 'rgba(255,255,255,0.6)', fontSize: 12 },

  navCard: {
    position: 'absolute',
    bottom: SCREEN_H * 0.22,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.72)',
    borderRadius: 20,
    paddingHorizontal: 32,
    paddingVertical: 24,
    alignItems: 'center',
    minWidth: 200,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  navCardDanger: {
    borderColor: '#DC2626',
    backgroundColor: 'rgba(120,0,0,0.85)',
  },
  navIcon: { fontSize: 64, lineHeight: 72 },
  navLabel: { fontSize: 22, fontWeight: '700', marginTop: 4 },
  speechText: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 15,
    textAlign: 'center',
    marginTop: 8,
    maxWidth: 260,
  },

  objectsBar: {
    position: 'absolute',
    bottom: SCREEN_H * 0.16,
    left: 16,
    right: 16,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
  },
  objectPill: {
    backgroundColor: 'rgba(0,0,0,0.65)',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  objectText: { color: '#fff', fontSize: 12 },

  bottomBar: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 48 : 28,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
  },
  ctrlBtn: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(30,30,30,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  ctrlBtnPrimary: {
    borderColor: '#3B82F6',
  },
  ctrlBtnText: { color: '#fff', fontSize: 22, fontWeight: '600' },
});

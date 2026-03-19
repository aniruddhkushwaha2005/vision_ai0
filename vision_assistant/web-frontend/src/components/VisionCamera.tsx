import React, { useEffect, useRef, useState } from 'react';

interface Props {
  onFrame: (base64: string) => void;
  isActive: boolean;
  fps?: number;
}

export const VisionCamera: React.FC<Props> = ({ onFrame, isActive, fps = 12 }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const requestRef = useRef<number>();
  const lastFrameTime = useRef<number>(0);

  useEffect(() => {
    let stream: MediaStream | null = null;
    if (isActive) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
          }
          setHasPermission(true);
        })
        .catch((err) => {
          console.error("Camera error:", err);
          setHasPermission(false);
        });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isActive]);

  const captureFrame = (time: number) => {
    if (!isActive || !videoRef.current || !canvasRef.current || videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA) {
      requestRef.current = requestAnimationFrame(captureFrame);
      return;
    }

    if (time - lastFrameTime.current >= 1000 / fps) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const base64 = canvas.toDataURL('image/jpeg', 0.6);
        onFrame(base64);
        lastFrameTime.current = time;
      }
    }
    requestRef.current = requestAnimationFrame(captureFrame);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(captureFrame);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isActive, fps, onFrame]);

  return (
    <div className="vision-camera-container">
      {hasPermission === false && <div className="camera-error">Camera access denied</div>}
      <video ref={videoRef} autoPlay playsInline muted className="vision-video" />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

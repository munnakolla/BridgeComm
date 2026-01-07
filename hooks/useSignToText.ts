/**
 * Custom hook for sign language to text workflow
 * Handles gesture recognition from camera input with video-based detection
 */

import { CameraView } from 'expo-camera';
import { RefObject, useCallback, useRef, useState } from 'react';
import BridgeCommAPI from '../services/api';

interface SignToTextState {
  isProcessing: boolean;
  isRecording: boolean;
  isContinuousMode: boolean;
  isCameraReady: boolean;
  intent: string;
  text: string;
  gestures: string[];
  confidence: number;
  error: string | null;
}

interface UseSignToTextReturn extends SignToTextState {
  captureAndRecognize: (cameraRef: RefObject<CameraView>) => Promise<void>;
  startVideoRecording: (cameraRef: RefObject<CameraView>) => Promise<void>;
  stopVideoRecording: () => Promise<void>;
  startContinuousDetection: (cameraRef: RefObject<CameraView>, intervalMs?: number) => void;
  stopContinuousDetection: () => void;
  clearSession: () => Promise<void>;
  clear: () => void;
  setCameraReady: (ready: boolean) => void;
}

export function useSignToText(userId?: string): UseSignToTextReturn {
  const [state, setState] = useState<SignToTextState>({
    isProcessing: false,
    isRecording: false,
    isContinuousMode: false,
    isCameraReady: false,
    intent: '',
    text: '',
    gestures: [],
    confidence: 0,
    error: null,
  });

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cameraRefForRecording = useRef<CameraView | null>(null);
  const recordingPromiseRef = useRef<Promise<{ uri: string } | undefined> | null>(null);
  const isCameraReadyRef = useRef<boolean>(false);

  // Keep ref in sync with state
  const setCameraReady = useCallback((ready: boolean) => {
    isCameraReadyRef.current = ready;
    setState((prev: SignToTextState) => ({ ...prev, isCameraReady: ready }));
  }, []);

  const captureAndRecognize = useCallback(async (
    cameraRef: RefObject<CameraView>
  ): Promise<void> => {
    if (!cameraRef.current) {
      setState((prev: SignToTextState) => ({ ...prev, error: 'Camera not ready' }));
      return;
    }

    try {
      setState((prev: SignToTextState) => ({ ...prev, isProcessing: true, error: null }));

      // Capture image
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.7, // Better quality for gesture detection
      });

      if (!photo?.base64) {
        throw new Error('Failed to capture image');
      }

      console.log(`Captured image: ${photo.base64.substring(0, 50)}... (${photo.base64.length} chars)`);

      // Send to API for recognition with session support
      const result = await BridgeCommAPI.signToIntent(
        photo.base64,
        userId
      );

      console.log('Sign language API result:', JSON.stringify(result, null, 2));

      setState((prev: SignToTextState) => ({
        ...prev,
        isProcessing: false,
        intent: result.intent,
        text: result.text,
        gestures: result.gestures_detected,
        confidence: result.confidence,
      }));

    } catch (error) {
      console.error('Failed to recognize sign:', error);
      setState((prev: SignToTextState) => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Recognition failed',
      }));
    }
  }, [userId]);

  // Start video recording for continuous gesture detection
  const startVideoRecording = useCallback(async (
    cameraRef: RefObject<CameraView>
  ): Promise<void> => {
    if (!cameraRef.current) {
      setState((prev: SignToTextState) => ({ ...prev, error: 'Camera not ready' }));
      return;
    }

    // Check if camera is ready for recording using ref (avoids stale closure)
    if (!isCameraReadyRef.current) {
      setState((prev: SignToTextState) => ({ 
        ...prev, 
        error: 'Camera is initializing. Please wait a moment and try again.' 
      }));
      return;
    }

    try {
      setState((prev: SignToTextState) => ({ 
        ...prev, 
        isRecording: true, 
        error: null,
        // Clear previous results when starting new recording
        text: '',
        gestures: [],
        intent: '',
      }));

      cameraRefForRecording.current = cameraRef.current;
      
      // Start recording video - this returns a promise that resolves when recording stops
      recordingPromiseRef.current = cameraRef.current.recordAsync({
        maxDuration: 30, // Max 30 seconds
      });

      console.log('Video recording started');
    } catch (error) {
      console.error('Failed to start video recording:', error);
      setState((prev: SignToTextState) => ({
        ...prev,
        isRecording: false,
        error: error instanceof Error ? error.message : 'Failed to start recording',
      }));
    }
  }, []);

  // Stop video recording and process the video
  const stopVideoRecording = useCallback(async (): Promise<void> => {
    if (!cameraRefForRecording.current || !recordingPromiseRef.current) {
      setState((prev: SignToTextState) => ({ ...prev, isRecording: false }));
      return;
    }

    try {
      // Stop the recording
      cameraRefForRecording.current.stopRecording();
      
      setState((prev: SignToTextState) => ({ 
        ...prev, 
        isRecording: false, 
        isProcessing: true 
      }));

      // Wait for the recording to finish and get the URI
      const video = await recordingPromiseRef.current;
      recordingPromiseRef.current = null;

      if (!video?.uri) {
        throw new Error('No video recorded');
      }

      console.log('Video recorded:', video.uri);

      // Convert video to base64 and send to API
      const videoBase64 = await BridgeCommAPI.uriToBase64(video.uri);

      // Use the new sign video API with I3D/Pose-LSTM models for better accuracy
      const result = await BridgeCommAPI.processSignVideo(
        videoBase64,
        true, // use Groq correction
        userId
      );

      setState((prev: SignToTextState) => ({
        ...prev,
        isProcessing: false,
        intent: 'sign_video_recognition',
        text: result.sentence,
        gestures: result.recognized_words,
        confidence: result.confidence,
      }));

    } catch (error) {
      console.error('Failed to process video:', error);
      recordingPromiseRef.current = null;
      setState((prev: SignToTextState) => ({
        ...prev,
        isRecording: false,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Video processing failed',
      }));
    }
  }, [userId]);

  const startContinuousDetection = useCallback((
    cameraRef: RefObject<CameraView>,
    intervalMs: number = 500 // Capture every 500ms for smooth video-like detection
  ) => {
    // Stop any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setState((prev: SignToTextState) => ({ ...prev, isContinuousMode: true }));

    // Start continuous capture
    intervalRef.current = setInterval(() => {
      captureAndRecognize(cameraRef);
    }, intervalMs);
  }, [captureAndRecognize]);

  const stopContinuousDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setState((prev: SignToTextState) => ({ ...prev, isContinuousMode: false }));
  }, []);

  const clearSession = useCallback(async () => {
    try {
      if (userId) {
        await BridgeCommAPI.clearSignSession(userId);
      }
      clear();
    } catch (error) {
      console.error('Failed to clear session:', error);
    }
  }, [userId]);

  const clear = useCallback(() => {
    stopContinuousDetection();
    if (cameraRefForRecording.current && recordingPromiseRef.current) {
      cameraRefForRecording.current.stopRecording();
    }
    recordingPromiseRef.current = null;
    setState((prev: SignToTextState) => ({
      isProcessing: false,
      isRecording: false,
      isContinuousMode: false,
      isCameraReady: prev.isCameraReady, // Preserve camera ready state
      intent: '',
      text: '',
      gestures: [],
      confidence: 0,
      error: null,
    }));
  }, [stopContinuousDetection]);

  return {
    ...state,
    captureAndRecognize,
    startVideoRecording,
    stopVideoRecording,
    startContinuousDetection,
    stopContinuousDetection,
    clearSession,
    clear,
    setCameraReady,
  };
}

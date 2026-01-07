/**
 * Custom hook for speech-to-symbols workflow
 * Handles the complete flow from speech to simplified text and symbols
 */

import { Audio } from 'expo-av';
import { useCallback, useState } from 'react';
import BridgeCommAPI, {
    Symbol
} from '../services/api';

interface SpeechToSymbolsState {
  isRecording: boolean;
  isProcessing: boolean;
  recognizedText: string;
  simplifiedText: string;
  symbols: Symbol[];
  confidence: number;
  error: string | null;
}

interface UseSpeechToSymbolsReturn extends SpeechToSymbolsState {
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  processText: (text: string) => Promise<void>;
  clear: () => void;
}

const initialState: SpeechToSymbolsState = {
  isRecording: false,
  isProcessing: false,
  recognizedText: '',
  simplifiedText: '',
  symbols: [],
  confidence: 0,
  error: null,
};

export function useSpeechToSymbols(userId?: string): UseSpeechToSymbolsReturn {
  const [state, setState] = useState<SpeechToSymbolsState>(initialState);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);

  // Helper function to safely cleanup any existing recording
  const cleanupRecording = useCallback(async (rec: Audio.Recording | null): Promise<void> => {
    if (!rec) return;
    try {
      const status = await rec.getStatusAsync();
      if (status.isRecording) {
        await rec.stopAndUnloadAsync();
      } else if (status.isDoneRecording === false) {
        // Recording was prepared but not started or already stopped
        await rec.stopAndUnloadAsync().catch(() => {});
      }
    } catch (e) {
      // Recording might already be unloaded, ignore
      console.log('Cleanup recording (may already be cleaned):', e);
    }
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      setState((prev: SpeechToSymbolsState) => ({ ...prev, error: null }));
      
      // CRITICAL: Clean up any existing recording first
      if (recording) {
        await cleanupRecording(recording);
        setRecording(null);
      }
      
      // Request permissions
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        setState((prev: SpeechToSymbolsState) => ({ ...prev, error: 'Microphone permission required' }));
        return;
      }

      // Reset audio mode to allow new recording
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });

      // Configure audio mode for recording
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      });

      // Use createAsync which properly handles recording lifecycle
      try {
        const { recording: newRecording } = await Audio.Recording.createAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY
        );
        setRecording(newRecording);
        setState((prev: SpeechToSymbolsState) => ({ ...prev, isRecording: true }));
      } catch (createError) {
        console.error('Recording creation failed:', createError);
        throw createError;
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      // Reset audio mode on error
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      }).catch(() => {});
      setState((prev: SpeechToSymbolsState) => ({ 
        ...prev, 
        error: `Failed to start recording: ${error}` 
      }));
    }
  }, [recording, cleanupRecording]);

  const stopRecording = useCallback(async (): Promise<void> => {
    if (!recording) return;

    try {
      setState((prev: SpeechToSymbolsState) => ({ ...prev, isRecording: false, isProcessing: true }));

      // Stop recording
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);

      // Reset audio mode after recording
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
      }).catch(() => {});

      if (!uri) {
        throw new Error('No recording URI');
      }

      // Convert to base64
      const audioBase64 = await BridgeCommAPI.uriToBase64(uri);

      // Step 1: Speech to Text
      const speechResult = await BridgeCommAPI.speechToText(
        audioBase64,
        'en-US',
        userId
      );

      if (!speechResult.text) {
        setState((prev: SpeechToSymbolsState) => ({
          ...prev,
          isProcessing: false,
          error: 'No speech detected',
        }));
        return;
      }

      // Step 2: Text to Symbols
      const symbolsResult = await BridgeCommAPI.textToSymbols(
        speechResult.text,
        true,
        10,
        userId
      );

      setState((prev: SpeechToSymbolsState) => ({
        ...prev,
        isProcessing: false,
        recognizedText: speechResult.text,
        simplifiedText: symbolsResult.simplified_text,
        symbols: symbolsResult.symbols,
        confidence: symbolsResult.confidence,
      }));

    } catch (error) {
      console.error('Failed to process recording:', error);
      setState((prev: SpeechToSymbolsState) => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Processing failed',
      }));
    }
  }, [recording, userId]);

  const processText = useCallback(async (text: string): Promise<void> => {
    if (!text.trim()) return;

    try {
      setState((prev: SpeechToSymbolsState) => ({ ...prev, isProcessing: true, error: null }));

      const symbolsResult = await BridgeCommAPI.textToSymbols(
        text,
        true,
        10,
        userId
      );

      setState((prev: SpeechToSymbolsState) => ({
        ...prev,
        isProcessing: false,
        recognizedText: text,
        simplifiedText: symbolsResult.simplified_text,
        symbols: symbolsResult.symbols,
        confidence: symbolsResult.confidence,
      }));
    } catch (error) {
      console.error('Failed to process text:', error);
      setState((prev: SpeechToSymbolsState) => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Processing failed',
      }));
    }
  }, [userId]);

  const clear = useCallback((): void => {
    setState(initialState);
  }, []);

  return {
    ...state,
    startRecording,
    stopRecording,
    processText,
    clear,
  };
}

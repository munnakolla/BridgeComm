/**
 * Custom hook for behavior to intent recognition
 * Tracks user interactions and sends for AI analysis
 */

import { useCallback, useRef, useState } from 'react';
import BridgeCommAPI, { BehaviorData } from '../services/api';

interface BehaviorState {
  isTracking: boolean;
  isProcessing: boolean;
  intent: string;
  text: string;
  behaviorType: string;
  confidence: number;
  error: string | null;
}

interface TouchEvent {
  type: 'tap' | 'double_tap' | 'long_press' | 'swipe';
  x: number;
  y: number;
  duration?: number;
  direction?: string;
  timestamp: number;
}

interface UseBehaviorToIntentReturn extends BehaviorState {
  startTracking: () => void;
  stopTracking: () => void;
  recordTouch: (event: TouchEvent) => void;
  recordInteraction: (element: string) => void;
  analyzeNow: () => Promise<void>;
  clear: () => void;
}

export function useBehaviorToIntent(userId?: string): UseBehaviorToIntentReturn {
  const [state, setState] = useState<BehaviorState>({
    isTracking: false,
    isProcessing: false,
    intent: '',
    text: '',
    behaviorType: '',
    confidence: 0,
    error: null,
  });

  const touchPatternsRef = useRef<TouchEvent[]>([]);
  const interactionSequenceRef = useRef<string[]>([]);

  const startTracking = useCallback(() => {
    touchPatternsRef.current = [];
    interactionSequenceRef.current = [];
    setState((prev: BehaviorState) => ({ ...prev, isTracking: true, error: null }));
  }, []);

  const stopTracking = useCallback(() => {
    setState((prev: BehaviorState) => ({ ...prev, isTracking: false }));
  }, []);

  const recordTouch = useCallback((event: TouchEvent) => {
    if (state.isTracking) {
      touchPatternsRef.current.push(event);
    }
  }, [state.isTracking]);

  const recordInteraction = useCallback((element: string) => {
    if (state.isTracking) {
      interactionSequenceRef.current.push(element);
    }
  }, [state.isTracking]);

  const analyzeNow = useCallback(async () => {
    const touchPatterns = touchPatternsRef.current;
    const interactionSequence = interactionSequenceRef.current;

    if (touchPatterns.length === 0 && interactionSequence.length === 0) {
      setState((prev: BehaviorState) => ({ ...prev, error: 'No behavior data recorded' }));
      return;
    }

    try {
      setState((prev: BehaviorState) => ({ ...prev, isProcessing: true, error: null }));

      const behaviorData: BehaviorData = {
        touch_patterns: touchPatterns.map((t: TouchEvent) => ({
          type: t.type,
          x: t.x,
          y: t.y,
          duration: t.duration,
          direction: t.direction,
        })),
        interaction_sequence: interactionSequence,
      };

      const result = await BridgeCommAPI.behaviorToIntent(
        behaviorData,
        undefined,
        userId
      );

      setState((prev: BehaviorState) => ({
        ...prev,
        isProcessing: false,
        intent: result.intent,
        text: result.text,
        behaviorType: result.behavior_type,
        confidence: result.confidence,
      }));

    } catch (error) {
      console.error('Failed to analyze behavior:', error);
      setState((prev: BehaviorState) => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
      }));
    }
  }, [userId]);

  const clear = useCallback(() => {
    touchPatternsRef.current = [];
    interactionSequenceRef.current = [];
    setState({
      isTracking: false,
      isProcessing: false,
      intent: '',
      text: '',
      behaviorType: '',
      confidence: 0,
      error: null,
    });
  }, []);

  return {
    ...state,
    startTracking,
    stopTracking,
    recordTouch,
    recordInteraction,
    analyzeNow,
    clear,
  };
}

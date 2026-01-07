/**
 * Custom hook for emotion detection from facial expressions
 * Uses camera to detect emotions and convert them to meaningful sentences
 * Also provides sign language mappings for each detected emotion
 */

import { CameraView } from 'expo-camera';
import { RefObject, useCallback, useState } from 'react';
import BridgeCommAPI from '../services/api';

// Emotion to sentence mappings with sign language equivalents
export interface EmotionMapping {
  emotion: string;
  emoji: string;
  sentences: string[];
  signLanguageGestures: string[];
  relatedSymbols: { emoji: string; name: string }[];
}

// Comprehensive emotion mappings trained for accessibility communication
export const EMOTION_MAPPINGS: Record<string, EmotionMapping> = {
  happy: {
    emotion: 'Happy',
    emoji: '😊',
    sentences: [
      'I am feeling happy right now.',
      'This makes me feel good.',
      'I am pleased with this.',
      'Everything is going well.',
    ],
    signLanguageGestures: ['smile', 'thumbs_up', 'clap'],
    relatedSymbols: [
      { emoji: '😊', name: 'Happy' },
      { emoji: '✅', name: 'Good' },
      { emoji: '👍', name: 'Thumbs Up' },
    ],
  },
  sad: {
    emotion: 'Sad',
    emoji: '😢',
    sentences: [
      'I am feeling sad.',
      'I need some comfort.',
      'Something is bothering me.',
      'I am not feeling well emotionally.',
    ],
    signLanguageGestures: ['frown', 'tear', 'down'],
    relatedSymbols: [
      { emoji: '😢', name: 'Sad' },
      { emoji: '🤗', name: 'Need Hug' },
      { emoji: '💙', name: 'Comfort' },
    ],
  },
  angry: {
    emotion: 'Angry',
    emoji: '😠',
    sentences: [
      'I am feeling frustrated.',
      'This is upsetting me.',
      'I need a moment to calm down.',
      'Please give me some space.',
    ],
    signLanguageGestures: ['fist', 'shake_head', 'push_away'],
    relatedSymbols: [
      { emoji: '😠', name: 'Angry' },
      { emoji: '🛑', name: 'Stop' },
      { emoji: '🙏', name: 'Please' },
    ],
  },
  surprised: {
    emotion: 'Surprised',
    emoji: '😮',
    sentences: [
      'That surprised me!',
      'I did not expect that.',
      'Wow, that is interesting!',
      'This is unexpected.',
    ],
    signLanguageGestures: ['wide_eyes', 'open_mouth', 'hands_up'],
    relatedSymbols: [
      { emoji: '😮', name: 'Surprised' },
      { emoji: '❗', name: 'Attention' },
      { emoji: '👀', name: 'Look' },
    ],
  },
  fearful: {
    emotion: 'Fearful',
    emoji: '😨',
    sentences: [
      'I am feeling scared.',
      'I need help, please.',
      'Something is frightening me.',
      'I feel unsafe.',
    ],
    signLanguageGestures: ['shake', 'hide', 'point_help'],
    relatedSymbols: [
      { emoji: '😨', name: 'Scared' },
      { emoji: '🆘', name: 'Help' },
      { emoji: '🤲', name: 'Please' },
    ],
  },
  disgusted: {
    emotion: 'Disgusted',
    emoji: '🤢',
    sentences: [
      'I do not like this.',
      'This is unpleasant.',
      'Please remove this.',
      'I want something different.',
    ],
    signLanguageGestures: ['nose_wrinkle', 'push_away', 'no'],
    relatedSymbols: [
      { emoji: '🤢', name: 'Dislike' },
      { emoji: '❌', name: 'No' },
      { emoji: '🔄', name: 'Change' },
    ],
  },
  neutral: {
    emotion: 'Neutral',
    emoji: '😐',
    sentences: [
      'I am here and listening.',
      'I am okay.',
      'I have no strong feelings right now.',
      'I am ready to communicate.',
    ],
    signLanguageGestures: ['calm', 'still', 'attention'],
    relatedSymbols: [
      { emoji: '😐', name: 'Neutral' },
      { emoji: '👂', name: 'Listening' },
      { emoji: '👋', name: 'Ready' },
    ],
  },
  tired: {
    emotion: 'Tired',
    emoji: '😴',
    sentences: [
      'I am feeling tired.',
      'I need to rest.',
      'I am exhausted.',
      'Can I take a break?',
    ],
    signLanguageGestures: ['yawn', 'close_eyes', 'sleep'],
    relatedSymbols: [
      { emoji: '😴', name: 'Tired' },
      { emoji: '🛏️', name: 'Rest' },
      { emoji: '⏸️', name: 'Break' },
    ],
  },
  confused: {
    emotion: 'Confused',
    emoji: '😕',
    sentences: [
      'I do not understand.',
      'Can you explain that again?',
      'I am confused.',
      'Please help me understand.',
    ],
    signLanguageGestures: ['tilt_head', 'shrug', 'point_question'],
    relatedSymbols: [
      { emoji: '😕', name: 'Confused' },
      { emoji: '❓', name: 'Question' },
      { emoji: '🆘', name: 'Help' },
    ],
  },
  pain: {
    emotion: 'In Pain',
    emoji: '😣',
    sentences: [
      'I am in pain.',
      'Something hurts.',
      'I need medical attention.',
      'Please help me, I am hurting.',
    ],
    signLanguageGestures: ['grimace', 'point_body', 'help'],
    relatedSymbols: [
      { emoji: '😣', name: 'Pain' },
      { emoji: '🏥', name: 'Medical' },
      { emoji: '🆘', name: 'Help' },
    ],
  },
  hungry: {
    emotion: 'Hungry',
    emoji: '😋',
    sentences: [
      'I am hungry.',
      'I would like something to eat.',
      'It is time for food.',
      'Can I have a meal?',
    ],
    signLanguageGestures: ['eat', 'point_mouth', 'food'],
    relatedSymbols: [
      { emoji: '🍽️', name: 'Hungry' },
      { emoji: '🍔', name: 'Food' },
      { emoji: '🙏', name: 'Please' },
    ],
  },
  thirsty: {
    emotion: 'Thirsty',
    emoji: '🥤',
    sentences: [
      'I am thirsty.',
      'I need something to drink.',
      'Can I have water?',
      'I would like a beverage.',
    ],
    signLanguageGestures: ['drink', 'point_throat', 'water'],
    relatedSymbols: [
      { emoji: '💧', name: 'Thirsty' },
      { emoji: '💦', name: 'Water' },
      { emoji: '🙏', name: 'Please' },
    ],
  },
};

// Facial feature analysis weights for emotion detection
interface FacialFeatures {
  eyebrowRaise: number;
  eyebrowFurrow: number;
  eyeOpenness: number;
  mouthOpenness: number;
  mouthCurve: number; // positive = smile, negative = frown
  noseWrinkle: number;
  headTilt: number;
}

// Emotion detection state
interface EmotionDetectionState {
  isProcessing: boolean;
  detectedEmotion: string;
  confidence: number;
  sentence: string;
  signLanguageGestures: string[];
  relatedSymbols: { emoji: string; name: string }[];
  allEmotions: { emotion: string; confidence: number }[];
  error: string | null;
}

interface UseEmotionDetectionReturn extends EmotionDetectionState {
  detectEmotion: (cameraRef: RefObject<CameraView>) => Promise<void>;
  setEmotionManually: (emotion: string) => void;
  getSentenceVariation: () => string;
  clear: () => void;
}

const initialState: EmotionDetectionState = {
  isProcessing: false,
  detectedEmotion: '',
  confidence: 0,
  sentence: '',
  signLanguageGestures: [],
  relatedSymbols: [],
  allEmotions: [],
  error: null,
};

// Simulated emotion detection model (trained patterns)
function analyzeEmotionFromFeatures(features: FacialFeatures): { emotion: string; confidence: number }[] {
  const scores: Record<string, number> = {
    happy: 0,
    sad: 0,
    angry: 0,
    surprised: 0,
    fearful: 0,
    disgusted: 0,
    neutral: 0,
    tired: 0,
    confused: 0,
    pain: 0,
  };

  // Happy: raised cheeks, smile, slightly closed eyes
  scores.happy = (features.mouthCurve * 0.4) + (1 - features.eyeOpenness) * 0.2 + 0.4;

  // Sad: frown, droopy eyes, lowered eyebrows
  scores.sad = (-features.mouthCurve * 0.3) + (1 - features.eyebrowRaise) * 0.2 + 0.2;

  // Angry: furrowed brows, tight mouth
  scores.angry = (features.eyebrowFurrow * 0.4) + (1 - features.mouthOpenness) * 0.2 + 0.1;

  // Surprised: raised eyebrows, wide eyes, open mouth
  scores.surprised = (features.eyebrowRaise * 0.3) + (features.eyeOpenness * 0.3) + (features.mouthOpenness * 0.3);

  // Fearful: wide eyes, raised eyebrows, tense
  scores.fearful = (features.eyeOpenness * 0.3) + (features.eyebrowRaise * 0.2) + 0.1;

  // Disgusted: nose wrinkle, raised upper lip
  scores.disgusted = (features.noseWrinkle * 0.5) + 0.1;

  // Tired: droopy eyes, relaxed face
  scores.tired = (1 - features.eyeOpenness) * 0.4 + (1 - features.eyebrowRaise) * 0.2 + 0.1;

  // Confused: tilted head, furrowed brow
  scores.confused = Math.abs(features.headTilt) * 0.3 + features.eyebrowFurrow * 0.2 + 0.1;

  // Pain: grimace, furrowed brow, closed eyes
  scores.pain = (features.eyebrowFurrow * 0.3) + (-features.mouthCurve * 0.2) + (1 - features.eyeOpenness) * 0.2;

  // Neutral: balanced features
  scores.neutral = 0.5 - Math.abs(features.mouthCurve) * 0.2 - Math.abs(features.eyebrowRaise - 0.5) * 0.2;

  // Normalize and sort
  const total = Object.values(scores).reduce((a, b) => a + Math.max(0, b), 0);
  const emotions = Object.entries(scores)
    .map(([emotion, score]) => ({
      emotion,
      confidence: total > 0 ? Math.max(0, score) / total : 0,
    }))
    .sort((a, b) => b.confidence - a.confidence);

  return emotions;
}

// Analyze emotion from backend face detection result
function analyzeEmotionFromBackendResult(expression: string): { emotion: string; confidence: number }[] {
  // Map backend expression to emotion
  const emotionMap: Record<string, string> = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'surprised': 'surprised',
    'fearful': 'fearful',
    'disgusted': 'disgusted',
    'neutral': 'neutral',
    'unknown': 'neutral',
  };
  
  const mappedEmotion = emotionMap[expression.toLowerCase()] || 'neutral';
  
  // Return with high confidence for detected emotion
  return [
    { emotion: mappedEmotion, confidence: 0.85 },
    { emotion: 'neutral', confidence: 0.15 },
  ];
}

export function useEmotionDetection(): UseEmotionDetectionReturn {
  const [state, setState] = useState<EmotionDetectionState>(initialState);
  const [sentenceIndex, setSentenceIndex] = useState(0);

  const detectEmotion = useCallback(async (
    cameraRef: RefObject<CameraView>
  ): Promise<void> => {
    setState(prev => ({ ...prev, isProcessing: true, error: null }));

    try {
      let emotionResults: { emotion: string; confidence: number }[];

      // Try to capture from camera and send to API
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.7,
          });

          if (photo?.base64) {
            // Try API call for emotion detection
            const result = await BridgeCommAPI.behaviorToIntent({
              facial_expressions: {
                image_base64: photo.base64,
                analysis_type: 'emotion',
              },
            });

            // Parse API response if available
            if (result.intent) {
              const emotion = result.intent.toLowerCase();
              const mapping = EMOTION_MAPPINGS[emotion] || EMOTION_MAPPINGS.neutral;
              
              setState(prev => ({
                ...prev,
                isProcessing: false,
                detectedEmotion: mapping.emotion,
                confidence: result.confidence,
                sentence: mapping.sentences[0],
                signLanguageGestures: mapping.signLanguageGestures,
                relatedSymbols: mapping.relatedSymbols,
                allEmotions: [{ emotion, confidence: result.confidence }],
              }));
              return;
            }
          }
        } catch (apiError) {
          console.log('API not available, using local emotion detection');
        }
      }

      // Fallback: Try direct vision API for face analysis
      if (cameraRef.current) {
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.7,
          });
          
          if (photo?.base64) {
            const faceResult = await BridgeCommAPI.analyzeFace(photo.base64);
            if (faceResult.faces && faceResult.faces.length > 0) {
              const expression = faceResult.faces[0].expression || 'neutral';
              emotionResults = analyzeEmotionFromBackendResult(expression);
            } else {
              // No face detected, default to neutral
              emotionResults = [{ emotion: 'neutral', confidence: 0.5 }];
            }
          } else {
            emotionResults = [{ emotion: 'neutral', confidence: 0.5 }];
          }
        } catch (faceError) {
          console.log('Face analysis error:', faceError);
          emotionResults = [{ emotion: 'neutral', confidence: 0.5 }];
        }
      } else {
        emotionResults = [{ emotion: 'neutral', confidence: 0.5 }];
      }

      const topEmotion = emotionResults[0];
      const mapping = EMOTION_MAPPINGS[topEmotion.emotion] || EMOTION_MAPPINGS.neutral;

      setState(prev => ({
        ...prev,
        isProcessing: false,
        detectedEmotion: mapping.emotion,
        confidence: topEmotion.confidence,
        sentence: mapping.sentences[0],
        signLanguageGestures: mapping.signLanguageGestures,
        relatedSymbols: mapping.relatedSymbols,
        allEmotions: emotionResults.slice(0, 5),
      }));

    } catch (error) {
      console.error('Emotion detection failed:', error);
      setState(prev => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Detection failed',
      }));
    }
  }, []);

  const setEmotionManually = useCallback((emotion: string) => {
    const mapping = EMOTION_MAPPINGS[emotion.toLowerCase()];
    if (mapping) {
      setState(prev => ({
        ...prev,
        detectedEmotion: mapping.emotion,
        confidence: 1.0,
        sentence: mapping.sentences[0],
        signLanguageGestures: mapping.signLanguageGestures,
        relatedSymbols: mapping.relatedSymbols,
        allEmotions: [{ emotion: emotion.toLowerCase(), confidence: 1.0 }],
        error: null,
      }));
    }
  }, []);

  const getSentenceVariation = useCallback((): string => {
    const emotion = state.detectedEmotion.toLowerCase();
    const mapping = EMOTION_MAPPINGS[emotion];
    if (mapping) {
      const newIndex = (sentenceIndex + 1) % mapping.sentences.length;
      setSentenceIndex(newIndex);
      const newSentence = mapping.sentences[newIndex];
      setState(prev => ({ ...prev, sentence: newSentence }));
      return newSentence;
    }
    return state.sentence;
  }, [state.detectedEmotion, sentenceIndex]);

  const clear = useCallback(() => {
    setState(initialState);
    setSentenceIndex(0);
  }, []);

  return {
    ...state,
    detectEmotion,
    setEmotionManually,
    getSentenceVariation,
    clear,
  };
}

export default useEmotionDetection;

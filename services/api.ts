/**
 * BridgeComm API Client
 * Connects the frontend app to the Azure-powered backend.
 */

// Declare __DEV__ global for React Native
declare const __DEV__: boolean;

// Configuration - driven by env for production; falls back to a safe default
const getApiBaseUrl = (): string => {
  const envUrl = (process as any)?.env?.EXPO_PUBLIC_API_BASE_URL as string | undefined;
  const trimmedEnv = envUrl?.trim().replace(/\/$/, '');

  if (trimmedEnv) return trimmedEnv;

  // Development convenience fallback (replace with your LAN IP if testing on device)
  const devDefault = 'http://localhost:8000';
  return typeof __DEV__ !== 'undefined' && __DEV__
    ? devDefault
    : 'https://bridgecomm-api.azurewebsites.net';
};

const API_BASE_URL = getApiBaseUrl();

// Types matching the backend schemas
export interface Symbol {
  id: string;
  name: string;
  url: string;
  category?: string;
}

export interface SpeechToTextResponse {
  text: string;
  confidence: number;
  language: string;
  duration_ms?: number;
}

export interface TextToSymbolsResponse {
  original_text: string;
  simplified_text: string;
  symbols: Symbol[];
  keywords: string[];
  confidence: number;
}

export interface SignToIntentResponse {
  intent: string;
  text: string;
  confidence: number;
  gestures_detected: string[];
  hand_landmarks?: any;
  session_summary?: {
    total_gestures: number;
    accumulated_gestures: string[];
    formed_sentence: string;
  };
}

export interface BehaviorData {
  touch_patterns?: any[];
  eye_tracking?: any;
  facial_expressions?: any;
  motion_data?: any[];
  interaction_sequence?: string[];
}

export interface BehaviorToIntentResponse {
  intent: string;
  text: string;
  confidence: number;
  behavior_type: string;
  features_extracted: any;
}

export interface GenerateTextResponse {
  text: string;
  intent: string;
  confidence: number;
  alternatives?: string[];
}

export interface TextToSpeechResponse {
  audio_url: string;
  audio_base64?: string;
  duration_ms: number;
  format: string;
}

export interface UserProfile {
  user_id: string;
  display_name?: string;
  communication_mode: 'sign' | 'eye' | 'touch' | 'behavior';
  preferences: any;
  personalization_data: any;
  created_at?: string;
  updated_at?: string;
}

export type FeedbackType = 'correct' | 'incorrect' | 'partial';
export type InputMode = 'sign' | 'eye' | 'touch' | 'behavior';

// API Error class
export class ApiError extends Error {
  status: number;
  
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

// Helper function for API calls
async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs = 12000
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Unknown error' }));
      const message = error.message || error.detail || `Request failed (${response.status})`;
      throw new ApiError(message, response.status);
    }

    return response.json();
  } catch (err: unknown) {
    if ((err as Error).name === 'AbortError') {
      throw new ApiError('Request timed out. Check network or server.', 408);
    }
    if (err instanceof ApiError) throw err;
    const message = err instanceof Error ? err.message : 'Network error';
    throw new ApiError(message, 500);
  } finally {
    clearTimeout(timer);
  }
}

export async function pingApi(): Promise<boolean> {
  try {
    await apiCall('/health', {}, 4000);
    return true;
  } catch (err) {
    return false;
  }
}

// =============================================================================
// SPEECH APIs
// =============================================================================

/**
 * Convert speech audio to text using Azure Speech-to-Text.
 */
export async function speechToText(
  audioBase64: string,
  language: string = 'en-US',
  userId?: string
): Promise<SpeechToTextResponse> {
  return apiCall('/azure/speech-to-text', {
    method: 'POST',
    body: JSON.stringify({
      audio_base64: audioBase64,
      language,
      user_id: userId,
    }),
  });
}

/**
 * Convert text to speech audio.
 */
export async function textToSpeech(
  text: string,
  voice: string = 'en-US-JennyNeural',
  rate: number = 1.0,
  pitch: number = 1.0
): Promise<TextToSpeechResponse> {
  return apiCall('/azure/text-to-speech', {
    method: 'POST',
    body: JSON.stringify({ text, voice, rate, pitch }),
  });
}

// =============================================================================
// SYMBOL APIs
// =============================================================================

/**
 * Convert text to simplified text and visual symbols.
 */
export async function textToSymbols(
  text: string,
  simplify: boolean = true,
  maxSymbols: number = 10,
  userId?: string
): Promise<TextToSymbolsResponse> {
  return apiCall('/azure/text-to-symbols', {
    method: 'POST',
    body: JSON.stringify({
      text,
      simplify,
      max_symbols: maxSymbols,
      user_id: userId,
    }),
  });
}

/**
 * Get all symbol categories.
 */
export async function getSymbolCategories(): Promise<{ categories: string[] }> {
  return apiCall('/azure/symbols/categories');
}

/**
 * Search for symbols.
 */
export async function searchSymbols(
  query: string,
  language: string = 'en',
  limit: number = 10
): Promise<{ query: string; symbols: Symbol[]; count: number }> {
  return apiCall(`/azure/symbols/search/${encodeURIComponent(query)}?language=${language}&limit=${limit}`);
}

// =============================================================================
// SIGN LANGUAGE APIs
// =============================================================================

/**
 * Recognize sign language gestures from an image.
 */
export async function signToIntent(
  imageBase64: string,
  userId?: string
): Promise<SignToIntentResponse> {
  return apiCall('/azure/sign-to-intent', {
    method: 'POST',
    body: JSON.stringify({
      image_base64: imageBase64,
      user_id: userId,
    }),
  });
}

/**
 * Clear gesture session for continuous detection.
 */
export async function clearSignSession(
  sessionId: string
): Promise<{ message: string }> {
  return apiCall('/azure/sign-to-intent/clear-session', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });
}

/**
 * Process a video for gesture recognition.
 * Extracts frames from the video and processes all gestures to form a sentence.
 * @deprecated Use processSignVideo for better accuracy with I3D/Pose-LSTM models
 */
export async function processVideoGestures(
  videoBase64: string,
  userId?: string
): Promise<SignToIntentResponse> {
  return apiCall('/azure/sign-to-intent/video', {
    method: 'POST',
    body: JSON.stringify({
      video_base64: videoBase64,
      user_id: userId,
    }),
  }, 60000); // 60 second timeout for video processing
}

/**
 * Video sign language recognition response from I3D/Pose-LSTM models.
 */
export interface SignVideoRecognitionResponse {
  recognized_words: string[];
  sentence: string;
  confidence: number;
  frames_processed?: number;
  windows_processed?: number;
  models_used?: string[];
  raw_sentence?: string;
  error?: string;
}

/**
 * Process a video for sign language recognition using pretrained I3D/Pose-LSTM models.
 * This endpoint provides more accurate ASL word recognition compared to basic gesture detection.
 * 
 * @param videoBase64 - Base64 encoded video (MP4, WebM, etc.)
 * @param useGroqCorrection - Whether to use Groq LLM for sentence correction (default: true)
 * @param userId - Optional user ID for personalization
 */
export async function processSignVideo(
  videoBase64: string,
  useGroqCorrection: boolean = true,
  userId?: string
): Promise<SignVideoRecognitionResponse> {
  return apiCall('/sign-video/recognize', {
    method: 'POST',
    body: JSON.stringify({
      video_base64: videoBase64,
      use_groq_correction: useGroqCorrection,
      user_id: userId,
    }),
  }, 90000); // 90 second timeout for video processing with deep models
}

/**
 * Check sign language model status.
 */
export async function getSignVideoStatus(): Promise<{
  status: string;
  models: Record<string, any>;
  vocabulary_size: number;
}> {
  return apiCall('/sign-video/status', {}, 5000);
}

/**
 * Low-level gesture analysis.
 */
export async function analyzeGesture(
  imageBase64: string,
  includeLandmarks: boolean = false
): Promise<any> {
  const formData = new FormData();
  formData.append('image_base64', imageBase64);
  formData.append('include_landmarks', String(includeLandmarks));
  
  return apiCall('/azure/analyze-gesture', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      image_base64: imageBase64,
      include_landmarks: String(includeLandmarks),
    }),
  });
}

/**
 * Analyze facial expression from an image.
 */
export async function analyzeFace(
  imageBase64: string
): Promise<{ faces_detected: number; faces: any[]; error?: string }> {
  return apiCall('/azure/analyze-face', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      image_base64: imageBase64,
    }),
  });
}

// =============================================================================
// BEHAVIOR APIs
// =============================================================================

/**
 * Interpret behavioral patterns.
 */
export async function behaviorToIntent(
  behaviorData: BehaviorData,
  context?: string,
  userId?: string
): Promise<BehaviorToIntentResponse> {
  return apiCall('/azure/behavior-to-intent', {
    method: 'POST',
    body: JSON.stringify({
      behavior_data: behaviorData,
      context,
      user_id: userId,
    }),
  });
}

// =============================================================================
// TEXT GENERATION APIs
// =============================================================================

/**
 * Generate natural text from an intent.
 */
export async function generateText(
  intent: string,
  context?: string,
  style: 'natural' | 'formal' | 'simple' | 'friendly' = 'natural',
  userId?: string
): Promise<GenerateTextResponse> {
  return apiCall('/azure/generate-text', {
    method: 'POST',
    body: JSON.stringify({
      intent,
      context,
      style,
      user_id: userId,
    }),
  });
}

// =============================================================================
// FEEDBACK APIs
// =============================================================================

/**
 * Submit feedback for AI improvement.
 */
export async function submitFeedback(
  userId: string,
  sessionId: string,
  interactionId: string,
  feedbackType: FeedbackType,
  originalOutput: string,
  inputMode: InputMode,
  correctedOutput?: string,
  additionalContext?: any
): Promise<{ success: boolean; feedback_id: string; message: string }> {
  return apiCall('/azure/feedback', {
    method: 'POST',
    body: JSON.stringify({
      user_id: userId,
      session_id: sessionId,
      interaction_id: interactionId,
      feedback_type: feedbackType,
      original_output: originalOutput,
      corrected_output: correctedOutput,
      input_mode: inputMode,
      additional_context: additionalContext,
    }),
  });
}

/**
 * Get feedback statistics for a user.
 */
export async function getFeedbackStats(userId: string): Promise<any> {
  return apiCall(`/azure/feedback/stats/${userId}`);
}

// =============================================================================
// USER APIs
// =============================================================================

/**
 * Create a new user.
 */
export async function createUser(
  displayName?: string,
  communicationMode: InputMode = 'sign'
): Promise<UserProfile> {
  return apiCall(`/users/?display_name=${encodeURIComponent(displayName || '')}&communication_mode=${communicationMode}`, {
    method: 'POST',
  });
}

/**
 * Get user profile.
 */
export async function getUser(userId: string): Promise<UserProfile> {
  return apiCall(`/users/${userId}`);
}

/**
 * Update user profile.
 */
export async function updateUser(
  userId: string,
  update: Partial<{
    display_name: string;
    communication_mode: InputMode;
    preferences: any;
  }>
): Promise<UserProfile> {
  return apiCall(`/users/${userId}`, {
    method: 'PATCH',
    body: JSON.stringify(update),
  });
}

/**
 * Get user statistics.
 */
export async function getUserStats(userId: string): Promise<any> {
  return apiCall(`/users/${userId}/stats`);
}

// =============================================================================
// HEALTH APIs
// =============================================================================

/**
 * Check API health status.
 */
export async function checkHealth(): Promise<{
  status: string;
  version: string;
  services: Record<string, string>;
}> {
  return apiCall('/health');
}

/**
 * Get list of available endpoints.
 */
export async function getEndpoints(): Promise<{ endpoints: any }> {
  return apiCall('/api/endpoints');
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Convert a Blob to base64 string.
 */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = reader.result as string;
      // Remove the data URL prefix
      resolve(base64.split(',')[1]);
    };
    reader.onerror = () => reject(new Error('Failed to read blob'));
    reader.readAsDataURL(blob);
  });
}

/**
 * Convert a URI to base64 (for React Native).
 */
export async function uriToBase64(uri: string): Promise<string> {
  const response = await fetch(uri);
  const blob = await response.blob();
  return blobToBase64(blob);
}

// Default export with all APIs
const BridgeCommAPI = {
  // Speech
  speechToText,
  textToSpeech,
  
  // Symbols
  textToSymbols,
  getSymbolCategories,
  searchSymbols,
  
  // Sign Language
  signToIntent,
  processVideoGestures,
  processSignVideo,
  getSignVideoStatus,
  clearSignSession,
  analyzeGesture,
  analyzeFace,
  
  // Behavior
  behaviorToIntent,
  
  // Text Generation
  generateText,
  
  // Feedback
  submitFeedback,
  getFeedbackStats,
  
  // Users
  createUser,
  getUser,
  updateUser,
  getUserStats,
  
  // Health
  checkHealth,
  getEndpoints,
  
  // Utilities
  blobToBase64,
  uriToBase64,
};

export default BridgeCommAPI;

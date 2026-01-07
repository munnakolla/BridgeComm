import Ionicons from '@expo/vector-icons/Ionicons';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import React, { useEffect, useRef, useState } from 'react';
import {
    ActivityIndicator, Appearance, Dimensions,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View
} from 'react-native';

import AppText from '../components/AppText';
import Screen from '../components/Screen';
import { themes } from '../constants/theme';
import { EMOTION_MAPPINGS, useEmotionDetection } from '../hooks/useEmotionDetection';
import { useSignToText } from '../hooks/useSignToText';
import { useSettings } from './context/SettingsContext';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');
const CAMERA_HEIGHT = Math.min(SCREEN_HEIGHT * 0.35, 280);

// Touch Input Symbols with sign language mappings
const TOUCH_SYMBOLS = [
  { id: 'hello', emoji: '👋', name: 'Hello', signGesture: 'wave' },
  { id: 'yes', emoji: '✅', name: 'Yes', signGesture: 'nod' },
  { id: 'no', emoji: '❌', name: 'No', signGesture: 'shake_head' },
  { id: 'help', emoji: '🆘', name: 'Help', signGesture: 'raise_hand' },
  { id: 'thanks', emoji: '🙏', name: 'Thanks', signGesture: 'prayer_hands' },
  { id: 'please', emoji: '🤲', name: 'Please', signGesture: 'open_palms' },
  { id: 'happy', emoji: '😊', name: 'Happy', signGesture: 'smile_gesture' },
  { id: 'sad', emoji: '😢', name: 'Sad', signGesture: 'tear_gesture' },
  { id: 'hungry', emoji: '🍽️', name: 'Hungry', signGesture: 'eat_gesture' },
  { id: 'thirsty', emoji: '💧', name: 'Thirsty', signGesture: 'drink_gesture' },
  { id: 'tired', emoji: '😴', name: 'Tired', signGesture: 'sleep_gesture' },
  { id: 'love', emoji: '❤️', name: 'Love', signGesture: 'heart_gesture' },
  { id: 'home', emoji: '🏠', name: 'Home', signGesture: 'house_gesture' },
  { id: 'stop', emoji: '🛑', name: 'Stop', signGesture: 'stop_hand' },
  { id: 'more', emoji: '➕', name: 'More', signGesture: 'add_gesture' },
  { id: 'done', emoji: '✔️', name: 'Done', signGesture: 'finish_gesture' },
];

// Sign language gesture mappings with outputs
const SIGN_LANGUAGE_OUTPUTS: Record<string, { text: string; emoji: string }> = {
  wave: { text: 'Hello! Nice to meet you.', emoji: '👋' },
  thumbs_up: { text: 'Yes, I agree with that.', emoji: '👍' },
  thumbs_down: { text: 'No, I disagree.', emoji: '👎' },
  point: { text: 'Look over there, please.', emoji: '👉' },
  open_palm: { text: 'Please help me.', emoji: '🤲' },
  fist: { text: 'I am strong. I can do this.', emoji: '✊' },
  peace: { text: 'Peace and calm.', emoji: '✌️' },
  ok_sign: { text: 'Okay, that sounds good.', emoji: '👌' },
  clap: { text: 'Great job! Well done!', emoji: '👏' },
  pray: { text: 'Thank you so much.', emoji: '🙏' },
  raise_hand: { text: 'I have something to say.', emoji: '✋' },
  wave_goodbye: { text: 'Goodbye, see you later!', emoji: '🖐️' },
};

export default function Communicate() {
  const [mode, setMode] = useState<'sign' | 'eye' | 'touch'>('sign');
  const [cameraOn, setCameraOn] = useState(false);
  const [facing] = useState<'front' | 'back'>('front');
  const [selectedSymbols, setSelectedSymbols] = useState<typeof TOUCH_SYMBOLS>([]);
  const [demoText, setDemoText] = useState('');
  const [demoEmoji, setDemoEmoji] = useState('');
  const [detectedGestures, setDetectedGestures] = useState<string[]>([]);
  const [signError, setSignError] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const signToText = useSignToText();
  const emotionDetection = useEmotionDetection();
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  const subtleText = resolvedTheme === 'dark' ? '#cbd5e1' : '#6b7280';

  const isProcessing = signToText.isProcessing || emotionDetection.isProcessing;
  const isRecording = signToText.isRecording;
  
  // Watch for signToText updates after video processing completes
  useEffect(() => {
    if (!signToText.isProcessing && !signToText.isRecording) {
      if (signToText.text) {
        setDetectedGestures(signToText.gestures);
        setDemoText('');
        setSignError(null);
      } else if (signToText.error) {
        setSignError(signToText.error);
        setDemoText('');
      }
    }
  }, [signToText.isProcessing, signToText.isRecording, signToText.text, signToText.error, signToText.gestures]);
  
  // Build translated text based on mode
  const getTranslatedText = () => {
    if (mode === 'touch') {
      return selectedSymbols.map(s => s.name).join(' ');
    } else if (mode === 'eye') {
      return emotionDetection.sentence || demoText;
    } else {
      return signToText.text || demoText;
    }
  };
  
  const translatedText = getTranslatedText();
  
  // Get display symbols for output - ONLY for touch mode
  const getOutputSymbols = () => {
    if (mode === 'touch') {
      return selectedSymbols.map(s => ({ emoji: s.emoji, name: s.name }));
    }
    // Eye tracking and sign language modes don't show symbols
    return [];
  };

  // Handle camera ready callback
  const handleCameraReady = () => {
    console.log('Camera is ready');
    signToText.setCameraReady(true);
  };

  // Reset camera ready when camera is turned off
  useEffect(() => {
    if (!cameraOn) {
      signToText.setCameraReady(false);
    }
  }, [cameraOn]);

  if (!permission) {
    return <Screen><View /></Screen>;
  }

  if (!permission.granted) {
    return (
      <Screen>
        <View style={styles.permissionBox}>
          <Ionicons name="camera-outline" size={48} color="#6b7280" />
          <AppText style={styles.permissionText}>Camera permission is required</AppText>
          <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
            <AppText style={styles.primaryBtnText}>Grant Permission</AppText>
          </TouchableOpacity>
        </View>
      </Screen>
    );
  }

  // Analyze sign language gesture (single frame capture)
  const analyzeSignLanguage = async () => {
    setDemoText('');
    setDemoEmoji('');
    setDetectedGestures([]);
    
    if (!cameraRef.current) {
      setDemoText('Please start camera first');
      return;
    }
    
    try {
      // Use actual API for sign language recognition
      await signToText.captureAndRecognize(cameraRef as React.RefObject<CameraView>);
      
      // If API worked and returned text, show detected gestures
      if (signToText.text) {
        setDetectedGestures(signToText.gestures);
      } else if (signToText.error) {
        setDemoText('Could not detect gesture. Please try again.');
      }
    } catch (error) {
      console.log('Sign recognition error:', error);
      setDemoText('Recognition failed. Please try again.');
    }
  };

  // Toggle video recording for continuous gesture capture
  const toggleVideoRecording = async () => {
    if (!cameraRef.current) {
      setSignError('Please start camera first');
      return;
    }

    // Check if camera is ready before starting recording
    if (!isRecording && !signToText.isCameraReady) {
      setSignError('Camera is still initializing. Please wait a moment...');
      return;
    }

    if (isRecording) {
      // Stop recording and process video
      setDemoText('Processing video...');
      setSignError(null);
      await signToText.stopVideoRecording();
      // State updates will be handled by the useEffect above
    } else {
      // Start recording
      setDemoText('Recording... Show your gestures, then tap again to stop.');
      setDetectedGestures([]);
      setSignError(null);
      try {
        await signToText.startVideoRecording(cameraRef as React.RefObject<CameraView>);
      } catch (error) {
        console.error('Recording start error:', error);
        setSignError('Failed to start recording. Please try again.');
        setDemoText('');
      }
    }
  };

  // Analyze emotion from facial expression (Eye Tracking mode)
  const analyzeEmotion = async () => {
    setDemoText('');
    setDemoEmoji('');
    
    if (!cameraRef.current) {
      // Use emotion detection with demo fallback
      await emotionDetection.detectEmotion(cameraRef as React.RefObject<CameraView>);
      return;
    }
    
    await emotionDetection.detectEmotion(cameraRef as React.RefObject<CameraView>);
  };

  // Main analyze function based on mode
  const analyzeGesture = async () => {
    if (mode === 'eye') {
      await analyzeEmotion();
    } else {
      await analyzeSignLanguage();
    }
  };

  const speakOutput = () => {
    if (translatedText) {
      Speech.speak(translatedText, { language: 'en' });
    }
  };

  const handleSymbolPress = (symbol: typeof TOUCH_SYMBOLS[0]) => {
    setSelectedSymbols(prev => [...prev, symbol]);
  };

  const clearAll = () => {
    setSelectedSymbols([]);
    setDemoText('');
    setDemoEmoji('');
    setDetectedGestures([]);
    setSignError(null);
    emotionDetection.clear();
    signToText.clear && signToText.clear();
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <AppText style={[styles.title, { color: activeTheme.text }]}>Communicate Your Message</AppText>
        <AppText style={[styles.subtitle, { color: subtleText }]}>
          Use sign language, eye tracking, or touch to express yourself. We'll translate it to natural speech.
        </AppText>

        {/* Input Mode Tabs */}
        <View style={[styles.tabContainer, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
          <AppText style={[styles.tabLabel, { color: subtleText }]}>Input Mode</AppText>
          <View style={styles.tabRow}>
            <TouchableOpacity style={[styles.tab, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }, mode === 'sign' && styles.tabActive]} onPress={() => setMode('sign')}>
              <Ionicons name="hand-left" size={18} color={mode === 'sign' ? activeTheme.primary : subtleText} />
              <AppText style={[styles.tabText, { color: subtleText }, mode === 'sign' && { color: activeTheme.primary, fontWeight: '600' }]}>Sign Language</AppText>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.tab, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }, mode === 'eye' && styles.tabActive]} onPress={() => setMode('eye')}>
              <Ionicons name="eye" size={18} color={mode === 'eye' ? activeTheme.primary : subtleText} />
              <AppText style={[styles.tabText, { color: subtleText }, mode === 'eye' && { color: activeTheme.primary, fontWeight: '600' }]}>Eye Tracking</AppText>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.tab, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }, mode === 'touch' && styles.tabActive]} onPress={() => setMode('touch')}>
              <Ionicons name="finger-print" size={18} color={mode === 'touch' ? activeTheme.primary : subtleText} />
              <AppText style={[styles.tabText, { color: subtleText }, mode === 'touch' && { color: activeTheme.primary, fontWeight: '600' }]}>Touch Input</AppText>
            </TouchableOpacity>
          </View>
        </View>

        {/* Main Content */}
        <View style={styles.mainContent}>
          {/* Input Section */}
          <View style={styles.inputSection}>
            <View style={styles.inputHeader}>
              <AppText style={styles.sectionTitle}>
                {mode === 'touch' ? 'Touch Input' : mode === 'eye' ? 'Emotion Detection' : 'Sign Language Input'}
              </AppText>
              <View style={[styles.statusDot, (cameraOn || mode === 'touch') && styles.statusDotActive]} />
            </View>
            
            {mode === 'touch' ? (
              /* Touch Input - Symbol Grid */
              <View style={[styles.touchGrid, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
                {TOUCH_SYMBOLS.map(symbol => (
                  <TouchableOpacity key={symbol.id} style={[styles.touchSymbol, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]} onPress={() => handleSymbolPress(symbol)}>
                    <Text style={styles.touchEmoji}>{symbol.emoji}</Text>
                    <AppText style={styles.touchName}>{symbol.name}</AppText>
                  </TouchableOpacity>
                ))}
              </View>
            ) : mode === 'eye' ? (
              /* Eye Tracking - Emotion Detection */
              <>
                <View style={styles.cameraBox}>
                  {cameraOn ? (
                    <CameraView 
                      ref={cameraRef} 
                      style={styles.camera} 
                      facing={facing}
                      onCameraReady={handleCameraReady}
                    />
                  ) : (
                    <View style={styles.cameraPlaceholder}>
                      <Ionicons name="happy-outline" size={40} color="#9ca3af" />
                      <AppText style={styles.cameraText}>Camera will detect your facial expression</AppText>
                    </View>
                  )}
                  {emotionDetection.detectedEmotion && cameraOn && (
                    <View style={styles.emotionOverlay}>
                      <Text style={styles.emotionOverlayEmoji}>
                        {EMOTION_MAPPINGS[emotionDetection.detectedEmotion.toLowerCase()]?.emoji || '😐'}
                      </Text>
                      <AppText style={styles.emotionOverlayText}>{emotionDetection.detectedEmotion}</AppText>
                    </View>
                  )}
                </View>
                
                <TouchableOpacity
                  style={[styles.startCameraBtn, { backgroundColor: activeTheme.primary }]}
                  onPress={() => cameraOn ? analyzeGesture() : setCameraOn(true)}
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <>
                      <Ionicons name={cameraOn ? "scan" : "camera"} size={18} color="#fff" />
                      <AppText style={styles.startCameraText}>
                        {cameraOn ? ' Detect Emotion' : ' Start Camera'}
                      </AppText>
                    </>
                  )}
                </TouchableOpacity>
              </>
            ) : (
              /* Sign Language Input */
              <>
                <View style={styles.cameraBox}>
                  {cameraOn ? (
                    <CameraView 
                      ref={cameraRef} 
                      style={styles.camera} 
                      facing={facing}
                      onCameraReady={handleCameraReady}
                    />
                  ) : (
                    <View style={styles.cameraPlaceholder}>
                      <Ionicons name="hand-left" size={40} color="#9ca3af" />
                      <AppText style={styles.cameraText}>Camera will recognize your sign gestures</AppText>
                    </View>
                  )}
                  {isRecording && cameraOn && (
                    <View style={[styles.gestureOverlay, { backgroundColor: 'rgba(239, 68, 68, 0.8)' }]}>
                      <Ionicons name="radio-button-on" size={16} color="#fff" />
                      <AppText style={[styles.gestureOverlayText, { marginLeft: 6 }]}>
                        Recording...
                      </AppText>
                    </View>
                  )}
                  {detectedGestures.length > 0 && cameraOn && !isRecording && (
                    <View style={styles.gestureOverlay}>
                      <AppText style={styles.gestureOverlayText}>
                        Detected: {detectedGestures.join(', ')}
                      </AppText>
                    </View>
                  )}
                </View>
                
                {!cameraOn ? (
                  <TouchableOpacity
                    style={[styles.startCameraBtn, { backgroundColor: activeTheme.primary }]}
                    onPress={() => setCameraOn(true)}
                  >
                    <Ionicons name="camera" size={18} color="#fff" />
                    <AppText style={styles.startCameraText}> Start Camera</AppText>
                  </TouchableOpacity>
                ) : (
                  <View style={styles.signButtonRow}>
                    {/* Camera Ready Indicator */}
                    {!signToText.isCameraReady && (
                      <View style={styles.cameraInitializing}>
                        <ActivityIndicator size="small" color={activeTheme.primary} />
                        <AppText style={[styles.initializingText, { color: subtleText }]}>
                          Camera initializing...
                        </AppText>
                      </View>
                    )}
                    
                    <View style={styles.signButtonsContainer}>
                      {/* Video Recording Button - Primary action */}
                      <TouchableOpacity
                        style={[
                          styles.signActionBtn, 
                          { 
                            backgroundColor: isRecording ? '#ef4444' : (signToText.isCameraReady ? activeTheme.primary : subtleText),
                            flex: 2,
                            opacity: signToText.isCameraReady || isRecording ? 1 : 0.6
                          }
                        ]}
                        onPress={toggleVideoRecording}
                        disabled={isProcessing || (!signToText.isCameraReady && !isRecording)}
                      >
                        {isProcessing ? (
                          <ActivityIndicator size="small" color="#fff" />
                        ) : (
                          <>
                            <Ionicons 
                              name={isRecording ? "stop-circle" : "videocam"} 
                              size={18} 
                              color="#fff" 
                            />
                            <AppText style={styles.signActionText}>
                              {isRecording ? ' Stop & Process' : (signToText.isCameraReady ? ' Record Gestures' : ' Wait...')}
                            </AppText>
                          </>
                        )}
                      </TouchableOpacity>
                      
                      {/* Single Capture Button - Secondary action */}
                      <TouchableOpacity
                        style={[
                          styles.signActionBtn, 
                          { 
                            backgroundColor: subtleText, 
                            flex: 1,
                            opacity: signToText.isCameraReady ? 1 : 0.6
                          }
                        ]}
                        onPress={analyzeSignLanguage}
                        disabled={isProcessing || isRecording || !signToText.isCameraReady}
                      >
                        <Ionicons name="scan" size={18} color="#fff" />
                        <AppText style={styles.signActionText}> Capture</AppText>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
                
                {/* Error display for sign language */}
                {signError && mode === 'sign' && (
                  <View style={styles.errorBox}>
                    <Ionicons name="alert-circle" size={16} color="#ef4444" />
                    <AppText style={styles.errorText}>{signError}</AppText>
                  </View>
                )}
                
                {/* Status message */}
                {demoText && mode === 'sign' && !signError && (
                  <View style={styles.statusBox}>
                    <ActivityIndicator size="small" color="#2563eb" />
                    <AppText style={styles.statusText}>{demoText}</AppText>
                  </View>
                )}
              </>
            )}
          </View>

          {/* Output Section */}
          <View style={styles.outputSection}>
            <AppText style={styles.sectionTitle}>Your Message</AppText>
            <View style={[styles.outputBox, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
              {translatedText ? (
                <View style={styles.translatedContent}>
                  {/* Show symbols for touch mode only */}
                  {mode === 'touch' && getOutputSymbols().length > 0 && (
                    <View style={styles.symbolsRow}>
                      {getOutputSymbols().map((s, i) => (
                        <Text key={i} style={styles.outputEmoji}>{s.emoji}</Text>
                      ))}
                    </View>
                  )}
                  
                  {/* Show sign language gestures if detected */}
                  {mode === 'sign' && detectedGestures.length > 0 && (
                    <View style={styles.gesturesInfo}>
                      <AppText style={styles.gesturesLabel}>Recognized Signs:</AppText>
                      <AppText style={styles.gesturesText}>{detectedGestures.join(' → ')}</AppText>
                    </View>
                  )}
                  
                  {/* The main translated text - shown for all modes */}
                  <AppText style={[styles.translatedText, { color: activeTheme.text }]}>{translatedText}</AppText>
                  
                  {/* Sentence variation button for eye tracking */}
                  {mode === 'eye' && emotionDetection.sentence && (
                    <TouchableOpacity style={styles.variationBtn} onPress={() => emotionDetection.getSentenceVariation()}>
                      <Ionicons name="refresh" size={14} color="#2563eb" />
                      <AppText style={styles.variationText}> Different phrase</AppText>
                    </TouchableOpacity>
                  )}
                  
                  <View style={styles.outputActions}>
                    <TouchableOpacity style={styles.speakBtn} onPress={speakOutput}>
                      <Ionicons name="volume-high" size={20} color="#fff" />
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.clearBtn} onPress={clearAll}>
                      <Ionicons name="trash-outline" size={20} color="#ef4444" />
                    </TouchableOpacity>
                  </View>
                </View>
              ) : (
                <View style={styles.outputPlaceholder}>
                  <Ionicons name="paper-plane-outline" size={36} color={subtleText} />
                  <AppText style={[styles.outputPlaceholderText, { color: subtleText }]}>Your translated message will appear here</AppText>
                </View>
              )}
            </View>
          </View>
        </View>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scrollContent: { padding: 20, paddingBottom: 50 },
  title: { fontSize: 26, fontWeight: 'bold', marginBottom: 6 },
  subtitle: { color: '#6b7280', marginBottom: 20, lineHeight: 22 },

  tabContainer: { marginBottom: 20, backgroundColor: '#fff', borderRadius: 16, padding: 16, borderWidth: 1, borderColor: '#e5e7eb' },
  tabLabel: { fontSize: 13, color: '#6b7280', marginBottom: 12 },
  tabRow: { flexDirection: 'row', gap: 10 },
  tab: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingVertical: 12, borderRadius: 12, borderWidth: 1, borderColor: '#e5e7eb', backgroundColor: '#fff', gap: 6,
  },
  tabActive: { backgroundColor: '#fef3c7', borderColor: '#fbbf24' },
  tabText: { fontSize: 12, color: '#6b7280', fontWeight: '500' },
  tabTextActive: { color: '#92400e', fontWeight: '600' },

  infoBox: {
    flexDirection: 'row',
    backgroundColor: '#eff6ff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    alignItems: 'flex-start',
    gap: 10,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#1e40af',
    lineHeight: 20,
  },

  mainContent: { flexDirection: 'row', gap: 16, marginBottom: 20 },
  inputSection: { flex: 1.2 },
  inputHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  sectionTitle: { fontSize: 16, fontWeight: '600', color: '#111827' },
  statusDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#d1d5db' },
  statusDotActive: { backgroundColor: '#22c55e' },

  cameraBox: {
    height: CAMERA_HEIGHT, backgroundColor: '#111827', borderRadius: 16,
    overflow: 'hidden', marginBottom: 14, justifyContent: 'center', alignItems: 'center',
  },
  camera: { width: '100%', height: '100%' },
  cameraPlaceholder: { alignItems: 'center' },
  cameraText: { color: '#9ca3af', fontSize: 13, marginTop: 10 },
  startCameraBtn: {
    backgroundColor: '#2563eb', paddingVertical: 14, borderRadius: 14,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
  },
  startCameraText: { color: '#fff', fontWeight: '600', fontSize: 15 },

  touchGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, backgroundColor: '#fff', borderRadius: 16, padding: 12, borderWidth: 1, borderColor: '#e5e7eb' },
  touchSymbol: {
    width: '23%', aspectRatio: 1, backgroundColor: '#fff', borderRadius: 12,
    alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#e5e7eb',
  },
  touchEmoji: { fontSize: 28, marginBottom: 4 },
  touchName: { fontSize: 10, color: '#374151', fontWeight: '500' },

  outputSection: { flex: 1 },
  outputBox: {
    flex: 1, backgroundColor: '#fafafa', borderRadius: 16, padding: 20,
    borderWidth: 1, borderColor: '#e5e7eb', minHeight: 300, justifyContent: 'center',
  },
  outputPlaceholder: { alignItems: 'center' },
  outputPlaceholderText: { color: '#9ca3af', fontSize: 13, marginTop: 12, textAlign: 'center' },
  translatedContent: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  symbolsRow: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 8, marginBottom: 16 },
  outputEmoji: { fontSize: 32 },
  translatedText: { fontSize: 18, color: '#111827', lineHeight: 28, textAlign: 'center' },
  outputActions: { flexDirection: 'row', gap: 12, marginTop: 20 },
  speakBtn: { backgroundColor: '#22c55e', padding: 14, borderRadius: 25 },
  clearBtn: { backgroundColor: '#fee2e2', padding: 14, borderRadius: 25 },

  // Emotion Detection Styles
  emotionOverlay: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 12,
    padding: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  emotionOverlayEmoji: { fontSize: 24 },
  emotionOverlayText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  
  // Emotion display in output section
  emotionDisplay: {
    alignItems: 'center',
    marginBottom: 16,
  },
  emotionLargeEmoji: {
    fontSize: 64,
    marginBottom: 8,
  },
  emotionLabel: {
    fontSize: 18,
    fontWeight: '600',
    color: '#374151',
    textTransform: 'capitalize',
  },
  
  quickSelectLabel: { fontSize: 13, color: '#6b7280', marginBottom: 10, marginTop: 10 },
  emotionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 14,
  },
  emotionQuickBtn: {
    width: '23%',
    aspectRatio: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  emotionQuickBtnActive: {
    backgroundColor: '#fef3c7',
    borderColor: '#fbbf24',
  },
  emotionQuickEmoji: { fontSize: 24, marginBottom: 2 },
  emotionQuickLabel: { fontSize: 9, color: '#374151', fontWeight: '500' },

  // Sign Language Gesture Styles
  gestureOverlay: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 12,
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  gestureOverlayText: { color: '#fff', fontSize: 12, textAlign: 'center' },
  
  // Sign Language Button Row
  signButtonRow: {
    flexDirection: 'column',
    gap: 10,
    marginTop: 12,
  },
  cameraInitializing: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    gap: 8,
  },
  initializingText: {
    fontSize: 13,
    fontWeight: '500',
  },
  signActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 14,
    gap: 6,
  },
  signActionText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  signButtonsContainer: {
    flexDirection: 'row',
    gap: 10,
  },
  
  gesturesInfo: {
    backgroundColor: '#f0fdf4',
    borderRadius: 10,
    padding: 10,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#bbf7d0',
  },
  gesturesLabel: { fontSize: 11, color: '#166534', fontWeight: '600', marginBottom: 4 },
  gesturesText: { fontSize: 13, color: '#15803d' },

  // Confidence Bar
  confidenceBar: {
    width: '100%',
    height: 24,
    backgroundColor: '#e5e7eb',
    borderRadius: 12,
    marginBottom: 12,
    overflow: 'hidden',
    position: 'relative',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: '#22c55e',
    borderRadius: 12,
  },
  confidenceText: {
    position: 'absolute',
    width: '100%',
    textAlign: 'center',
    lineHeight: 24,
    fontSize: 11,
    color: '#374151',
    fontWeight: '600',
  },

  // Variation Button
  variationBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    padding: 8,
    backgroundColor: '#eff6ff',
    borderRadius: 8,
  },
  variationText: { fontSize: 12, color: '#2563eb', fontWeight: '500' },

  // Error and Status Messages
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    padding: 12,
    backgroundColor: '#fef2f2',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#fecaca',
    gap: 8,
  },
  errorText: { flex: 1, fontSize: 13, color: '#dc2626' },
  statusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    padding: 12,
    backgroundColor: '#eff6ff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#bfdbfe',
    gap: 10,
  },
  statusText: { flex: 1, fontSize: 13, color: '#2563eb' },

  permissionBox: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  permissionText: { marginVertical: 12 },
  primaryBtn: { backgroundColor: '#2563eb', paddingVertical: 14, paddingHorizontal: 28, borderRadius: 14 },
  primaryBtnText: { color: '#fff', fontWeight: '600' },
});

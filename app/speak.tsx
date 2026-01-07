import Ionicons from '@expo/vector-icons/Ionicons';
import * as Speech from 'expo-speech';
import { useEffect, useState } from 'react';
import {
    ActivityIndicator, Appearance, ScrollView,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View
} from 'react-native';

import AppText from '../components/AppText';
import Screen from '../components/Screen';
import { themes } from '../constants/theme';
import { useSpeechToSymbols } from '../hooks/useSpeechToSymbols';
import { useSettings } from './context/SettingsContext';

// Demo symbols with emojis (fallback when API is unavailable)
const DEMO_SYMBOLS: Record<string, { emoji: string; name: string }> = {
  hello: { emoji: '👋', name: 'Hello' },
  hi: { emoji: '👋', name: 'Hello' },
  goodbye: { emoji: '🖐️', name: 'Goodbye' },
  bye: { emoji: '🖐️', name: 'Goodbye' },
  thanks: { emoji: '🙏', name: 'Thanks' },
  thank: { emoji: '🙏', name: 'Thanks' },
  please: { emoji: '🤲', name: 'Please' },
  help: { emoji: '🆘', name: 'Help' },
  yes: { emoji: '✅', name: 'Yes' },
  no: { emoji: '❌', name: 'No' },
  happy: { emoji: '😊', name: 'Happy' },
  sad: { emoji: '😢', name: 'Sad' },
  hungry: { emoji: '🍽️', name: 'Hungry' },
  thirsty: { emoji: '💧', name: 'Thirsty' },
  tired: { emoji: '😴', name: 'Tired' },
  love: { emoji: '❤️', name: 'Love' },
  home: { emoji: '🏠', name: 'Home' },
  water: { emoji: '💦', name: 'Water' },
  food: { emoji: '🍔', name: 'Food' },
  i: { emoji: '👤', name: 'I' },
  you: { emoji: '👉', name: 'You' },
  want: { emoji: '🙋', name: 'Want' },
  need: { emoji: '🙏', name: 'Need' },
  go: { emoji: '🚶', name: 'Go' },
  stop: { emoji: '🛑', name: 'Stop' },
};

export default function Speak() {
  const [message, setMessage] = useState('');
  const [localSimplified, setLocalSimplified] = useState('');
  const [localSymbols, setLocalSymbols] = useState<{ emoji: string; name: string }[]>([]);
  const [isConverting, setIsConverting] = useState(false);
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  const subtleText = resolvedTheme === 'dark' ? '#cbd5e1' : '#6b7280';
  
  // API hook for speech-to-text and text-to-symbols
  const speechToSymbols = useSpeechToSymbols();

  // Sync recognized text from voice recording to message input
  useEffect(() => {
    if (speechToSymbols.recognizedText) {
      setMessage(speechToSymbols.recognizedText);
      // Also sync the simplified text and symbols from the API result
      if (speechToSymbols.simplifiedText) {
        setLocalSimplified(speechToSymbols.simplifiedText);
      }
      if (speechToSymbols.symbols.length > 0) {
        setLocalSymbols(speechToSymbols.symbols.map(s => ({ emoji: '📷', name: s.name })));
      }
    }
  }, [speechToSymbols.recognizedText, speechToSymbols.simplifiedText, speechToSymbols.symbols]);

  // Handle mic button - real recording only
  const handleMicPress = async () => {
    if (speechToSymbols.isRecording) {
      // Stop recording and process
      try {
        await speechToSymbols.stopRecording();
      } catch (e) {
        console.log('Error stopping recording:', e);
      }
    } else {
      // Start recording
      try {
        await speechToSymbols.startRecording();
      } catch (error) {
        console.log('Recording error:', error);
        // Show error to user instead of using fake demo
        setMessage('');
      }
    }
  };

  const isRecording = speechToSymbols.isRecording;

  // Local conversion function (works without backend)
  const convertLocally = (text: string) => {
    const words = text.toLowerCase().split(/\s+/);
    const symbols: { emoji: string; name: string }[] = [];
    
    words.forEach(word => {
      const cleanWord = word.replace(/[^a-z]/g, '');
      if (DEMO_SYMBOLS[cleanWord]) {
        symbols.push(DEMO_SYMBOLS[cleanWord]);
      }
    });
    
    // Simplified text - just clean up the original
    const simplified = text.replace(/[^\w\s]/g, '').trim();
    
    return { simplified, symbols };
  };

  const handleConvert = async () => {
    if (!message.trim()) return;
    
    setIsConverting(true);
    setLocalSimplified('');
    setLocalSymbols([]);
    
    // Always do local conversion first for immediate feedback
    const { simplified, symbols } = convertLocally(message);
    setLocalSimplified(simplified || message);
    setLocalSymbols(symbols);
    
    try {
      // Then try API for better results
      await speechToSymbols.processText(message.trim());
      // If API succeeds, the speechToSymbols state will update and override local
    } catch (error) {
      console.log('API not available, using local conversion');
    }
    
    setIsConverting(false);
  };

  const handleClear = () => {
    setMessage('');
    setLocalSimplified('');
    setLocalSymbols([]);
    speechToSymbols.clear();
  };

  const displaySimplified = speechToSymbols.simplifiedText || localSimplified;
  const displaySymbols = speechToSymbols.symbols.length > 0 
    ? speechToSymbols.symbols.map(s => ({ emoji: '📷', name: s.name }))
    : localSymbols;

  const speakSimplified = () => {
    if (displaySimplified) {
      Speech.speak(displaySimplified, { language: 'en', rate: 0.9 });
    }
  };

  return (
    <Screen>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <AppText style={[styles.title, { color: activeTheme.text }]}>Speak to User</AppText>
        <AppText style={[styles.subtitle, { color: subtleText }]}>
          Speak naturally or type your message. We’ll convert it into simplified
          text and visual symbols for easy understanding.
        </AppText>

        {/* INPUT CARD */}
        <View style={[styles.card, { backgroundColor: activeTheme.card, borderColor: activeTheme.border, borderWidth: 1 }]}>
          <View style={styles.cardHeader}>
            <AppText style={styles.cardTitle}>Your Message</AppText>
            <TouchableOpacity 
              style={[
                styles.micButton, 
                { backgroundColor: activeTheme.primary },
                isRecording && styles.micButtonRecording
              ]}
              onPress={handleMicPress}
            >
              <Ionicons 
                name={isRecording ? "stop" : "mic"} 
                size={18} 
                color="#fff" 
              />
            </TouchableOpacity>
          </View>

          {isRecording && (
            <View style={styles.recordingIndicator}>
              <View style={styles.recordingDot} />
              <AppText style={styles.recordingText}>Recording...</AppText>
            </View>
          )}

          <TextInput
            style={[styles.input, { borderColor: activeTheme.border, backgroundColor: activeTheme.bg, color: activeTheme.text }]}
            placeholder="Speak or type your message here…"
            placeholderTextColor={subtleText}
            multiline
            value={message}
            onChangeText={setMessage}
          />

          <View style={styles.actionRow}>
            <TouchableOpacity
              style={[styles.convertButton, { backgroundColor: activeTheme.primary }, speechToSymbols.isProcessing && styles.disabledButton]}
              onPress={handleConvert}
              disabled={speechToSymbols.isProcessing}
            >
              {speechToSymbols.isProcessing ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <>
                  <Ionicons name="send" size={16} color="#fff" />
                  <AppText style={styles.convertText}> Convert</AppText>
                </>
              )}
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.clearButton}
              onPress={handleClear}
            >
              <Ionicons name="trash-outline" size={16} color={subtleText} />
              <AppText style={[styles.clearText, { color: activeTheme.text }]}> Clear</AppText>
            </TouchableOpacity>
          </View>

          {speechToSymbols.error && (
            <View style={styles.errorBox}>
              <Ionicons name="alert-circle" size={16} color="#ef4444" />
              <AppText style={styles.errorText}>{speechToSymbols.error}</AppText>
            </View>
          )}
        </View>

        {/* OUTPUT CARD */}
        <View style={[styles.card, { backgroundColor: activeTheme.card, borderColor: activeTheme.border, borderWidth: 1 }]}>
          <View style={styles.cardHeader}>
            <AppText style={styles.cardTitle}>Simplified Output</AppText>
            {displaySimplified && (
              <TouchableOpacity onPress={speakSimplified}>
                <Ionicons name="volume-high" size={20} color="#22c55e" />
              </TouchableOpacity>
            )}
          </View>

          <View style={[styles.outputBox, { borderColor: activeTheme.border, backgroundColor: activeTheme.bg }]}>
            <AppText style={[
              styles.outputText,
              displaySimplified && styles.outputTextActive
            ]}>
              {displaySimplified || 'Your simplified message will appear here'}
            </AppText>
          </View>

          {/* Visual Symbols */}
          <AppText style={styles.symbolTitle}>Visual Symbols</AppText>

          <View style={[styles.symbolBox, { borderColor: activeTheme.border, backgroundColor: activeTheme.bg }]}>
            {displaySymbols.length > 0 ? (
              <ScrollView 
                horizontal 
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.symbolsContainer}
              >
                {displaySymbols.map((symbol, index) => (
                  <View key={index} style={styles.symbolItem}>
                    <Text style={styles.symbolEmoji}>{symbol.emoji}</Text>
                    <AppText style={styles.symbolLabel}>{symbol.name}</AppText>
                  </View>
                ))}
              </ScrollView>
            ) : (
              <>
                <Ionicons name="image-outline" size={34} color="#9ca3af" />
                <AppText style={styles.symbolPlaceholder}>
                  Symbols will appear here
                </AppText>
              </>
            )}
          </View>
        </View>
      </ScrollView>
    </Screen>
  );
}

/* ================= STYLES ================= */

const styles = StyleSheet.create({
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },

  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginBottom: 4,
  },

  subtitle: {
    marginBottom: 20,
    lineHeight: 20,
  },

  card: {
    borderRadius: 20,
    padding: 16,
    marginBottom: 18,
  },

  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },

  cardTitle: {
    fontWeight: 'bold',
    fontSize: 16,
  },

  micButton: {
    backgroundColor: '#2563eb',
    padding: 8,
    borderRadius: 20,
  },

  micButtonRecording: {
    backgroundColor: '#ef4444',
  },

  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    padding: 8,
    backgroundColor: '#fef2f2',
    borderRadius: 8,
  },

  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ef4444',
    marginRight: 8,
  },

  recordingText: {
    color: '#ef4444',
    fontSize: 12,
    fontWeight: '600',
  },

  input: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 14,
    padding: 14,
    minHeight: 110,
    textAlignVertical: 'top',
    marginBottom: 14,
    fontSize: 14,
  },

  actionRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  convertButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2563eb',
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 14,
    marginRight: 14,
  },

  disabledButton: {
    opacity: 0.7,
  },

  convertText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },

  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  clearText: {
    fontWeight: '500',
    marginLeft: 4,
  },

  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    padding: 10,
    backgroundColor: '#fef2f2',
    borderRadius: 10,
  },

  errorText: {
    color: '#ef4444',
    fontSize: 13,
    marginLeft: 8,
    flex: 1,
  },

  outputBox: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 14,
    padding: 14,
    minHeight: 90,
    justifyContent: 'center',
    marginBottom: 12,
  },

  outputText: {
    fontSize: 14,
    color: '#9ca3af',
  },

  outputTextActive: {
    color: '#111827',
    fontSize: 16,
    lineHeight: 24,
  },

  symbolTitle: {
    fontSize: 12,
    marginBottom: 6,
    fontWeight: '600',
    color: '#6b7280',
  },

  symbolBox: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 14,
    paddingVertical: 20,
    alignItems: 'center',
    minHeight: 120,
    justifyContent: 'center',
  },

  symbolPlaceholder: {
    fontSize: 12,
    marginTop: 6,
    color: '#9ca3af',
  },

  symbolsContainer: {
    paddingHorizontal: 10,
    gap: 12,
  },

  symbolItem: {
    alignItems: 'center',
    width: 80,
  },

  symbolEmoji: {
    fontSize: 48,
    marginBottom: 4,
  },

  symbolImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
  },

  symbolLabel: {
    fontSize: 11,
    marginTop: 4,
    color: '#374151',
    textAlign: 'center',
  },
});

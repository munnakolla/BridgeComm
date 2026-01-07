import Ionicons from '@expo/vector-icons/Ionicons';
import { useRouter } from 'expo-router';
import {
    Appearance,
    ScrollView,
    StyleSheet,
    TouchableOpacity,
    View,
} from 'react-native';

import AppText from '../components/AppText';
import Screen from '../components/Screen';
import { themes } from '../constants/theme';
import { useSettings } from './context/SettingsContext';

export default function Home() {
  const router = useRouter();
  
  // Theme support
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  const subtleText = resolvedTheme === 'dark' ? '#cbd5e1' : '#6b7280';

  return (
    <Screen>
      <ScrollView
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.headerSection}>
          <AppText style={[styles.welcome, { color: activeTheme.text }]}>Welcome to</AppText>
          <AppText style={[styles.title, { color: activeTheme.primary }]}>BridgeComm</AppText>
          <AppText style={[styles.subtitle, { color: subtleText }]}>
            Breaking communication barriers with AI-powered bi-directional
            translation between speech and non-verbal communication.
          </AppText>
        </View>

        {/* Cards Row */}
        <View style={styles.cardRow}>
          {/* Speak */}
          <TouchableOpacity
            style={[styles.card, styles.blueCard]}
            onPress={() => router.push('/speak')}
          >
            <View style={styles.cardIconContainer}>
              <Ionicons name="mic" size={24} color="#fff" />
            </View>
            <AppText style={styles.cardTitle}>Speak to User</AppText>
            <AppText style={styles.cardSub}>
              Normal → Disabled Person
            </AppText>
            <AppText style={styles.cardDesc}>
              Speak naturally and have your words converted into simplified
              text and visual symbols for easy understanding.
            </AppText>

            <View style={styles.cardFooter}>
              <AppText style={styles.startText}>Start Speaking</AppText>
              <Ionicons name="arrow-forward" size={16} color="#fff" />
            </View>
          </TouchableOpacity>

          {/* Communicate */}
          <TouchableOpacity
            style={[styles.card, styles.purpleCard]}
            onPress={() => router.push('/communicate')}
          >
            <View style={styles.cardIconContainer}>
              <Ionicons name="hand-left" size={24} color="#fff" />
            </View>
            <AppText style={styles.cardTitle}>Communicate</AppText>
            <AppText style={styles.cardSub}>
              Disabled → Normal Person
            </AppText>
            <AppText style={styles.cardDesc}>
              Use sign language, gestures, eye tracking, or touch to express
              yourself - we'll translate it to natural speech.
            </AppText>

            <View style={styles.cardFooter}>
              <AppText style={styles.startText}>
                Start Communicating
              </AppText>
              <Ionicons name="arrow-forward" size={16} color="#fff" />
            </View>
          </TouchableOpacity>
        </View>

        {/* Symbol Board */}
        <TouchableOpacity 
          style={[styles.symbolBoard, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}
          onPress={() => router.push('/symbols')}
        >
          <View style={[styles.symbolIconContainer, { backgroundColor: resolvedTheme === 'dark' ? '#1e3a5f' : '#eff6ff' }]}>
            <Ionicons name="grid" size={22} color={activeTheme.primary} />
          </View>
          <View style={styles.symbolContent}>
            <AppText style={[styles.symbolTitle, { color: activeTheme.text }]}>Symbol Board</AppText>
            <AppText style={[styles.symbolSub, { color: subtleText }]}>
              Browse and select from our visual symbol library
            </AppText>
          </View>
          <Ionicons
            name="chevron-forward"
            size={20}
            color={subtleText}
          />
        </TouchableOpacity>

        {/* Quick Start Section */}
        <View style={[styles.quickStartSection, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
          <AppText style={[styles.quickStartTitle, { color: activeTheme.text }]}>Quick Start</AppText>
          <AppText style={[styles.quickStartSubtitle, { color: subtleText }]}>Choose a communication mode to begin</AppText>
          <TouchableOpacity style={styles.quickStartOption} onPress={() => router.push('/speak')}>
            <View style={[styles.quickStartIconContainer, { backgroundColor: resolvedTheme === 'dark' ? '#1e3a5f' : '#eff6ff' }]}>
              <Ionicons name="mic" size={18} color={activeTheme.primary} />
            </View>
            <AppText style={[styles.quickStartText, { color: activeTheme.primary }]}>I want to speak</AppText>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickStartOption} onPress={() => router.push('/communicate')}>
            <View style={[styles.quickStartIconContainer, { backgroundColor: resolvedTheme === 'dark' ? '#3b1d5c' : '#f3e8ff' }]}>
              <Ionicons name="hand-left" size={18} color="#9333ea" />
            </View>
            <AppText style={[styles.quickStartText, { color: '#9333ea' }]}>I want to communicate</AppText>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: 20,
    paddingBottom: 40,
  },

  headerSection: {
    alignItems: 'center',
    marginBottom: 28,
  },

  welcome: {
    fontSize: 24,
    color: '#374151',
    marginBottom: 4,
  },

  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2563eb',
    marginBottom: 12,
  },

  subtitle: {
    textAlign: 'center',
    lineHeight: 22,
    color: '#6b7280',
    paddingHorizontal: 10,
  },

  cardRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },

  card: {
    width: '48%',
    borderRadius: 20,
    padding: 18,
    minHeight: 220,
  },

  blueCard: {
    backgroundColor: '#2563eb',
  },

  purpleCard: {
    backgroundColor: '#9333ea',
  },

  cardIconContainer: {
    width: 42,
    height: 42,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  cardTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 12,
  },

  cardSub: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    marginBottom: 8,
  },

  cardDesc: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 12,
    lineHeight: 18,
    marginBottom: 14,
  },

  cardFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 'auto',
  },

  startText: {
    color: '#fff',
    fontWeight: '600',
    marginRight: 6,
    fontSize: 13,
  },

  symbolBoard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 16,
    backgroundColor: '#f9fafb',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    marginBottom: 28,
  },

  symbolIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#eff6ff',
    alignItems: 'center',
    justifyContent: 'center',
  },

  symbolContent: {
    flex: 1,
    marginLeft: 14,
  },

  symbolTitle: {
    fontWeight: '600',
    fontSize: 16,
    color: '#111827',
  },

  symbolSub: {
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },

  quickStartSection: {
    backgroundColor: '#f9fafb',
    borderRadius: 18,
    padding: 18,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },

  quickStartTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 4,
  },

  quickStartSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 16,
  },

  quickStartOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
  },

  quickStartIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: '#eff6ff',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },

  quickStartText: {
    fontSize: 14,
    color: '#2563eb',
    fontWeight: '500',
  },
});

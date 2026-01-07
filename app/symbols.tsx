 import Ionicons from '@expo/vector-icons/Ionicons';
import * as Speech from 'expo-speech';
import { useState } from 'react';
import {
    Appearance,
    ScrollView,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View,
} from 'react-native';

import AppText from '../components/AppText';
import Screen from '../components/Screen';
import { themes } from '../constants/theme';
import { useSettings } from './context/SettingsContext';

// All Symbols with Emojis
const ALL_SYMBOLS = [
  { id: 'hello', emoji: '👋', name: 'Hello', category: 'greetings' },
  { id: 'goodbye', emoji: '🖐️', name: 'Goodbye', category: 'greetings' },
  { id: 'thankyou', emoji: '🙏', name: 'Thank you', category: 'greetings' },
  { id: 'please', emoji: '🤲', name: 'Please', category: 'greetings' },
  { id: 'yes', emoji: '✅', name: 'Yes', category: 'greetings' },
  { id: 'no', emoji: '❌', name: 'No', category: 'greetings' },
  { id: 'help', emoji: '🆘', name: 'Help', category: 'actions' },
  { id: 'i', emoji: '👤', name: 'I', category: 'people' },
  { id: 'you', emoji: '👉', name: 'You', category: 'people' },
  { id: 'want', emoji: '🙋', name: 'Want', category: 'actions' },
  { id: 'happy', emoji: '😊', name: 'Happy', category: 'emotions' },
  { id: 'sad', emoji: '😢', name: 'Sad', category: 'emotions' },
  { id: 'hungry', emoji: '🍽️', name: 'Hungry', category: 'food' },
  { id: 'thirsty', emoji: '💧', name: 'Thirsty', category: 'food' },
  { id: 'home', emoji: '🏠', name: 'Home', category: 'places' },
  { id: 'school', emoji: '🏫', name: 'School', category: 'places' },
  { id: 'doctor', emoji: '👨‍⚕️', name: 'Doctor', category: 'people' },
  { id: 'love', emoji: '❤️', name: 'Love', category: 'emotions' },
  { id: 'stop', emoji: '🛑', name: 'Stop', category: 'actions' },
  { id: 'go', emoji: '🚶', name: 'Go', category: 'actions' },
  { id: 'what', emoji: '❓', name: 'What', category: 'questions' },
  { id: 'where', emoji: '📍', name: 'Where', category: 'questions' },
  { id: 'when', emoji: '🕐', name: 'When', category: 'questions' },
  { id: 'why', emoji: '🤔', name: 'Why', category: 'questions' },
  { id: 'water', emoji: '💦', name: 'Water', category: 'food' },
  { id: 'food', emoji: '🍔', name: 'Food', category: 'food' },
  { id: 'tired', emoji: '😴', name: 'Tired', category: 'emotions' },
  { id: 'angry', emoji: '😠', name: 'Angry', category: 'emotions' },
  { id: 'scared', emoji: '😨', name: 'Scared', category: 'emotions' },
  { id: 'play', emoji: '🎮', name: 'Play', category: 'actions' },
  { id: 'sleep', emoji: '💤', name: 'Sleep', category: 'actions' },
  { id: 'bathroom', emoji: '🚽', name: 'Bathroom', category: 'places' },
  { id: 'mom', emoji: '👩', name: 'Mom', category: 'people' },
  { id: 'dad', emoji: '👨', name: 'Dad', category: 'people' },
  { id: 'friend', emoji: '🤝', name: 'Friend', category: 'people' },
  { id: 'outside', emoji: '🌳', name: 'Outside', category: 'places' },
  { id: 'more', emoji: '➕', name: 'More', category: 'actions' },
  { id: 'done', emoji: '✔️', name: 'Done', category: 'actions' },
  { id: 'sorry', emoji: '😔', name: 'Sorry', category: 'greetings' },
  { id: 'wait', emoji: '✋', name: 'Wait', category: 'actions' },
];

const CATEGORIES = [
  { id: 'all', name: 'All Symbols', icon: 'grid' },
  { id: 'greetings', name: 'Greetings', icon: 'hand-left' },
  { id: 'emotions', name: 'Emotions', icon: 'heart' },
  { id: 'actions', name: 'Actions', icon: 'flash' },
  { id: 'people', name: 'People', icon: 'people' },
  { id: 'food', name: 'Food & Drink', icon: 'restaurant' },
  { id: 'places', name: 'Places', icon: 'location' },
  { id: 'questions', name: 'Questions', icon: 'help-circle' },
];

export default function Symbols() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sentence, setSentence] = useState<typeof ALL_SYMBOLS>([]);
  const [activeTab, setActiveTab] = useState<'search' | 'favorites' | 'recent'>('search');
  
  // Theme support
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  const subtleText = resolvedTheme === 'dark' ? '#cbd5e1' : '#6b7280';

  const filteredSymbols = ALL_SYMBOLS.filter(symbol => {
    const matchesCategory = selectedCategory === 'all' || symbol.category === selectedCategory;
    const matchesSearch = symbol.name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const addToSentence = (symbol: typeof ALL_SYMBOLS[0]) => {
    setSentence(prev => [...prev, symbol]);
  };

  const removeFromSentence = (index: number) => {
    setSentence(prev => prev.filter((_, i) => i !== index));
  };

  const clearSentence = () => {
    setSentence([]);
  };

  const speakSentence = () => {
    if (sentence.length > 0) {
      const text = sentence.map(s => s.name).join(' ');
      Speech.speak(text, { language: 'en' });
    }
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <AppText style={[styles.title, { color: activeTheme.text }]}>Symbol Board</AppText>
        <AppText style={[styles.subtitle, { color: subtleText }]}>
          Tap symbols below to build sentences. Use the speaker button to read aloud.
        </AppText>

        <View style={styles.mainContent}>
          {/* Left Side - Symbol Grid */}
          <View style={styles.leftSection}>
            {/* Sentence Builder */}
            <View style={[styles.sentenceBuilder, { backgroundColor: resolvedTheme === 'dark' ? '#422006' : '#fef9c3' }]}>
              {sentence.length > 0 ? (
                <View style={styles.sentenceContent}>
                  <View style={styles.sentenceSymbols}>
                    {sentence.map((s, i) => (
                      <TouchableOpacity key={i} style={styles.sentenceItem} onPress={() => removeFromSentence(i)}>
                        <Text style={styles.sentenceEmoji}>{s.emoji}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                  <View style={styles.sentenceActions}>
                    <TouchableOpacity style={styles.speakBtn} onPress={speakSentence}>
                      <Ionicons name="volume-high" size={20} color="#fff" />
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.clearBtn, { backgroundColor: resolvedTheme === 'dark' ? '#7f1d1d' : '#fee2e2' }]} onPress={clearSentence}>
                      <Ionicons name="trash-outline" size={20} color="#ef4444" />
                    </TouchableOpacity>
                  </View>
                </View>
              ) : (
                <AppText style={[styles.sentencePlaceholder, { color: resolvedTheme === 'dark' ? '#fbbf24' : '#92400e' }]}>Tap symbols below to build your sentence</AppText>
              )}
            </View>

            {/* Search Bar */}
            <View style={styles.searchRow}>
              <View style={[styles.searchBox, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
                <Ionicons name="search" size={18} color={subtleText} />
                <TextInput
                  style={[styles.searchInput, { color: activeTheme.text }]}
                  placeholder="Search symbols..."
                  placeholderTextColor={subtleText}
                  value={searchQuery}
                  onChangeText={setSearchQuery}
                />
              </View>
              <View style={styles.tabRow}>
                <TouchableOpacity
                  style={[styles.tabBtn, { backgroundColor: resolvedTheme === 'dark' ? '#1e293b' : '#f3f4f6' }, activeTab === 'favorites' && styles.tabBtnActive]}
                  onPress={() => setActiveTab('favorites')}
                >
                  <Ionicons name="star" size={16} color={activeTab === 'favorites' ? activeTheme.primary : subtleText} />
                  <AppText style={[styles.tabBtnText, { color: subtleText }, activeTab === 'favorites' && { color: activeTheme.primary }]}>Favorites</AppText>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.tabBtn, { backgroundColor: resolvedTheme === 'dark' ? '#1e293b' : '#f3f4f6' }, activeTab === 'recent' && styles.tabBtnActive]}
                  onPress={() => setActiveTab('recent')}
                >
                  <Ionicons name="time" size={16} color={activeTab === 'recent' ? activeTheme.primary : subtleText} />
                  <AppText style={[styles.tabBtnText, { color: subtleText }, activeTab === 'recent' && { color: activeTheme.primary }]}>Recent</AppText>
                </TouchableOpacity>
              </View>
            </View>

            {/* Category Tabs */}
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categoryScroll}>
              <View style={styles.categoryRow}>
                {CATEGORIES.map(cat => (
                  <TouchableOpacity
                    key={cat.id}
                    style={[
                      styles.categoryTab, 
                      { backgroundColor: resolvedTheme === 'dark' ? '#1e293b' : '#f3f4f6' },
                      selectedCategory === cat.id && { backgroundColor: activeTheme.primary }
                    ]}
                    onPress={() => setSelectedCategory(cat.id)}
                  >
                    <AppText style={[
                      styles.categoryTabText, 
                      { color: subtleText },
                      selectedCategory === cat.id && { color: '#fff' }
                    ]}>
                      {cat.name}
                    </AppText>
                  </TouchableOpacity>
                ))}
              </View>
            </ScrollView>

            {/* Symbol Grid */}
            <View style={[styles.symbolGrid, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]}>
              {filteredSymbols.map(symbol => (
                <TouchableOpacity 
                  key={symbol.id} 
                  style={[styles.symbolCard, { backgroundColor: activeTheme.card, borderColor: activeTheme.border }]} 
                  onPress={() => addToSentence(symbol)}
                >
                  <Text style={styles.symbolEmoji}>{symbol.emoji}</Text>
                  <AppText style={[styles.symbolName, { color: activeTheme.text }]}>{symbol.name}</AppText>
                </TouchableOpacity>
              ))}
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

  mainContent: { flexDirection: 'row', gap: 20 },
  leftSection: { flex: 1 },

  sentenceBuilder: {
    backgroundColor: '#fef9c3', borderRadius: 16, padding: 16,
    minHeight: 70, marginBottom: 16, justifyContent: 'center',
  },
  sentenceContent: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  sentenceSymbols: { flexDirection: 'row', flexWrap: 'wrap', flex: 1, gap: 8 },
  sentenceItem: { padding: 4 },
  sentenceEmoji: { fontSize: 28 },
  sentenceActions: { flexDirection: 'row', gap: 10 },
  speakBtn: { backgroundColor: '#22c55e', padding: 12, borderRadius: 20 },
  clearBtn: { backgroundColor: '#fee2e2', padding: 12, borderRadius: 20 },
  sentencePlaceholder: { color: '#92400e', fontSize: 14, textAlign: 'center' },

  searchRow: { flexDirection: 'row', gap: 12, marginBottom: 16 },
  searchBox: {
    flex: 1, flexDirection: 'row', alignItems: 'center',
    backgroundColor: '#fff', borderRadius: 12, paddingHorizontal: 14,
    borderWidth: 1, borderColor: '#e5e7eb', height: 44,
  },
  searchInput: { flex: 1, marginLeft: 10, fontSize: 14, color: '#111827' },
  tabRow: { flexDirection: 'row', gap: 8 },
  tabBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    paddingHorizontal: 12, paddingVertical: 10, borderRadius: 10,
    backgroundColor: '#f3f4f6',
  },
  tabBtnActive: { backgroundColor: '#eff6ff' },
  tabBtnText: { fontSize: 12, color: '#6b7280', fontWeight: '500' },
  tabBtnTextActive: { color: '#2563eb' },

  categoryScroll: { marginBottom: 16 },
  categoryRow: { flexDirection: 'row', gap: 10 },
  categoryTab: {
    paddingHorizontal: 16, paddingVertical: 10, borderRadius: 20,
    backgroundColor: '#f3f4f6',
  },
  categoryTabActive: { backgroundColor: '#2563eb' },
  categoryTabText: { fontSize: 13, color: '#6b7280', fontWeight: '500' },
  categoryTabTextActive: { color: '#fff' },

  symbolGrid: {
    flexDirection: 'row', flexWrap: 'wrap', gap: 12,
    backgroundColor: '#fff', borderRadius: 16, padding: 16,
    borderWidth: 1, borderColor: '#e5e7eb',
  },
  symbolCard: {
    width: '18%', aspectRatio: 1, backgroundColor: '#fff', borderRadius: 14,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: '#e5e7eb',
    shadowColor: '#000', shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05, shadowRadius: 2, elevation: 1,
  },
  symbolEmoji: { fontSize: 32, marginBottom: 4 },
  symbolName: { fontSize: 11, color: '#374151', fontWeight: '500', textAlign: 'center' },
});

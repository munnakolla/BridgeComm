import Ionicons from '@expo/vector-icons/Ionicons';
import Slider from '@react-native-community/slider';
import {
    Appearance,
    ScrollView,
    StyleSheet,
    Switch,
    TouchableOpacity,
    View,
} from 'react-native';

import AppText from '../components/AppText';
import { themes } from '../constants/theme';
import { useSettings } from './context/SettingsContext';

export default function Settings() {
  const {
    theme,
    setTheme,
    fontSize,
    setFontSize,
    highContrast,
    setHighContrast,
    speechRate,
    setSpeechRate,
    volume,
    setVolume,
  } = useSettings();

  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: activeTheme.bg }]}
      contentContainerStyle={styles.scrollContent}
      showsVerticalScrollIndicator={false}
    >
      {/* HEADER */}
      <AppText style={styles.title}>
        Settings
      </AppText>
      <AppText style={styles.subtitle}>
        Customize your BridgeComm experience
      </AppText>

      {/* APPEARANCE */}
      <View style={[styles.card, { backgroundColor: activeTheme.card }]}>
        <View style={styles.sectionHeader}>
          <Ionicons name="color-palette-outline" size={20} color="#2563eb" />
          <AppText style={styles.sectionTitle}>
            Appearance
          </AppText>
        </View>

        {/* THEME */}
        <AppText style={styles.label}>
          Theme
        </AppText>
        <View style={styles.optionRow}>
          <OptionButton label="Light" active={theme === 'light'} onPress={() => setTheme('light')} />
          <OptionButton label="Dark" active={theme === 'dark'} onPress={() => setTheme('dark')} />
          <OptionButton label="System" active={theme === 'system'} onPress={() => setTheme('system')} />
        </View>

        {/* FONT SIZE */}
        <AppText style={styles.label}>
          Font Size
        </AppText>
        <View style={styles.optionRow}>
          <OptionButton label="Small" active={fontSize === 'small'} onPress={() => setFontSize('small')} />
          <OptionButton label="Medium" active={fontSize === 'medium'} onPress={() => setFontSize('medium')} />
          <OptionButton label="Large" active={fontSize === 'large'} onPress={() => setFontSize('large')} />
          <OptionButton label="Extra Large" active={fontSize === 'xl'} onPress={() => setFontSize('xl')} />
        </View>

        {/* HIGH CONTRAST */}
        <View style={styles.switchRow}>
          <AppText style={styles.label}>
            High Contrast Mode
          </AppText>
          <Switch value={highContrast} onValueChange={setHighContrast} />
        </View>
      </View>

      {/* SPEECH & AUDIO */}
      <View style={[styles.card, { backgroundColor: activeTheme.card }]}>
        <View style={styles.sectionHeader}>
          <Ionicons name="volume-high-outline" size={20} color="#2563eb" />
          <AppText style={styles.sectionTitle}>
            Speech & Audio
          </AppText>
        </View>

        {/* SPEECH RATE */}
        <AppText style={styles.label}>
          Speech Rate
        </AppText>
        <View style={styles.sliderRow}>
          <Slider
            style={{ flex: 1 }}
            minimumValue={0.5}
            maximumValue={2}
            value={speechRate}
            onValueChange={setSpeechRate}
            minimumTrackTintColor="#2563eb"
          />
          <AppText style={styles.sliderValue}>
            {speechRate.toFixed(1)}x
          </AppText>
        </View>

        {/* VOLUME */}
        <AppText style={styles.label}>
          Speech Volume
        </AppText>
        <View style={styles.sliderRow}>
          <Slider
            style={{ flex: 1 }}
            minimumValue={0}
            maximumValue={1}
            value={volume}
            onValueChange={setVolume}
            minimumTrackTintColor="#2563eb"
          />
          <AppText style={styles.sliderValue}>
            {Math.round(volume * 100)}%
          </AppText>
        </View>
      </View>
    </ScrollView>
  );
}

/* ---------- REUSABLE OPTION BUTTON ---------- */
function OptionButton({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.optionButton, active && styles.optionActive]}
      onPress={onPress}
    >
      <AppText style={[styles.optionText, active && styles.optionTextActive]}>
        {label}
      </AppText>
    </TouchableOpacity>
  );
}

/* ================= STYLES ================= */

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollContent: { padding: 20, paddingBottom: 40 },

  title: { fontSize: 26, fontWeight: 'bold', marginBottom: 4 },
  subtitle: { marginBottom: 24, opacity: 0.7 },

  card: {
    borderRadius: 22,
    padding: 18,
    marginBottom: 22,
    elevation: 3,
  },

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
  },

  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },

  label: {
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
    marginTop: 10,
  },

  optionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 10,
  },

  optionButton: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    marginRight: 10,
    marginBottom: 10,
  },

  optionActive: {
    backgroundColor: '#eff6ff',
    borderColor: '#2563eb',
  },

  optionText: {
    fontSize: 13,
    color: '#6b7280',
  },

  optionTextActive: {
    color: '#2563eb',
    fontWeight: '600',
  },

  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 10,
  },

  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
  },

  sliderValue: {
    width: 50,
    textAlign: 'right',
    fontSize: 13,
    marginLeft: 8,
  },
});

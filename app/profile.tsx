import Ionicons from '@expo/vector-icons/Ionicons';
import { Appearance, ScrollView, StyleSheet, View } from 'react-native';
import Svg, { Circle, Polyline } from 'react-native-svg';

import AppText from '../components/AppText';
import Screen from '../components/Screen';
import { themes } from '../constants/theme';
import { useSettings } from './context/SettingsContext';

/* Weekly progress data */
const weeklyData = [
  { day: 'Mon', value: 30 },
  { day: 'Tue', value: 60 },
  { day: 'Wed', value: 20 },
  { day: 'Thu', value: 80 },
  { day: 'Fri', value: 50 },
  { day: 'Sat', value: 10 },
  { day: 'Sun', value: 40 },
];

export default function Profile() {
  const chartWidth = 300;
  const chartHeight = 120;
  const maxValue = 100;
  
  // Theme support
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  const subtleText = resolvedTheme === 'dark' ? '#cbd5e1' : '#6b7280';

  const points = weeklyData
    .map((item, index) => {
      const x = (index / (weeklyData.length - 1)) * chartWidth;
      const y = chartHeight - (item.value / maxValue) * chartHeight;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <Screen>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* HEADER */}
        <AppText style={[styles.title, { color: activeTheme.text }]}>User Profile</AppText>
        <AppText style={[styles.subtitle, { color: subtleText }]}>
          Manage your profile and view your progress
        </AppText>

        {/* USER CARD */}
        <View style={styles.userCard}>
          <Ionicons name="person-circle" size={80} color="#fff" />
          <AppText style={styles.userName}>User</AppText>
          <AppText style={styles.memberText}>
            Member since January 2024
          </AppText>
        </View>

        {/* STATS */}
        <View style={styles.statsRow}>
          <Stat icon="pulse" label="Sessions" value="0" activeTheme={activeTheme} subtleText={subtleText} />
          <Stat icon="time" label="Time Used" value="0m" activeTheme={activeTheme} subtleText={subtleText} />
          <Stat icon="checkmark-circle" label="Accuracy" value="0%" activeTheme={activeTheme} subtleText={subtleText} />
          <Stat icon="trending-up" label="Improvement" value="+0%" activeTheme={activeTheme} subtleText={subtleText} />
        </View>

        {/* WEEKLY PROGRESS */}
        <View style={[styles.chartCard, { backgroundColor: activeTheme.card, borderColor: activeTheme.border, borderWidth: 1 }]}>
          <View style={styles.chartHeader}>
            <AppText style={[styles.chartTitle, { color: activeTheme.text }]}>
              Weekly Progress
            </AppText>
            <AppText style={[styles.chartFilter, { color: activeTheme.primary }]}>
              This Week
            </AppText>
          </View>

          <Svg width={chartWidth} height={chartHeight}>
            <Polyline
              points={points}
              fill="none"
              stroke={activeTheme.primary}
              strokeWidth={3}
            />
            {weeklyData.map((item, index) => {
              const x = (index / (weeklyData.length - 1)) * chartWidth;
              const y =
                chartHeight - (item.value / maxValue) * chartHeight;
              return (
                <Circle
                  key={index}
                  cx={x}
                  cy={y}
                  r={4}
                  fill={activeTheme.primary}
                />
              );
            })}
          </Svg>

          <View style={styles.daysRow}>
            {weeklyData.map(item => (
              <AppText key={item.day} style={[styles.dayText, { color: subtleText }]}>
                {item.day}
              </AppText>
            ))}
          </View>
        </View>
      </ScrollView>
    </Screen>
  );
}

/* STAT CARD */
function Stat({ icon, label, value, activeTheme, subtleText }: any) {
  return (
    <View style={[styles.statCard, { backgroundColor: activeTheme.card, borderColor: activeTheme.border, borderWidth: 1 }]}>
      <Ionicons name={icon} size={20} color={activeTheme.primary} />
      <AppText style={[styles.statValue, { color: activeTheme.text }]}>{value}</AppText>
      <AppText style={[styles.statLabel, { color: subtleText }]}>{label}</AppText>
    </View>
  );
}

/* STYLES */
const styles = StyleSheet.create({
  scrollContent: {
    padding: 20,
    paddingBottom: 50,
  },

  title: {
    fontSize: 26,
    fontWeight: 'bold',
  },

  subtitle: {
    marginBottom: 20,
  },

  userCard: {
    backgroundColor: '#2563eb',
    borderRadius: 20,
    padding: 25,
    alignItems: 'center',
    marginBottom: 20,
  },

  userName: {
    color: '#fff',
    fontSize: 22,
    fontWeight: 'bold',
  },

  memberText: {
    color: '#e0e7ff',
    fontSize: 13,
  },

  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },

  statCard: {
    width: '23%',
    borderRadius: 14,
    padding: 10,
    alignItems: 'center',
  },

  statValue: {
    fontWeight: 'bold',
    marginTop: 4,
  },

  statLabel: {
    fontSize: 11,
    textAlign: 'center',
  },

  chartCard: {
    borderRadius: 16,
    padding: 15,
  },

  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },

  chartTitle: {
    fontWeight: 'bold',
  },

  chartFilter: {
    color: '#2563eb',
    fontSize: 12,
  },

  daysRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 6,
  },

  dayText: {
    fontSize: 11,
  },
});

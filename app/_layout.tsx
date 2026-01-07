import Ionicons from '@expo/vector-icons/Ionicons';
import { useRouter } from 'expo-router';
import { Tabs } from 'expo-router/tabs';
import { Appearance, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { themes } from '../constants/theme';
import { SettingsProvider, useSettings } from './context/SettingsContext';

// Custom header component with logo and profile
function CustomHeader() {
  const router = useRouter();
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];
  
  return (
    <SafeAreaView style={[headerStyles.safeArea, { backgroundColor: activeTheme.card }]}> 
      <View style={[headerStyles.container, { borderBottomColor: activeTheme.border }]}> 
        <View style={headerStyles.logoContainer}>
          <View style={[headerStyles.logoIcon, { backgroundColor: activeTheme.primary }]}>
            <Text style={[headerStyles.logoText, { color: activeTheme.bg }]}>B</Text>
          </View>
          <View>
            <Text style={[headerStyles.title, { color: activeTheme.text }]}>BridgeComm</Text>
            <Text style={[headerStyles.subtitle, { color: '#6b7280' }]}>Communication Without Barriers</Text>
          </View>
        </View>
        
        <View style={headerStyles.rightSection}>
          <TouchableOpacity style={headerStyles.iconButton}>
            <Ionicons name="help-circle-outline" size={22} color={activeTheme.text} />
          </TouchableOpacity>
          <TouchableOpacity style={headerStyles.iconButton}>
            <Ionicons name="settings-outline" size={22} color={activeTheme.text} />
          </TouchableOpacity>
          <TouchableOpacity 
            style={[headerStyles.profileButton, { backgroundColor: activeTheme.bg, borderColor: activeTheme.border }]}
            onPress={() => router.push('/profile')}
          >
            <Ionicons name="person" size={16} color={activeTheme.text} />
            <Text style={[headerStyles.profileText, { color: activeTheme.text }]}>Guest</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const headerStyles = StyleSheet.create({
  safeArea: {
    backgroundColor: '#fff',
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  logoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  logoIcon: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: '#2563eb',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
  },
  subtitle: {
    fontSize: 11,
    color: '#6b7280',
  },
  rightSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconButton: {
    padding: 8,
  },
  profileButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
  },
  profileText: {
    fontSize: 13,
    color: '#374151',
    fontWeight: '500',
  },
  banner: {
    marginTop: 4,
    marginHorizontal: 12,
    padding: 8,
    borderRadius: 10,
    borderWidth: 1,
  },
  bannerText: {
    fontSize: 12,
    textAlign: 'center',
    fontWeight: '600',
  },
});

function ThemedTabs() {
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';
  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];

  return (
    <Tabs
      screenOptions={({ route }) => ({
        header: () => <CustomHeader />,
        tabBarIcon: ({ color, size }) => {
          let icon: keyof typeof Ionicons.glyphMap = 'home';

          if (route.name === 'home') icon = 'home';
          if (route.name === 'speak') icon = 'mic';
          if (route.name === 'communicate') icon = 'hand-left';
          if (route.name === 'symbols') icon = 'grid';
          if (route.name === 'settings') icon = 'settings';

          return <Ionicons name={icon} size={size} color={color} />;
        },
        tabBarActiveTintColor: activeTheme.primary,
        tabBarInactiveTintColor: '#6b7280',
        tabBarStyle: {
          backgroundColor: activeTheme.card,
          borderTopColor: activeTheme.border,
          paddingTop: 8,
          height: 60,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: '500',
          marginBottom: 6,
        },
      })}
    >
      <Tabs.Screen 
        name="home" 
        options={{ 
          title: 'Home',
        }} 
      />
      <Tabs.Screen 
        name="speak" 
        options={{ 
          title: 'Speak to User',
        }} 
      />
      <Tabs.Screen 
        name="communicate" 
        options={{ 
          title: 'Communicate',
        }} 
      />
      <Tabs.Screen 
        name="symbols" 
        options={{ 
          title: 'Symbol Board',
        }} 
      />
      <Tabs.Screen 
        name="settings" 
        options={{ 
          title: 'Settings',
        }} 
      />
      {/* Hidden screens */}
      <Tabs.Screen 
        name="index" 
        options={{ 
          href: null,
        }} 
      />
      <Tabs.Screen 
        name="profile" 
        options={{ 
          href: null,
        }} 
      />
      <Tabs.Screen 
        name="context" 
        options={{ 
          href: null,
        }} 
      />
    </Tabs>
  );
}

export default function RootLayout() {
  return (
    <SettingsProvider>
      <ThemedTabs />
    </SettingsProvider>
  );
}

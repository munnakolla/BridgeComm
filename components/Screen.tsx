import { Appearance, StyleSheet, View, ViewStyle } from 'react-native';
import { useSettings } from '../app/context/SettingsContext';
import { themes } from '../constants/theme';

export default function Screen({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: ViewStyle | ViewStyle[];
}) {
  const { theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';

  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];

  return (
    <View
      style={[
        styles.container,
        { backgroundColor: activeTheme.bg },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

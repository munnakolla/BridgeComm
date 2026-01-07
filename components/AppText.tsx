import { Appearance, Text, TextProps } from 'react-native';
import { useSettings } from '../app/context/SettingsContext';
import { fontSizes, themes } from '../constants/theme';

export default function AppText(props: TextProps) {
  const { fontSize, theme, highContrast } = useSettings();
  const systemTheme = Appearance.getColorScheme() || 'light';

  const resolvedTheme = theme === 'system' ? systemTheme : theme;
  const activeTheme = highContrast
    ? themes.contrast
    : themes[resolvedTheme as keyof typeof themes];

  return (
    <Text
      {...props}
      style={[
        {
          fontSize: fontSizes[fontSize as keyof typeof fontSizes],
          color: activeTheme.text,
        },
        props.style,
      ]}
    />
  );
}

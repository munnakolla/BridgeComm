import { createContext, useContext, useState } from 'react';

type ThemeType = 'light' | 'dark' | 'system';
type FontSizeType = 'small' | 'medium' | 'large' | 'xl';

type SettingsContextType = {
  theme: ThemeType;
  setTheme: (t: ThemeType) => void;
  fontSize: FontSizeType;
  setFontSize: (f: FontSizeType) => void;
  highContrast: boolean;
  setHighContrast: (v: boolean) => void;
  speechRate: number;
  setSpeechRate: (v: number) => void;
  volume: number;
  setVolume: (v: number) => void;
};

const SettingsContext = createContext<SettingsContextType | null>(null);

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<ThemeType>('system');
  const [fontSize, setFontSize] = useState<FontSizeType>('medium');
  const [highContrast, setHighContrast] = useState(false);
  const [speechRate, setSpeechRate] = useState(1);
  const [volume, setVolume] = useState(100);

  return (
    <SettingsContext.Provider
      value={{
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
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('useSettings must be used inside SettingsProvider');
  return ctx;
}

// Default export to satisfy expo-router (this file is context, not a route)
export default function SettingsContextPlaceholder() {
  return null;
}

import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Tutor turn labelling',
  description: 'Judge one thing about one tutor turn at a time.',
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: '#10131a',
  width: 'device-width',
  initialScale: 1,
};

// Applied before paint so the remembered theme does not flash the other one at a rater who is
// about to read three hundred turns on it.
const THEME_BOOT = `try{var t=localStorage.getItem('tl_theme');if(t)document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

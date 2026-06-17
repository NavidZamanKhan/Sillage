import type { Metadata } from "next";
import { Inter, Playfair_Display, DM_Serif_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  weight: ["500", "600", "700"],
  style: ["normal", "italic"],
  display: "swap",
});

const dmSerif = DM_Serif_Display({
  subsets: ["latin"],
  variable: "--font-dm-serif",
  weight: "400",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sillage",
  description:
    "Describe a mood, a note, a season, or a vibe, and let Sillage hand-pick fragrances written just for you. AI-powered perfume discovery.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${playfair.variable} ${dmSerif.variable} min-h-screen font-[family-name:var(--font-inter)] antialiased`}
        style={{ backgroundColor: "#fafaf8", color: "#2c3e2d" }}
      >
        {children}
      </body>
    </html>
  );
}

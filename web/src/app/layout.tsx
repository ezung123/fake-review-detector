import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Trust & Safety: Fake & AI Review Detector",
  description: "Independent Authorship and Deception Analysis for E-Commerce & Hospitality Reviews",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}


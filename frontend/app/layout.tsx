import "./globals.css";

type PageMetadata = {
  title: string;
  description: string;
};

export const metadata: PageMetadata = {
  title: "Vietnamese News Search Workbench",
  description: "Search demo, preprocessing report, and retrieval evaluation workbench for Vietnamese news.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}

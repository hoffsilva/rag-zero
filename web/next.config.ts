import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  // In production, vercel.json routes /api/* to the Python service before Next.js sees it.
  // Locally, forward /api/* to the FastAPI server started with uvicorn.
  async rewrites() {
    if (process.env.NODE_ENV !== "development") {
      return [];
    }
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};

export default nextConfig;

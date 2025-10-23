/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_GITHUB_URL: process.env.NEXT_PUBLIC_GITHUB_URL || '',
    NEXT_PUBLIC_WANDB_URL: process.env.NEXT_PUBLIC_WANDB_URL || '',
  },
};

module.exports = nextConfig;
import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

const isDev = process.env.NODE_ENV !== 'production';

export default defineConfig({
  plugins: isDev ? [basicSsl()] : [],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2020',
  },
  server: {
    host: true,
  },
});

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: {
    colors: { forest: '#12352c', pine: '#1c5b45', mint: '#e5f1e8', canvas: '#f5f7f2', ink: '#172a23', muted: '#708077' },
    fontFamily: { sans: ['DM Sans', 'sans-serif'], display: ['Manrope', 'sans-serif'] },
    boxShadow: { panel: '0 10px 30px rgba(30, 60, 42, .045)' },
  } },
  plugins: [],
}

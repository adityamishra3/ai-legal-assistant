/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          light: '#4B6BFB',
          DEFAULT: '#3D5BF5',
          dark: '#2941D9',
        },
        secondary: {
          light: '#62A9FF',
          DEFAULT: '#2E7CF6',
          dark: '#1E60CA',
        },
        neutral: {
          light: '#F5F7FB',
          DEFAULT: '#E4E7ED',
          dark: '#9095A1',
        },
      },
    },
  },
  plugins: [],
}
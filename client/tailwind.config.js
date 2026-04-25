/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"]
      },
      colors: {
        rent: {
          bg: "#0b0d11",
          surface: "#111318",
          card: "#161a21",
          elevated: "#1c2029",
          border: "#252a35",
          "border-light": "#2f3645",
          muted: "#8b95a8",
          orange: "#f97316",
          pink: "#ec4899",
          purple: "#7c5dfa",
          green: "#22c55e",
          cyan: "#06b6d4"
        }
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(124,93,250,0.12), 0 24px 68px rgba(0,0,0,0.45), 0 0 40px rgba(124,93,250,0.04)",
        soft: "0 8px 32px rgba(0,0,0,0.32), 0 1px 2px rgba(0,0,0,0.2)",
        card: "0 12px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03)",
        "input-glow": "0 0 0 1px rgba(124,93,250,0.18), 0 16px 48px rgba(0,0,0,0.4)",
        "cta-glow": "0 8px 24px rgba(124,93,250,0.25), 0 2px 8px rgba(124,93,250,0.15)"
      },
      backgroundImage: {
        "cta-gradient": "linear-gradient(135deg, #7c5dfa 0%, #ec4899 50%, #f97316 100%)",
        "cta-gradient-hover": "linear-gradient(135deg, #8b6ffe 0%, #f06cac 50%, #fb8c3c 100%)",
        "header-gradient": "linear-gradient(180deg, rgba(11,13,17,0.97) 0%, rgba(11,13,17,0.92) 100%)",
        "card-gradient": "linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%)",
        "purple-gradient": "linear-gradient(180deg, rgba(124,93,250,0.15) 0%, rgba(11,13,17,0.95) 100%)",
        "green-gradient": "linear-gradient(180deg, rgba(34,197,94,0.1) 0%, rgba(11,13,17,0.95) 100%)",
        "red-gradient": "linear-gradient(180deg, rgba(239,68,68,0.08) 0%, rgba(11,13,17,0.95) 100%)"
      },
      borderRadius: {
        "2xl": "16px",
        "3xl": "20px",
        "4xl": "24px",
        "5xl": "28px"
      },
      animation: {
        "fade-in": "fadeIn 300ms ease-out",
        "slide-up": "slideUp 350ms cubic-bezier(0.16, 1, 0.3, 1)",
        "pulse-dot": "pulseDot 1.4s infinite ease-in-out",
        "shimmer": "shimmer 2s infinite linear",
        "glow-pulse": "glowPulse 3s infinite ease-in-out"
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" }
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" }
        },
        pulseDot: {
          "0%, 80%, 100%": { transform: "scale(0.6)", opacity: "0.3" },
          "40%": { transform: "scale(1)", opacity: "1" }
        },
        shimmer: {
          from: { backgroundPosition: "200% center" },
          to: { backgroundPosition: "-200% center" }
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 20px rgba(124,93,250,0.1)" },
          "50%": { boxShadow: "0 0 30px rgba(124,93,250,0.2)" }
        }
      }
    }
  },
  plugins: []
};

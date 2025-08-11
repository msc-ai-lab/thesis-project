# Skin Cancer Detection Frontend

A React-based web application for dermatological image analysis using CNN models, Explainable AI (XAI), and Large Language Models (LLM) for generating detailed medical reports.

## 🎯 Overview

This frontend application provides a professional interface for dermatological image classification, featuring:

- **Image Upload & Preview**: Support for JPEG and PNG dermatological images
- **Real-time Analysis**: Asynchronous processing with progress tracking
- **AI-Powered Predictions**: Binary classification (Malignant/Benign) with confidence scores
- **Explainable AI Visualizations**: GradCAM and SHAP interpretability methods
- **LLM-Generated Reports**: Detailed explanatory reports for clinical understanding
- **Responsive Design**: Professional UI optimized for clinical and academic use

## 🛠 Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **Build Tool**: Vite 6.0
- **UI Library**: Material-UI (MUI) 5.15
- **HTTP Client**: Axios
- **Styling**: Emotion (CSS-in-JS)
- **Icons**: Material Icons
- **Linting**: ESLint with TypeScript support

## 📋 Prerequisites

- Node.js 18+ (Alpine Linux compatible)
- npm or yarn package manager
- Backend API server running on `http://localhost:8000`

## 🚀 Quick Start

### Development Setup

1. **Clone and navigate to the frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Open http://localhost:3000 in your browser
   - Ensure the backend API is running on http://localhost:8000

### Production Build

1. **Build the application**
   ```bash
   npm run build
   ```

2. **Preview production build**
   ```bash
   npm run preview
   ```

## 🐳 Docker Deployment

### Multi-stage Docker Build

The application uses a multi-stage Dockerfile for optimized production deployment:

1. **Build the Docker image**
   ```bash
   docker build -t skin-cancer-frontend .
   ```

2. **Run the container**
   ```bash
   docker run -p 80:80 skin-cancer-frontend
   ```

3. **Access the application**
   - Open http://localhost in your browser

### Docker Architecture

- **Stage 1**: Node.js build environment for compilation
- **Stage 2**: Nginx Alpine for serving static files
- **Configuration**: Custom nginx.conf for SPA routing support

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── App.tsx            # Main application component
│   ├── main.tsx           # Application entry point
│   ├── index.css          # Global styles and full-width layout
│   ├── vite-env.d.ts      # Vite type definitions
│   └── assets/            # Images and static resources
├── Dockerfile             # Multi-stage Docker configuration
├── nginx.conf             # Nginx configuration for SPA
├── package.json           # Dependencies and scripts
├── vite.config.ts         # Vite build configuration
├── tsconfig.json          # TypeScript configuration
├── tsconfig.app.json      # App-specific TypeScript config
├── tsconfig.node.json     # Node-specific TypeScript config
└── eslint.config.js       # ESLint configuration
```

## 🔧 Configuration

### Environment Variables

The application expects the backend API to be available at:
```
http://localhost:8000
```

To change the API endpoint, modify the axios calls in `src/App.tsx`.

### Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
  },
})
```

### Nginx Configuration

The production build uses Nginx with SPA-specific routing:

```nginx
try_files $uri $uri/ /index.html;
```

This ensures all routes are handled by React Router.

## 🎨 UI/UX Features

### Design System

- **Color Palette**: Clinical blue (#0d47a1) with red accents for critical information
- **Typography**: Roboto font family for medical professionalism
- **Layout**: Full-width responsive design optimized for data visualization
- **Components**: Material Design components with custom clinical styling

### User Workflow

1. **Upload**: Select dermatological image (JPEG/PNG)
2. **Preview**: Review uploaded image before analysis
3. **Analyze**: Submit for AI processing with real-time progress
4. **Results**: View prediction, confidence, and explanatory visualizations
5. **Report**: Read LLM-generated clinical insights

## 🔌 API Integration

### Endpoints

- `POST /analyze` - Submit image for analysis
- `GET /status/{job_id}` - Check analysis progress

### Response Format

```typescript
interface AnalysisResult {
  prediction: 'Malignant' | 'Benign'
  probabilities: { Benign: number; Malignant: number }
  llm_report: string
  images: { 
    original: string
    grad_cam: string
    shap: string 
  }
}
```

## 📱 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🧪 Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### Code Quality

- **ESLint**: Configured with React and TypeScript rules
- **TypeScript**: Strict mode enabled for type safety
- **Formatting**: Consistent code styling

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This application is for research and educational purposes only and is not intended for clinical diagnosis.**

### Security Considerations

- No client-side data persistence
- All analysis performed server-side
- Image data transmitted securely via HTTPS in production

## 👥 Project Team

**MSc AI Thesis Project**
- Dangol
- Iskierka  
- Yildirim

**Institution**: UWE Bristol

## 📄 License

This project is part of an academic thesis and is intended for educational and research purposes.

---

For backend API documentation and model details, please refer to the backend repository documentation.
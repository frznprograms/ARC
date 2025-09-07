# ARC React Frontend

Modern React/Next.js frontend for the ARC (Automated Review Checking) system.

## Requirements

### System Requirements
- **Node.js**: `>=18.18.0` (recommended: `>=20.0.0`)
- **npm**: `>=9.0.0` or **yarn**: `>=1.22.0`

### Dependencies
- Next.js 15.5.2
- React 19.1.0
- TypeScript 5.x
- Tailwind CSS 4.x
- Google Maps JavaScript API
- Lucide React (icons)

## Google Maps API Setup

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable billing (required even for free tier)

### 2. Enable Required APIs
Enable these APIs in your Google Cloud Console:
- **Maps JavaScript API** - For map display and interaction
- **Places API** - For location search and business data
- **Geocoding API** - For address ↔ coordinates conversion

### 3. Create API Key
1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **API Key**
3. **Restrict the key** (recommended):
   - **Application restrictions**: HTTP referrers
     - Add: `localhost:3000/*`
     - Add: `127.0.0.1:3000/*` 
     - Add your production domain: `yourdomain.com/*`
   - **API restrictions**: Select only the 3 APIs listed above

### 4. Environment Configuration
Create `frontend/.env` file:
```bash
# Google Maps API Key
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_api_key_here
```

⚠️ **Important**: 
- The key must start with `NEXT_PUBLIC_` to be available in the browser
- Never commit your actual API key to version control
- Use environment-specific keys for development/production

## Installation & Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Configure Environment
```bash
# Edit .env and add your Google Maps API key
```

### 3. Development Server
```bash
npm run dev
```
The application will be available at: http://localhost:3000

### 4. Production Build
```bash
npm run build
npm start
```

## Features

### Location Selection
- 🔍 **Smart search**: Type business names for auto-complete
- 🗺️ **Interactive map**: Click anywhere to select location
- 🏢 **Business detection**: Auto-detects businesses within 100m radius
- 📝 **Auto-fill**: Automatically populates category and description for recognised businesses

### Form Features
- ✅ **Auto-fill**: Category and description based on selected location
- 🎚️ **Rating slider**: 1-5 star rating selection
- 🔄 **Real-time validation**: Form validation with error messages
- 🎨 **Dark theme**: Professional dark UI throughout

### Integration
- 🔌 **FastAPI backend**: Connects to ML analysis pipeline at `http://127.0.0.1:8000`
- 📊 **Results display**: Shows analysis results with status indicators
- ⚡ **Real-time updates**: Loading states and error handling

## Troubleshooting

### Google Maps Not Loading
1. Check API key is correctly set in `.env`
2. Verify all 3 APIs are enabled in Google Cloud Console
3. Check browser console for API errors
4. Ensure API key restrictions allow your domain

### Build Errors
1. Check Node.js version: `node --version`
2. Clear npm cache: `npm ci`
3. Delete `.next` folder and rebuild

### Environment Issues
1. Ensure `.env` file is in `frontend/` directory
2. Restart development server after changing `.env`
3. Check environment variables are prefixed with `NEXT_PUBLIC_`
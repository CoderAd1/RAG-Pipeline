# Deployment Guide for RAG Pipeline

This branch contains all necessary configuration files for deploying the RAG Pipeline to Railway.

## Quick Deploy to Railway

### Prerequisites
- Railway account (sign up at [railway.app](https://railway.app))
- GitHub repository pushed
- Environment variables ready

### Environment Variables Needed

#### Backend Service
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PORT=8000
```

#### Frontend Service
```
VITE_API_URL=https://your-backend-url.up.railway.app
```

### Deployment Steps

1. **Deploy Backend**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select this repository
   - Add all backend environment variables
   - Generate domain and copy the URL

2. **Deploy Frontend**
   - In the same Railway project, click "New"
   - Select "GitHub Repo" → Same repository
   - Set Root Directory: `frontend`
   - Add `VITE_API_URL` with your backend URL
   - Generate domain

3. **Done!** Visit your frontend URL

## Files Added for Deployment

- `railway.json` - Railway configuration
- `Procfile` - Backend start command
- `frontend/vercel.json` - Frontend routing configuration
- `requirements.txt` - Python dependencies
- Updated `app/main.py` - CORS configuration for production
- Updated `frontend/src/services/api.js` - Environment-based API URL

## Alternative Deployment Options

### Render + Vercel
- Backend: Deploy to Render
- Frontend: Deploy to Vercel
- See main README for detailed steps

### Heroku
- Use `Procfile` for backend
- Deploy frontend separately to Vercel/Netlify

## Troubleshooting

### Backend Issues
- Check Railway logs for errors
- Verify all environment variables are set
- Ensure `requirements.txt` is up to date

### Frontend Issues
- Verify `VITE_API_URL` points to correct backend
- Check browser console for CORS errors
- Ensure backend CORS allows frontend domain

### CORS Errors
- Backend automatically allows Railway, Vercel, and Render domains
- If using custom domain, update `app/main.py` CORS settings

## Cost Estimates

- **Railway Free Tier**: $5 credit/month (~500 hours)
- **Supabase Free Tier**: 500MB database, 1GB storage
- **Qdrant Cloud Free Tier**: Available

## Support

For issues, check:
- Railway logs
- Browser console
- Network tab for failed requests

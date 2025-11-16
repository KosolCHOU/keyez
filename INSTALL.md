# KeyEZ - Installation Guide

## Prerequisites
- Python 3.12 or higher
- pip (Python package manager)

## Installation

### 1. Create Virtual Environment
```bash
cd keyez
python -m venv .venv
```

### 2. Activate Virtual Environment

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

**For GPU support (with CUDA):**
```bash
pip install -r requirements.txt
```

**For CPU only (lighter, no GPU):**
```bash
pip install -r requirements-cpu.txt
```

### 4. Setup Django

**Apply database migrations:**
```bash
python manage.py migrate
```

**Collect static files:**
```bash
python manage.py collectstatic --noinput
```

### 5. Run Development Server
```bash
python manage.py runserver
```

Visit `http://localhost:8000` in your browser.

## Project Structure
```
keyez/
├── keyez_site/          # Django settings
├── landing/             # Main app
│   ├── model/          # ML models
│   │   ├── emnist_letters_traced.pt
│   │   └── idx_to_char.json
│   ├── static/         # CSS, JS files
│   ├── templates/      # HTML templates
│   └── views.py        # API endpoints
├── manage.py
└── requirements.txt
```

## Features
- **SingKhmer Input**: Type Latin Khmer, get real Khmer script
- **Handwriting Recognition**: Draw English letters, get Khmer transliteration
- **Real-time Predictions**: AI-powered character recognition
- **Offline Support**: Works without internet (after initial load)

## API Endpoints
- `GET /` - Main landing page
- `POST /predict/` - Handwriting recognition endpoint
  - Input: `{"image": "base64_png_data"}`
  - Output: `{"predictions": [{"char": "a", "confidence": 0.95}, ...]}`

## Troubleshooting

**ImportError: No module named 'django'**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

**Model not found error**
- Ensure `emnist_letters_traced.pt` exists in `landing/model/`
- Ensure `idx_to_char.json` exists in `landing/model/`

**PIL/Pillow errors**
- Run `pip install Pillow` separately if needed

## Production Deployment

**Using Gunicorn:**
```bash
gunicorn keyez_site.wsgi:application --bind 0.0.0.0:8000
```

**Environment Variables:**
- Set `DEBUG=False` in production
- Configure `ALLOWED_HOSTS` in settings.py
- Use PostgreSQL or MySQL instead of SQLite

## License
© 2025 AstroAI • Made in Cambodia

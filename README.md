# 🩸 Blood Group Identifier — Fingerprint-Based Prediction System

A full-stack web application that predicts blood groups from fingerprint images using a CNN (Convolutional Neural Network) deep learning model.

## 🏗️ Architecture

| Component | Technology | Port |
|-----------|-----------|------|
| **Frontend** | React 19 + Vite | 5173 |
| **Backend API** | Node.js + Express 5 | 5001 |
| **ML Service** | Python + Flask + TensorFlow | 5000 |
| **Database** | MongoDB Atlas | Cloud |

## ✨ Features

- **Fingerprint-based blood group prediction** using trained CNN model
- **Image validation** — rejects non-fingerprint images and AI-generated fakes
- **Physical scanner support** — Mantra MFS100 fingerprint scanner integration
- **AES-256 encryption** for stored fingerprint images
- **JWT authentication** with OTP email verification
- **Admin dashboard** with analytics, user management, blood group distribution charts
- **Prediction history** with confidence scores and reliability analysis
- **Age verification** (18+ requirement for biometric data handling)
- **Dark/Light theme** support

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- MongoDB Atlas account (or local MongoDB)

### 1. Clone the repository
```bash
git clone https://github.com/PoovarasanJ2005/bloodgroup-project.git
cd bloodgroup-project
```

### 2. Setup Backend Server
```bash
cd server
cp .env.example .env    # Edit .env with your MongoDB URI and secrets
npm install
npm run dev
```

### 3. Setup ML Service
```bash
cd ml-model
pip install -r requirements.txt
python app.py
```

### 4. Setup Frontend
```bash
cd client-app
npm install
npm run dev
```

### 5. Open the app
Visit `http://localhost:5173` in your browser.

## 📁 Project Structure

```
├── client-app/          # React + Vite frontend
│   ├── src/
│   │   ├── pages/       # Login, Register, Dashboard, Predict, History, Admin
│   │   ├── components/  # Layout, shared UI components
│   │   ├── context/     # Auth & Theme providers
│   │   └── services/    # API client (axios)
│   └── ...
├── server/              # Node.js + Express backend
│   ├── config/          # MongoDB connection
│   ├── models/          # Mongoose schemas (User, Prediction)
│   ├── routes/          # API routes (auth, prediction, admin)
│   ├── middleware/       # JWT auth, age verification
│   └── utils/           # Encryption, email, token helpers
├── ml-model/            # Flask + TensorFlow ML service
│   ├── app.py           # Flask API with prediction endpoints
│   ├── image_validator.py # Fingerprint validation & AI detection
│   ├── saved_model/     # Trained CNN model files (.h5)
│   └── requirements.txt
├── dataset/             # Training data (not included in repo)
├── render.yaml          # Render.com deployment config
└── README.md
```

## 🌐 Deployment

### Recommended: Render (Backend + ML) + Vercel (Frontend)

| Service | Platform | Plan |
|---------|----------|------|
| Frontend | **Vercel** | Free |
| Backend API | **Render** | Free |
| ML Service | **Render** | Starter ($7/mo) |

#### Deploy Backend to Render
1. Connect your GitHub repo on [render.com](https://render.com)
2. The `render.yaml` auto-configures both services
3. Set environment variables in the Render dashboard

#### Deploy Frontend to Vercel
1. Import the repo on [vercel.app](https://vercel.com)
2. Set root directory to `client-app`
3. Set `VITE_API_URL` to your Render backend URL (e.g., `https://bloodgroup-server.onrender.com/api`)
4. Deploy!

## 🔑 Default Admin Credentials

| Field | Value |
|-------|-------|
| Email | `admin@bloodgroup.com` |
| Password | `Admin@123456` |

## 📊 API Endpoints

### Auth
- `POST /api/auth/register` — Register new user
- `POST /api/auth/login` — Login
- `POST /api/auth/verify-otp` — Verify email OTP
- `GET /api/auth/profile` — Get profile
- `PUT /api/auth/profile` — Update profile

### Predictions
- `POST /api/predictions/predict` — Upload fingerprint & get prediction
- `POST /api/predictions/scanner-predict` — Scanner-based prediction
- `POST /api/predictions/mfs100-capture` — MFS100 scanner proxy
- `GET /api/predictions/history` — Prediction history
- `GET /api/predictions/stats` — User stats

### Admin
- `GET /api/admin/dashboard` — Admin analytics
- `GET /api/admin/users` — All users
- `GET /api/admin/predictions` — All predictions
- `DELETE /api/admin/users/:id` — Delete user

### Health
- `GET /api/health` — Server health check

## 🛡️ Security

- JWT-based authentication with configurable expiry
- AES-256 encryption for fingerprint image storage
- Bcrypt password hashing (12 rounds)
- Rate limiting on all API and auth endpoints
- Helmet.js security headers
- CORS with whitelisted origins
- Age verification (18+) for biometric data compliance

## 📝 License

This project is part of a Final Year Project (Batch 11).

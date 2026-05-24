import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import connectDB, { isDatabaseConnected } from './config/db.js';
import authRoutes from './routes/auth.js';
import predictionRoutes from './routes/prediction.js';
import adminRoutes from './routes/admin.js';
import User from './models/User.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

const allowedOrigins = process.env.CORS_ORIGINS
  ? process.env.CORS_ORIGINS.split(',').map(o => o.trim())
  : ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:3000'];

app.use(cors({
  origin: allowedOrigins,
  credentials: true,
}));

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: { error: 'Too many requests. Please try again later.' },
});
app.use('/api/', limiter);

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  message: { error: 'Too many auth attempts. Please try again later.' },
});
app.use('/api/auth/', authLimiter);

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

const requireDatabase = (req, res, next) => {
  if (!isDatabaseConnected()) {
    return res.status(503).json({
      error: 'Database unavailable. Start MongoDB or update MONGODB_URI before using this endpoint.',
    });
  }

  next();
};

app.use('/api/auth', requireDatabase, authRoutes);
app.use('/api/predictions', requireDatabase, predictionRoutes);
app.use('/api/admin', requireDatabase, adminRoutes);

app.get('/api/health', (req, res) => {
  const databaseConnected = isDatabaseConnected();

  res.status(databaseConnected ? 200 : 503).json({
    status: databaseConnected ? 'healthy' : 'degraded',
    databaseConnected,
    timestamp: new Date().toISOString(),
  });
});

const seedAdmin = async () => {
  try {
    const adminExists = await User.findOne({ role: 'admin' });
    if (!adminExists) {
      await User.create({
        name: 'Admin',
        email: process.env.ADMIN_EMAIL || 'admin@bloodgroup.com',
        password: process.env.ADMIN_PASSWORD || 'Admin@123456',
        dateOfBirth: new Date('1990-01-01'),
        role: 'admin',
        isVerified: true,
      });
      console.log('Default admin user created');
    }
  } catch (error) {
    console.log('Admin seed skipped:', error.message);
  }
};

const startServer = async () => {
  const databaseReady = await connectDB();
  if (databaseReady) {
    await seedAdmin();
  }

  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`ML API expected at ${process.env.ML_API_URL}`);
    console.log(`MongoDB: ${process.env.MONGODB_URI}`);
  });
};

startServer();

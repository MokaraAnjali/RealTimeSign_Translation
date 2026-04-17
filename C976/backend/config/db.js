const mongoose = require("mongoose");

const connectDB = async () => {
  const mongoUri = process.env.MONGODB_URI;

  if (!mongoUri) {
    console.log("MongoDB skipped: set MONGODB_URI to enable the Node backend database.");
    return;
  }

  try {
    await mongoose.connect(mongoUri, {
      serverSelectionTimeoutMS: 5000,
    });
    console.log("Connected to DB");
  } catch (error) {
    console.log("MongoDB connection failed:", error.message);
  }
};

module.exports = connectDB;

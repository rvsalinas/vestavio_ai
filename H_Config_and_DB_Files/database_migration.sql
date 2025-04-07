-- Example migration to store camera feed info and streaming configurations

-- 1) Create a table for camera feeds if it doesn't exist
CREATE TABLE IF NOT EXISTS camera_feeds (
  id SERIAL PRIMARY KEY,
  camera_name VARCHAR(100) NOT NULL,
  streaming_url TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- 2) Optionally insert some initial data (sample camera entries)
INSERT INTO camera_feeds (camera_name, streaming_url)
VALUES
  ('Camera 1', 'rtsp://localhost:8554/cam1'),
  ('Camera 2', 'rtsp://localhost:8554/cam2');
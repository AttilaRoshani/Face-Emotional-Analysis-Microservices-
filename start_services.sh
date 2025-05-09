#!/bin/bash

# Kill any existing services on the required ports
echo "Stopping any existing services..."
pkill -f "python services/face_landmark/face_landmark_service.py"
pkill -f "python services/emotion/emotion_service.py"
pkill -f "python services/data_storage/data_storage_service.py"

# Wait a moment for services to stop
sleep 2

# Start Face Landmark Service
echo "Starting Face Landmark Service..."
python services/face_landmark/face_landmark_service.py &
LANDMARK_PID=$!

# Start Emotion Service
echo "Starting Emotion Service..."
python services/emotion/emotion_service.py &
EMOTION_PID=$!

# Start Data Storage Service
echo "Starting Data Storage Service..."
python services/data_storage/data_storage_service.py &
STORAGE_PID=$!

# Wait for services to initialize
echo "Waiting for services to initialize..."
sleep 5

# Check if services are running
if ps -p $LANDMARK_PID > /dev/null && ps -p $EMOTION_PID > /dev/null && ps -p $STORAGE_PID > /dev/null; then
    echo "All services started successfully!"
    echo "Face Landmark Service PID: $LANDMARK_PID"
    echo "Emotion Service PID: $EMOTION_PID"
    echo "Data Storage Service PID: $STORAGE_PID"
    echo "Starting webcam_analysis.py..."
    
    # Run webcam_analysis.py
    python webcam_analysis.py &
    WEBCAM_PID=$!
    
    # Keep script running and handle cleanup on exit
    trap "echo 'Stopping services...'; kill $LANDMARK_PID $EMOTION_PID $STORAGE_PID $WEBCAM_PID 2>/dev/null" EXIT
    
    # Wait for user to press Ctrl+C
    echo "Press Ctrl+C to stop all services"
    wait
else
    echo "Error: One or more services failed to start"
    exit 1
fi 
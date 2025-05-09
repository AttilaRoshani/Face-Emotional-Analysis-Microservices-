import cv2
import numpy as np
import time
import logging
import dlib
import os
import grpc
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from protos import aggregator_pb2
from protos import aggregator_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenCV to use GTK backend
os.environ['OPENCV_VIDEOIO_BACKEND'] = 'gtk'
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB platform
cv2.setUseOptimized(True)  # Enable OpenCV optimizations

# Service ports
EMOTION_PORT = 50052
FACE_LANDMARK_PORT = 50051

class WebcamAnalyzer:
    def __init__(self):
        # Initialize face detector
        logger.info("Loading face detector...")
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize gRPC channels
        logger.info("Initializing gRPC channels...")
        self.emotion_channel = grpc.insecure_channel(f'localhost:{EMOTION_PORT}')
        self.face_landmark_channel = grpc.insecure_channel(f'localhost:{FACE_LANDMARK_PORT}')
        
        # Initialize stubs
        self.emotion_stub = aggregator_pb2_grpc.AggregatorStub(self.emotion_channel)
        self.face_landmark_stub = aggregator_pb2_grpc.AggregatorStub(self.face_landmark_channel)
        
        logger.info("Webcam analyzer initialized")

    def analyze_face(self, face_img):
        """Analyze a single face image using emotion and landmark services"""
        try:
            # Convert image to bytes
            _, buffer = cv2.imencode('.jpg', face_img)
            img_bytes = buffer.tobytes()
            
            # Create request
            request = aggregator_pb2.FaceResult(
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
                frame=img_bytes,
                image_id="webcam_frame",
                original_filename="webcam",
                faces=[
                    aggregator_pb2.FaceData(
                        face_id="webcam_face",
                        face_image=img_bytes,
                        bbox=aggregator_pb2.BoundingBox(
                            x=0,
                            y=0,
                            width=face_img.shape[1],
                            height=face_img.shape[0]
                        )
                    )
                ]
            )
            
            # Get emotion analysis
            emotion_response = self.emotion_stub.SaveFaceAttributes(request)
            logger.info(f"Emotion response: {emotion_response}")
            
            # Get face landmarks
            landmark_response = self.face_landmark_stub.SaveFaceAttributes(request)
            logger.info(f"Landmark response: {landmark_response}")
            
            # Extract emotion info from the first face
            emotion = "Unknown"
            confidence = 0.0
            landmarks = []
            
            if emotion_response and emotion_response.faces and len(emotion_response.faces) > 0:
                face = emotion_response.faces[0]
                if hasattr(face, 'emotion'):
                    emotion = face.emotion.emotion
                    confidence = face.emotion.confidence
            
            # Extract landmark info from the first face
            if landmark_response and landmark_response.faces and len(landmark_response.faces) > 0:
                face = landmark_response.faces[0]
                if hasattr(face, 'landmarks'):
                    landmarks = face.landmarks.landmarks
            
            return emotion, confidence, landmarks
            
        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            return "Unknown", 0.0, []

    def process_frame(self, frame):
        """Process a single frame from webcam"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            # Process each face
            for face in faces:
                # Get face coordinates
                x = face.left()
                y = face.top()
                width = face.right() - x
                height = face.bottom() - y
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, frame.shape[1] - x)
                height = min(height, frame.shape[0] - y)
                
                # Extract face region
                face_img = frame[y:y+height, x:x+width]
                
                # Skip if face region is too small
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                # Analyze face
                emotion, confidence, landmarks = self.analyze_face(face_img)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Add emotion text with background
                text = f"{emotion} ({confidence:.1f}%)"
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                # Draw background rectangle
                cv2.rectangle(frame, (x, y-30), (x + text_width, y), (0, 0, 0), -1)
                # Draw text
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw landmarks if available
                if landmarks:
                    # Draw each landmark point
                    for i in range(0, len(landmarks), 2):
                        if i + 1 < len(landmarks):
                            # Adjust coordinates relative to face position
                            lx = int(landmarks[i]) + x
                            ly = int(landmarks[i + 1]) + y
                            # Draw only red filled circle for each landmark point
                            cv2.circle(frame, (lx, ly), 4, (0, 0, 255), -1)  # Red filled circle
            
            return frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def run(self):
        """Run webcam analysis"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return

        logger.info("Webcam opened successfully")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Face Emotion Analysis', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam analysis stopped")

if __name__ == '__main__':
    analyzer = WebcamAnalyzer()
    analyzer.run() 
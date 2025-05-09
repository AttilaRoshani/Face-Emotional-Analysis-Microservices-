import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import grpc
import time
import logging
import redis
import json
import cv2
import numpy as np
from concurrent import futures
from deepface import DeepFace
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
EMOTION_PORT = 50052

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class EmotionService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        logger.info("Emotion service initialized")

    def analyze_face(self, face_image):
        """Analyze a single face image using DeepFace"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(face_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode face image")
                return {
                    "emotion": "Unknown",
                    "confidence": 0.0,
                    "emotions": {}
                }
            
            # Resize image for faster processing
            img = cv2.resize(img, (224, 224))
            
            # Convert to RGB for DeepFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Analyze face
            result = DeepFace.analyze(
                img_rgb, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )

            if isinstance(result, list):
                result = result[0]

            # Get emotion information
            emotions = result.get("emotion", {})
            dominant_emotion = result.get("dominant_emotion", "neutral")
            
            # Get confidence
            confidence = emotions.get(dominant_emotion, 0.0)

            # Capitalize emotion names
            emotions = {k.capitalize(): v for k, v in emotions.items()}
            dominant_emotion = dominant_emotion.capitalize()

            return {
                "emotion": dominant_emotion,
                "confidence": float(confidence),
                "emotions": emotions
            }
        except Exception as e:
            logger.error(f"Error analyzing face: {e}")
            return {
                "emotion": "Unknown",
                "confidence": 0.0,
                "emotions": {}
            }

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Processing emotions for image {request.image_id}")

        # Process each face in the image
        face_results = []
        for face in request.faces:
            # Analyze face using DeepFace
            emotion_result = self.analyze_face(face.face_image)

            face_result = {
                "face_id": face.face_id,
                "emotion_data": emotion_result
            }
            face_results.append(face_result)

            logger.info(f"Face {face.face_id}: Emotion={emotion_result['emotion']}, Confidence={emotion_result['confidence']:.2f}%")

        # Store results in Redis
        redis_key = f"emotion:{request.image_id}"
        self.redis_client.set(redis_key, json.dumps(face_results))

        processing_time = time.time() - start_time
        logger.info(f"Emotion processing completed in {processing_time:.2f} seconds")

        # Create response with emotion info
        response = aggregator_pb2.FaceResultResponse(
            response=True,
            message=f"Processed emotions for {len(face_results)} faces in image {request.image_id}"
        )

        # Add face data to response
        for face_result in face_results:
            emotion_data = face_result["emotion_data"]
            face_data = aggregator_pb2.FaceData(
                face_id=face_result["face_id"],
                emotion=aggregator_pb2.EmotionInfo(
                    emotion=emotion_data["emotion"],
                    confidence=emotion_data["confidence"],
                    emotions=emotion_data["emotions"]
                )
            )
            response.faces.append(face_data)

        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(EmotionService(), server)
    server.add_insecure_port(f'[::]:{EMOTION_PORT}')
    server.start()
    logger.info(f"Emotion Service started on port {EMOTION_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 
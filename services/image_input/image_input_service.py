import os
import sys
import uuid
import time
import logging
import redis
import cv2
import numpy as np
import grpc
from concurrent import futures
from dotenv import load_dotenv
from protos import aggregator_pb2
from protos import aggregator_pb2_grpc

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service ports
IMAGE_INPUT_PORT = 50050
FACE_LANDMARK_PORT = 50051
EMOTION_PORT = 50052
DATA_STORAGE_PORT = 50053

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

class ImageInputService(aggregator_pb2_grpc.AggregatorServicer):
    def __init__(self):
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        # Initialize face detector
        logger.info("Loading dlib face detector...")
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        logger.info("dlib face detector loaded successfully")
        
        # Initialize gRPC channels with retry options
        self.face_landmark_channel = grpc.insecure_channel(
            f'localhost:{FACE_LANDMARK_PORT}',
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_send_message_length', 100 * 1024 * 1024)  # 100MB
            ]
        )
        self.face_landmark_stub = aggregator_pb2_grpc.AggregatorStub(self.face_landmark_channel)
        
        self.emotion_channel = grpc.insecure_channel(
            f'localhost:{EMOTION_PORT}',
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
        )
        self.emotion_stub = aggregator_pb2_grpc.AggregatorStub(self.emotion_channel)
        
        self.data_storage_channel = grpc.insecure_channel(
            f'localhost:{DATA_STORAGE_PORT}',
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.max_send_message_length', 100 * 1024 * 1024)
            ]
        )
        self.data_storage_stub = aggregator_pb2_grpc.AggregatorStub(self.data_storage_channel)
        
        logger.info("Image input service initialized")

    def detect_faces(self, image):
        """Detect faces in the image and return face data"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detected_faces = self.detector(gray)
            
            # Process each detected face
            faces = []
            for face in detected_faces:
                # Generate unique ID for this face
                face_id = str(uuid.uuid4())
                
                # Get face coordinates
                x = face.left()
                y = face.top()
                width = face.right() - x
                height = face.bottom() - y
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, image.shape[1] - x)
                height = min(height, image.shape[0] - y)
                
                # Extract face region
                face_img = image[y:y+height, x:x+width]
                
                # Skip if face region is too small
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                # Encode face image
                _, face_bytes = cv2.imencode('.jpg', face_img)
                
                # Create face data
                face_data = aggregator_pb2.FaceData(
                    face_id=face_id,
                    face_image=face_bytes.tobytes(),
                    bbox=aggregator_pb2.BoundingBox(
                        x=x,
                        y=y,
                        width=width,
                        height=height
                    )
                )
                faces.append(face_data)
                logger.info(f"Detected face {face_id}")
            
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def SaveFaceAttributes(self, request, context):
        start_time = time.time()
        logger.info(f"Processing image {request.image_id}")

        try:
            # Store original image filename in Redis
            if hasattr(request, 'original_filename'):
                image_filename_key = f"image_filename:{request.image_id}"
                self.redis_client.set(image_filename_key, request.original_filename)

            # Convert frame data to image
            nparr = np.frombuffer(request.frame, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image data")

            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                logger.warning(f"No faces detected in image {request.image_id}")
                return aggregator_pb2.FaceResultResponse(
                    response=False,
                    message="No faces detected in image"
                )

            # Create new request with faces
            new_request = aggregator_pb2.FaceResult(
                time=request.time,
                frame=request.frame,
                image_id=request.image_id,
                faces=faces
            )

            # Send to face landmark service
            try:
                landmark_response = self.face_landmark_stub.SaveFaceAttributes(new_request)
                logger.info(f"Face landmark service response: {landmark_response.message}")
            except Exception as e:
                logger.error(f"Error calling face landmark service: {str(e)}")
                return aggregator_pb2.FaceResultResponse(
                    response=False,
                    message=f"Error calling face landmark service: {str(e)}"
                )

            # Send to emotion service
            try:
                emotion_response = self.emotion_stub.SaveFaceAttributes(new_request)
                logger.info(f"Emotion service response: {emotion_response.message}")
            except Exception as e:
                logger.error(f"Error calling emotion service: {str(e)}")
                return aggregator_pb2.FaceResultResponse(
                    response=False,
                    message=f"Error calling emotion service: {str(e)}"
                )

            # Send to data storage service
            try:
                storage_response = self.data_storage_stub.SaveFaceAttributes(new_request)
                logger.info(f"Data storage service response: {storage_response.message}")
            except Exception as e:
                logger.error(f"Error calling data storage service: {str(e)}")
                return aggregator_pb2.FaceResultResponse(
                    response=False,
                    message=f"Error calling data storage service: {str(e)}"
                )

            processing_time = time.time() - start_time
            logger.info(f"Face detection completed in {processing_time:.2f} seconds")

            return aggregator_pb2.FaceResultResponse(
                response=True,
                message=f"Detected {len(faces)} faces in image {request.image_id}",
                faces=faces
            )

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return aggregator_pb2.FaceResultResponse(
                response=False,
                message=f"Error processing image: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    aggregator_pb2_grpc.add_AggregatorServicer_to_server(ImageInputService(), server)
    server.add_insecure_port(f'[::]:{IMAGE_INPUT_PORT}')
    server.start()
    logger.info(f"Image Input Service started on port {IMAGE_INPUT_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 
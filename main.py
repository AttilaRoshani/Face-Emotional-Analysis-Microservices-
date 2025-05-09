import logging
import os
from services.image_input.image_input_service import ImageInputService
from services.face_landmark.face_landmark_service import FaceLandmarkService
from services.age_gender.age_gender_service import EmotionService
from services.data_storage.data_storage_service import DataStorageService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_services():
    # Start Data Storage Service
    logger.info("Starting Data Storage Service...")
    data_storage = DataStorageService()
    data_storage.start()

    # Start Face Landmark Service
    logger.info("Starting Face Landmark Service...")
    face_landmark = FaceLandmarkService()
    face_landmark.start()

    # Start Emotion Service
    logger.info("Starting Emotion Service...")
    emotion = EmotionService()
    emotion.start()

    # Start Image Input Service
    logger.info("Starting Image Input Service...")
    image_input = ImageInputService()
    image_input.start()

    logger.info("All services started successfully!")
    logger.info("Waiting for real-time requests...")

    try:
        # Keep the main process running
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        data_storage.stop()
        face_landmark.stop()
        emotion.stop()
        image_input.stop()

if __name__ == "__main__":
    start_services() 
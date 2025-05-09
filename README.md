
# Face Emotional Analysis Microservices

A microservices-based face emotion analysis system that detects faces, extracts facial landmarks, and estimates age and gender using Docker containers. The system uses Redis for data storage and gRPC for service communication, making it scalable and efficient for processing multiple images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Face Emotional Analysis Microservices** project is designed to analyze facial emotions, estimate age, and determine gender using a modular microservices architecture. This system processes multiple images efficiently, leveraging Docker for containerization, Redis for caching and data storage, and gRPC for fast communication between services.

---

## Features

- **Face Detection**: Accurately detect and localize faces in images.
- **Facial Landmark Extraction**: Identify key facial landmarks for emotion analysis.
- **Emotion Estimation**: analyze facial emotions, of detected faces.
- **Scalable Microservices**: Designed for processing multiple images concurrently.
- **Inter-Process Communication**: Uses gRPC for seamless communication between services.
- **Data Storage**: Redis is used for efficient caching and data management.

---

## System Architecture

The system is composed of the following components:
1. **Face Detection Service**: Identifies and crops faces from input images.
2. **Landmark Detection Service**: Extracts key facial features for further processing.
3. **Age and Gender Estimation Service**: Predicts age and gender from facial features.
4. **Redis**: Acts as the centralized data store for caching and temporary data storage.
5. **gRPC**: Enables fast and efficient communication between microservices.
6. **Docker**: Each service runs in an isolated Docker container for scalability and reliability.

---

## Technologies Used

- **Programming Language**: Python (95.5%)
- **Scripting**: Shell Scripts (3.3%)
- **Containerization**: Docker (1.2%)
- **Data Storage**: Redis
- **Communication Protocol**: gRPC

---

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/AttilaRoshani/Face-Emotional-Analysis-Microservices-.git
   cd face-analysis-microservices

   # Run with Docker Compose
   docker-compose up --build
```

### Manual Installation
```bash
# Make the script executable
chmod +x start_services.sh

# Run services
./start_services.sh
```


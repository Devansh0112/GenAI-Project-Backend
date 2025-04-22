# Text-to-Image Generative AI Web Application

This project implements a full-stack web application using the Stable Diffusion model to generate images from text prompts. The implementation includes both a Python backend for model fine-tuning and inference, and a React frontend for user interaction.

## Architecture Overview

The application consists of the following components:

1. **Backend (Python)**
   - Stable Diffusion model implementation with:
     - CLIP tokenizer for text input processing
     - VAE encoder/decoder for image latent representations
     - U-Net architecture as the core diffusion model
   - Fine-tuning pipeline using the HuggingFace dataset
   - Flask API for serving the model
   - SQLite database for storing generated images and user feedback

2. **Frontend (React)**
   - Modern React application with React Router for navigation
   - Chakra UI for responsive, accessible interface components
   - Image generation interface with advanced parameters
   - Gallery for viewing and managing generated images
   - Retraining interface for model fine-tuning

## Key Features

- Text-to-image generation with customizable parameters
- Generation of image variations
- Image gallery with search and filtering
- User feedback collection for generated images
- Model retraining using the stored images and feedback
- Fine-tuning that only updates the U-Net component while keeping text and image encoders frozen

## Backend Components

### Model Architecture

The Stable Diffusion implementation follows the standard architecture:

- **CLIP Tokenizer & Text Encoder**: Converts text prompts into embeddings (frozen during fine-tuning)
- **VAE Encoder/Decoder**: Handles conversion between image and latent space (frozen during fine-tuning)
- **U-Net**: Core diffusion model that is fine-tuned during training

### Fine-Tuning Pipeline

The fine-tuning process:

1. Loads the HuggingFace dataset (`LeroyDyer/image-description_text_to_image_BASE64`)
2. Preprocesses images and captions
3. Updates only the U-Net parameters while keeping the CLIP and VAE components frozen
4. Saves checkpoints during training
5. Stores the fine-tuned model for later inference

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate image from text prompt |
| `/api/variations` | POST | Generate variations of an existing image |
| `/api/images` | GET | Get list of generated images |
| `/api/images/<id>` | GET | Get details of a specific image |
| `/api/feedback` | POST | Save user feedback for an image |
| `/api/retrain` | POST | Start model retraining |
| `/api/retrain/status` | GET | Get status of model training |

## Frontend Pages

1. **Image Generator**: Main interface for creating images from text prompts
2. **Gallery**: Browse and manage previously generated images
3. **Image Details**: View and interact with a specific image
4. **Retrain Model**: Configure and initiate model retraining

## Database Schema

The SQLite database includes the following structure:

```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    image_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    seed INTEGER,
    params TEXT,
    feedback INTEGER DEFAULT 0
)
```

## Getting Started

### 1. Set up your development environment

First, make sure you have the following prerequisites installed:
- Docker and Docker Compose (for containerized setup)
- Python 3.8+ (for local backend development)
- Node.js and npm (for local frontend development)
- Git (for version control)

### 2. Clone or create the project structure

Create the project directory structure as described:

```bash
mkdir -p stable-diffusion-app/backend stable-diffusion-app/frontend stable-diffusion-app/models stable-diffusion-app/data
cd stable-diffusion-app
```

### 3. Set up the files

Add all the files we've created to their appropriate locations in your directory structure:
- Place all the backend Python files in the `backend/` directory
- Place all the React files in the `frontend/` directory
- Place the Docker files in their respective locations:
  - `docker-compose.yml` in the root directory
  - Backend `Dockerfile` in the `backend/` directory
  - Frontend `Dockerfile` in the `frontend/` directory

### 4. Using Docker (Recommended for Production)

The simplest way to test everything is using Docker:

```bash
docker compose up --build
```

This will:
- Build both your frontend and backend containers
- Set up the appropriate network between them
- Mount volumes for persistent data storage
- Start both services with the correct environment variables

Once everything is running:
- The frontend will be available at http://localhost:3000
- The backend API will be available at http://localhost:5001/api

### 5. Running Locally (Better for Development)

For development, you might want to run things locally for faster iterations:

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## Testing the Application

Once your application is running:

1. **Generate your first image**:
   - Go to http://localhost:3000
   - Enter a text prompt like "A beautiful sunset over mountains"
   - Adjust parameters if desired
   - Click "Generate Image"

2. **View your gallery**:
   - Navigate to the Gallery page to see all generated images
   - Click on an image to view details

3. **Create variations**:
   - From the image details page, you can create variations of an image

4. **Retrain the model**:
   - After generating several images and providing feedback, try retraining the model
   - Note that training will take significant time and resources, especially on CPU

## Training and Fine-Tuning

The model fine-tuning process can be initiated through:

1. The web interface (Retrain Model page)
2. Directly running the training script:
   ```
   python backend/fine_tuning.py
   ```

By default, the fine-tuning process starts with the pre-trained Stable Diffusion v1-4 model and updates only the U-Net component using the images in the database.

## Important Testing Considerations

- **GPU Requirements**: For reasonable performance, you'll need a CUDA-compatible GPU, especially for model training. Without it, image generation will be very slow.

- **Memory Usage**: The Stable Diffusion model requires significant RAM (at least 8GB) and VRAM (at least 4GB for inference, 8GB+ for training).

- **First-time Downloads**: On first run, the system will download the base Stable Diffusion model (~4GB), which may take some time.

- **Dataset Size**: The HuggingFace dataset is quite large and will be downloaded during training.

## Troubleshooting

If you encounter issues:

- **CUDA errors**: Ensure you have the correct CUDA toolkit installed that matches your PyTorch version
- **Memory errors**: Try reducing batch size or image dimensions
- **API connection errors**: Check that your frontend is correctly configured to connect to the backend URL
- **Platform compatibility**: If using Mac with Apple Silicon (M1/M2), specify the platform in docker-compose.yml

## For Apple Silicon Macs

If using an Apple Silicon Mac (M1/M2/M3), update your docker-compose.yml to specify the platform:

```yaml
services:
  backend:
    platform: linux/arm64  # For Apple Silicon Macs
    # other settings...
    
  frontend:
    platform: linux/arm64  # For Apple Silicon Macs
    # other settings...
```

## Project Structure

```
stable-diffusion-app/
├── docker-compose.yml           # Main Docker Compose configuration
├── backend/                     
│   ├── Dockerfile               # Backend Docker configuration
│   ├── app.py                   # Flask API server
│   ├── model.py                 # Stable Diffusion model implementation
│   ├── fine_tuning.py           # Training pipeline
│   ├── inference.py             # Image generation service
│   ├── utils.py                 # Utility functions
│   └── requirements.txt         # Python dependencies
├── frontend/                    
│   ├── Dockerfile               # Frontend Docker configuration
│   ├── public/                  # Static assets
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/               # Page components
│   │   ├── utils/               # Utility functions
│   │   ├── App.js               # Main application component 
│   │   └── index.js             # React entry point
│   └── package.json             # Node.js dependencies
├── models/                      # Mounted volume for model storage
└── data/                        # Mounted volume for database storage
```

## Production Considerations

For a production deployment, consider the following:

- Use a more robust database like PostgreSQL
- Implement proper authentication and authorization
- Add model versioning and A/B testing capabilities
- Set up a job queue for handling training and inference tasks
- Deploy the backend on GPU-enabled infrastructure for faster inference
- Implement caching for frequently accessed images

## Future Enhancements

Potential improvements to the system:

- Support for image-to-image generation
- Inpainting and outpainting capabilities
- Style transfer functionality
- Integration with other diffusion models
- Advanced prompt engineering tools
- Batch processing of multiple prompts
- User galleries and sharing options

## License

This project is for educational purposes. The Stable Diffusion model is subject to its own license terms.
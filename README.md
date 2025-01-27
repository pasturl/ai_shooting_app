# AI Image Generator 🎨

A professional AI-powered image generation platform built with Streamlit and cutting-edge AI models. This application enables users to generate and manage AI-generated images with an intuitive interface.

## 🌟 Features

### Core Functionality
- **Image Generation**: Generate AI images using state-of-the-art models through Replicate API
- **Batch Processing**: Generate multiple images simultaneously
- **Fine-Tuning Capabilities**: Train and customize AI models for specific use cases
- **Image Management**: Organize and manage generated images with metadata
- **User-Friendly Interface**: Intuitive Streamlit-based UI for easy interaction

### Technical Features
- **Cloud Storage**: AWS integration with DynamoDB for data management
- **Streamlit Framework**: Modern and responsive web interface
- **Replicate Integration**: Access to cutting-edge AI models
- **Error Handling**: Robust error management and logging
- **Metrics Tracking**: Monitor usage statistics and performance

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- AWS Account (for DynamoDB)
- Replicate API access

### Setup
```bash
# Clone repository
git clone https://github.com/pasturl/ai_shooting_app.git
cd ai-image-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
# Configure your AWS credentials and Replicate API key

# Start the application
streamlit run app.py
```

## 🏗️ Project Structure
```
.
├── app.py              # Main application file
├── pages/             # Streamlit pages
│   ├── 1_Generate.py  # Image generation page
│   ├── 2_Batch_Generate.py  # Batch processing
│   └── 3_Fine_Tune.py # Model fine-tuning
├── utils/             # Utility functions
│   ├── config.py      # Configuration management
│   ├── db.py         # Database operations
│   └── sync_time.py  # Time synchronization
├── scripts/           # Helper scripts
└── requirements.txt   # Project dependencies
```

## 📦 Dependencies

Key dependencies include:
- `streamlit`: Web application framework
- `boto3`: AWS SDK for Python
- `replicate`: AI model integration
- `pillow`: Image processing
- `langchain`: AI/ML utilities
- Additional utilities for time sync and file handling

## 🔧 Configuration

### Required Environment Variables
- AWS credentials for DynamoDB access
- Replicate API key for model access
- Additional configuration in `.streamlit/config.toml`

## 🚀 Usage

1. **Generate Images**
   - Navigate to the Generate page
   - Enter your prompt
   - Configure generation parameters
   - Click generate to create images

2. **Batch Generation**
   - Use the Batch Generate page
   - Set up multiple prompts
   - Generate multiple images simultaneously

3. **Fine-Tuning**
   - Access the Fine-Tune page
   - Upload training data
   - Configure and start training process

## 📈 Monitoring

The application provides real-time metrics for:
- Total Images Generated
- Available Models
- Active Users

## 🔐 Security

- AWS IAM integration
- Secure API key management
- Error logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Replicate](https://replicate.com/) for AI model hosting
- [Streamlit](https://streamlit.io/) for the web framework
- All contributors and maintainers

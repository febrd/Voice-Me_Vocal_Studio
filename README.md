# A.I Voice Me

A.I Voice Me is a web based application that allows users to record, upload, and modify audio files using advanced AI-based voice processing techniques. The app includes user authentication, file storage, and a database integration to maintain a history of audio files.

---

## Features

### **1. User Authentication**
- **Register**: Users can register with a username and password.
- **Login**: Secure authentication using hashed passwords.
- **Logout**: Session-based logout functionality.

### **2. Audio Management**
- **Record Audio**: Record audio directly from the microphone.
- **Upload Audio**: Upload `.wav` files for processing.
- **Audio History**: Store and view previously uploaded or recorded audio files.

### **3. Audio Processing**
- **Adjust Pitch and Vibrato**: Modify audio with customizable pitch shifts and vibrato effects.
- **Voice Styles**: Apply pre-trained AI voice styles using Silero models.
- **Download Processed Audio**: Save processed audio files locally.
- **Denoising**: Remove background noise to enhance audio clarity.

---

## Technical Details

### **1. Application Stack**
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**:
  - Python with libraries: `librosa`, `sounddevice`, `scipy`, `soundfile`
  - AI models using `demucs`.
- **Database**: MySQL for storing user and audio history data.
- **Authentication**: Passwords securely hashed using `werkzeug.security`.

### **2. Hardware Support**
- **CPU**: Supports Intel CPU optimizations with `intel_extension_for_pytorch`.
- **GPU**: Utilizes CUDA for AI processing if available.
- **Apple Silicon**: Leverages MPS (Metal Performance Shaders) when running on macOS.

### **3. Folder Structure**
```
project_root/
├── v2.py                 # Main Streamlit application
├── uploads/               # Directory for saving audio files
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── voice_me.sql           # SQL script for database schema
```



---

## Setup Instructions

### **1. Prerequisites**
- Python 3.8+ (Recommended: Python 3.11)
- MySQL Database
- Optional: GPU support (CUDA-enabled GPU)

### **2. Installation**

#### Create a Python Virtual Environment
1. Ensure Python 3.11 is installed on your system.
2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

#### Clone the Repository
```bash
git clone https://github.com/febrd/ai-voice-me.git
cd ai-voice-me.git
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Set Up the Database
1. Create a MySQL database named `ai_voice_me`.
2. Run the provided `database.sql` script to create the necessary tables:
   ```bash
   mysql -u your_username -p your_database < voice_me.sql
   ```

#### Configure Database Credentials
Update the database connection in `app.py`:
```python
def create_connection():
    return mysql.connector.connect(
        host="localhost",  
        user="your_username",  
        password="your_password",  
        database="your_database",  
        collation="utf8mb4_general_ci"
    )
```

#### Create "Uploads" Folder
Ensure the `uploads` folder exists in the project directory:
```bash
mkdir uploads
```

#### Install Required Libraries (if not automatically resolved)
For systems requiring BLAS and LAPACK for audio processing, install them:
```bash
sudo apt install -y libopenblas-dev liblapack-dev gfortran
```

#### Optional: Install Additional Dependencies
For Torch 2.5.0 and Intel optimizations, manually install:
```bash
pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
```
Recommended Torch version:
```bash
pip install torch==2.5.0
```

### **3. Run the Application**
```bash
streamlit run app.py
```

---

## Usage Instructions

### **1. Login or Register**
- Navigate to the sidebar and either log in or create a new account.

### **2. Record, Upload, or Select Audio**
- **Record Audio**: Set the duration and start recording directly from your microphone.
- **Upload Audio**: Upload an existing `.wav` file.
- **Select from History**: Access and reprocess previously uploaded or recorded audio.

### **3. Modify Audio**
- Adjust pitch, vibrato depth, and vibrato rate using the provided sliders.
- Apply AI voice styles using pre-trained Demucs models.

### **4. Save or Download**
- Save processed audio to your local machine using the download button.

---

## API and Libraries

### **1. Key Python Libraries**
- **Streamlit**: Web app framework.
- **librosa**: Audio analysis and manipulation.
- **sounddevice**: Audio recording.
- **scipy**: Signal processing.
- **soundfile**: Audio file I/O.
- **torch**: AI processing.
- **werkzeug.security**: Secure password hashing.
- **mysql-connector-python**: MySQL database connection.

### **2. AI Models**
- [Demucs](https://github.com/facebookresearch/demucs): Pre-trained models for source separation and audio processing.

---

## Future Enhancements
- **Real-time Audio Effects**: Implement live audio processing.
- **More AI Models**: Add support for multilingual TTS and STT.
- **Cloud Storage**: Integrate AWS S3 or Google Cloud Storage for audio file storage.
- **User Profiles**: Allow users to manage settings and preferences.

---

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [Demucs Models](https://github.com/facebookresearch/demucs)
- Streamlit Team for their awesome framework
- OpenAI for inspiring this development journey.


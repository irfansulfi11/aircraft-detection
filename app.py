from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
import base64
import secrets # For generating a secure secret key
import json # For formatting JSON output

app = Flask(__name__)

# Generate a strong, random secret key for session management and security
# IMPORTANT: For production, consider loading this from an environment variable
app.secret_key = secrets.token_hex(32) 

# Configuration for file uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
# Allowed image extensions for SAR images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True) # For static assets like CSS/JS if needed later
os.makedirs('templates', exist_ok=True) # Ensure templates directory exists for index.html

# Define the path to the YOLO model
# It's crucial that 'runs/detect/sar_aircraft_detector/weights/best.pt'
# exists relative to where app.py is executed, or provide an absolute path.
basedir = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(basedir, 'runs', 'detect', 'sar_aircraft_detector', 'weights', 'best.pt')

# Load the YOLO model globally when the application starts
model = None
MODEL_LOADED = False
try:
    model = YOLO(MODEL_PATH)
    MODEL_LOADED = True
    print(f"🚀 YOLO model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure the model file exists and is accessible.")
    MODEL_LOADED = False

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """
    Converts an image file to a base64 string for embedding in HTML.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error encoding image {image_path} to base64: {e}")
        return None

@app.route('/')
def index():
    """
    Renders the main homepage of the SAR aircraft detection system.
    """
    # HTML template content for the frontend
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AeroVision: SAR Aircraft Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Base styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif; /* Modern UI font */
            background: linear-gradient(135deg, #0D0D0D 0%, #1A1A1A 50%, #262626 100%); /* Dark charcoal gradient */
            color: #FFD700; /* Gold for main text */
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start; /* Align content to top */
            padding-top: 40px; /* Space from top */
        }

        .container {
            width: 95%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(26, 26, 26, 0.8); /* Slightly transparent dark */
            border-radius: 20px;
            box-shadow: 0 0 40px rgba(255, 215, 0, 0.2); /* Golden glow */
            border: 1px solid rgba(255, 215, 0, 0.3); /* Subtle golden border */
            backdrop-filter: blur(15px); /* Stronger blur effect */
            animation: fadeIn 1s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 50px;
            position: relative;
            padding-bottom: 20px;
        }

        .header::after { /* Underline effect */
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 150px;
            height: 3px;
            background: linear-gradient(90deg, transparent, #FFD700, transparent); /* Golden underline */
            border-radius: 5px;
        }

        h1 {
            font-family: 'Poppins', sans-serif; /* Futuristic font for title */
            font-size: 3em;
            margin-bottom: 10px;
            color: #FFD700; /* Bright Gold */
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.6); /* Stronger glow */
            letter-spacing: 3px;
            animation: textGlow 2s infinite alternate;
        }

        @keyframes textGlow {
            from { text-shadow: 0 0 10px rgba(255, 215, 0, 0.4), 0 0 20px rgba(255, 215, 0, 0.2); }
            to { text-shadow: 0 0 20px rgba(255, 215, 0, 0.6), 0 0 40px rgba(255, 215, 0, 0.4); }
        }

        .subtitle {
            font-size: 1.3em;
            color: #AAAAAA; /* Lighter gray for contrast on dark */
            margin-bottom: 20px;
        }

        /* Upload Section */
        .upload-section {
            background: rgba(35, 35, 35, 0.6);
            border: 2px dashed #FFD700; /* Neon dashed golden border */
            border-radius: 18px;
            padding: 50px;
            text-align: center;
            margin-bottom: 50px;
            transition: all 0.4s ease-in-out;
            backdrop-filter: blur(12px);
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.15);
            position: relative; /* For pulsing ring */
        }

        .upload-section:hover {
            border-color: #FFEA00;
            background: rgba(255, 215, 0, 0.08);
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 0 35px rgba(255, 215, 0, 0.3);
        }

        .upload-section.dragover {
            border-color: #FFFF00;
            background: rgba(255, 215, 0, 0.15);
            transform: scale(1.03);
        }

        .upload-icon {
            font-size: 5em;
            color: #FFD700; /* Gold */
            margin-bottom: 25px;
            animation: pulse 2s infinite alternate;
        }

        @keyframes pulse {
            from { transform: scale(1); opacity: 0.8; }
            to { transform: scale(1.05); opacity: 1; }
        }

        .upload-section h3 {
            font-family: 'Poppins', sans-serif;
            font-size: 1.8em;
            color: #FFD700; /* Gold */
            margin-bottom: 15px;
        }

        .upload-section p {
            font-size: 1.1em;
            color: #CCCCCC; /* Lighter gray */
            margin: 20px 0;
        }

        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            margin-top: 20px;
        }

        .file-input {
            position: absolute;
            left: -9999px;
        }

        .file-input-button {
            background: linear-gradient(45deg, #FFD700, #FFA500); /* Gold to Orange-Gold gradient */
            color: black; /* Black text on gold button */
            padding: 18px 35px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.2em;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(255, 215, 0, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Poppins', sans-serif;
            position: relative;
            z-index: 1;
        }

        .file-input-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6);
            background: linear-gradient(45deg, #FFE000, #FFC000); /* Brighter gold on hover */
        }

        /* Pulsing ring animation for button */
        .file-input-button.pulsing::before {
            content: '';
            position: absolute;
            top: -5px;
            left: -5px;
            right: -5px;
            bottom: -5px;
            border: 2px solid rgba(255, 215, 0, 0.5); /* Golden pulse ring */
            border-radius: 50px;
            animation: pulseRing 1.5s infinite ease-out;
            z-index: 0;
        }

        @keyframes pulseRing {
            0% { transform: scale(0.8); opacity: 0.7; }
            50% { transform: scale(1.1); opacity: 0.3; }
            100% { transform: scale(0.8); opacity: 0.7; }
        }

        /* Progress Bar */
        .progress-bar {
            width: 80%;
            height: 6px;
            background: #333333; /* Darker background for progress bar */
            border-radius: 3px;
            overflow: hidden;
            margin: 30px auto 0;
            display: none;
            box-shadow: inset 0 0 5px rgba(255, 215, 0, 0.2); /* Golden shadow */
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, rgba(255, 215, 0, 0.2) 0%, #FFD700 50%, rgba(255, 215, 0, 0.2) 100%); /* Golden gradient fill */
            background-size: 200% 100%;
            animation: progressShimmer 1.5s infinite linear;
            border-radius: 3px;
        }

        @keyframes progressShimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* Error Message */
        .error-message {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(239, 68, 68, 0.4);
            margin-top: 30px;
            display: none;
            font-size: 1.1em;
            text-align: center;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
        }

        /* Results Section */
        .results-section {
            display: none;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); /* Adjusted for smaller images */
            gap: 40px;
            margin-top: 50px;
            padding: 30px;
            background: rgba(25, 35, 45, 0.6); /* Dark transparent */
            border-radius: 18px;
            box-shadow: 0 0 30px rgba(255, 215, 0, 0.1); /* Golden shadow */
            border: 1px solid rgba(255, 215, 0, 0.2); /* Golden border */
        }

        .image-container {
            background: rgba(35, 45, 55, 0.8); /* Dark transparent */
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 110, 120, 0.4); /* Slightly lighter dark border */
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.1); /* Golden shadow */
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .image-container h3 {
            color: #FFD700; /* Gold */
            margin-bottom: 20px;
            font-size: 1.6em;
            font-family: 'Poppins', sans-serif;
            text-align: center;
        }

        .image-container img {
            width: 100%;
            max-width: 400px; /* Further reduced image size */
            height: auto;
            border-radius: 12px;
            border: 3px solid rgba(255, 215, 0, 0.5); /* Golden border */
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.3); /* Golden shadow */
            transition: transform 0.3s ease;
        }

        .image-container img:hover {
            transform: scale(1.02);
        }

        .download-button {
            background: linear-gradient(45deg, #FFD700, #FFA500); /* Gold to Orange-Gold gradient */
            color: black; /* Black text on button */
            padding: 12px 25px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin-top: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(255, 215, 0, 0.3); /* Golden shadow */
            font-family: 'Inter', sans-serif;
        }

        .download-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 215, 0, 0.5);
        }

        /* Stats Panel */
        .stats-panel {
            display: none; /* Hidden by default */
            background: rgba(25, 35, 45, 0.8); /* Dark transparent */
            border-radius: 20px;
            padding: 35px;
            margin-top: 50px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 215, 0, 0.3); /* Golden border */
            box-shadow: 0 0 40px rgba(255, 215, 0, 0.2); /* Golden shadow */
            animation: slideUp 0.8s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stats-panel h3 {
            color: #FFD700; /* Gold */
            margin-bottom: 30px;
            font-size: 1.8em;
            text-align: center;
            font-family: 'Poppins', sans-serif;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.4); /* Golden shadow */
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
        }

        .stat-item {
            text-align: center;
            padding: 20px;
            background: rgba(255, 215, 0, 0.1); /* Transparent gold background */
            border-radius: 15px;
            border: 1px solid rgba(255, 215, 0, 0.3); /* Golden border */
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.1); /* Golden shadow */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stat-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 25px rgba(255, 215, 0, 0.3);
        }

        .stat-number {
            font-size: 2.8em;
            font-weight: bold;
            color: #FFD700; /* Brighter Gold */
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.7); /* Golden shadow */
            font-family: 'Poppins', sans-serif;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #AAAAAA; /* Lighter gray */
            font-size: 1em;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }

        /* Threat Level Specific Styles */
        #threatLevel.HIGH { color: #ff6b6b; text-shadow: 0 0 10px #ff6b6b; } /* Red */
        #threatLevel.MEDIUM { color: #ffd166; text-shadow: 0 0 10px #ffd166; } /* Orange */
        #threatLevel.LOW { color: #06d6a0; text-shadow: 0 0 10px #06d6a0; } /* Green */

        /* JSON Viewer */
        /* Removed .json-viewer-container as per request */

        /* Loading Overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95); /* Pure black transparent */
            display: none;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            animation: overlayFadeIn 0.5s ease-out;
        }

        @keyframes overlayFadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .loading-spinner {
            width: 80px;
            height: 80px;
            border: 5px solid #333333; /* Dark gray border */
            border-top: 5px solid #FFD700; /* Golden spinner */
            border-radius: 50%;
            animation: spin 1.2s linear infinite, glow 1.5s infinite alternate;
            margin-bottom: 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes glow {
            from { box-shadow: 0 0 10px #FFD700; }
            to { box-shadow: 0 0 25px #FFD700, 0 0 40px rgba(255, 215, 0, 0.5); }
        }

        .loading-overlay p {
            color: #FFD700; /* Gold */
            font-size: 1.5em;
            font-family: 'Poppins', sans-serif;
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.5); /* Golden shadow */
        }

        /* Radar Background Animation */
        .radar-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1; /* Above aircraft particles */
            overflow: hidden;
        }

        .radar-sweep {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 800px; /* Larger radar */
            height: 800px;
            border: 3px solid rgba(255, 215, 0, 0.1); /* Subtle golden border */
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: radar-pulse 6s ease-in-out infinite, rotate 20s linear infinite; /* Added rotation */
            box-shadow: 0 0 50px rgba(255, 215, 0, 0.1); /* Golden shadow */
        }

        .radar-sweep::before { /* Inner sweep line */
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            width: 2px;
            height: 100%;
            background: linear-gradient(to bottom, rgba(255, 215, 0, 0.8), transparent); /* Golden sweep */
            transform-origin: bottom;
            animation: sweep 6s linear infinite;
        }

        @keyframes radar-pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(0.9); opacity: 0.2; }
            50% { transform: translate(-50%, -50%) scale(1.1); opacity: 0.05; }
        }

        @keyframes rotate {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }

        /* Blurry Aircraft Particles Background */
        .aircraft-particles-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -2; /* Behind radar */
            overflow: hidden;
        }

        .aircraft-particles-bg .particle {
            position: absolute;
            background-color: rgba(255, 215, 0, 0.1); /* Subtle golden hue */
            border-radius: 50%; /* Circular blob */
            filter: blur(25px); /* Heavy blur */
            opacity: 0; /* Start hidden */
            animation-iteration-count: infinite;
            animation-timing-function: linear;
            transform: translate(-50%, -50%); /* Center origin for transforms */
        }

        /* Individual particle sizes, positions, and animations */
        .aircraft-particles-bg .particle-1 {
            width: 100px;
            height: 60px;
            top: 10%;
            left: 5%;
            animation: floatParticle 20s infinite alternate, fadeInOut 10s infinite;
            animation-delay: 0s;
            background-color: rgba(255, 215, 0, 0.15);
            border-radius: 40%; /* More elliptical */
        }
        .aircraft-particles-bg .particle-2 {
            width: 80px;
            height: 50px;
            top: 80%;
            left: 90%;
            animation: floatParticle 25s infinite alternate reverse, fadeInOut 12s infinite;
            animation-delay: 2s;
            background-color: rgba(255, 215, 0, 0.1);
            border-radius: 60%;
        }
        .aircraft-particles-bg .particle-3 {
            width: 120px;
            height: 70px;
            top: 40%;
            left: 70%;
            animation: floatParticle 18s infinite alternate, fadeInOut 9s infinite;
            animation-delay: 4s;
            background-color: rgba(255, 215, 0, 0.2);
            border-radius: 30%;
        }
        .aircraft-particles-bg .particle-4 {
            width: 70px;
            height: 40px;
            top: 25%;
            left: 30%;
            animation: floatParticle 22s infinite alternate reverse, fadeInOut 11s infinite;
            animation-delay: 6s;
            background-color: rgba(255, 215, 0, 0.12);
            border-radius: 55%;
        }
        .aircraft-particles-bg .particle-5 {
            width: 90px;
            height: 55px;
            top: 65%;
            left: 15%;
            animation: floatParticle 28s infinite alternate, fadeInOut 13s infinite;
            animation-delay: 8s;
            background-color: rgba(255, 215, 0, 0.18);
            border-radius: 45%;
        }

        @keyframes floatParticle {
            0% { transform: translate(-50%, -50%) translateX(0) translateY(0) rotate(0deg); }
            25% { transform: translate(-50%, -50%) translateX(10vw) translateY(5vh) rotate(10deg); }
            50% { transform: translate(-50%, -50%) translateX(0) translateY(10vh) rotate(20deg); }
            75% { transform: translate(-50%, -50%) translateX(-10vw) translateY(5vh) rotate(10deg); }
            100% { transform: translate(-50%, -50%) translateX(0) translateY(0) rotate(0deg); }
        }

        @keyframes fadeInOut {
            0%, 100% { opacity: 0; }
            20%, 80% { opacity: 0.3; }
        }

        /* Responsive adjustments */
        @media (max-width: 1024px) {
            .container {
                width: 98%;
                padding: 15px;
            }
            h1 { font-size: 2.5em; }
            .subtitle { font-size: 1.1em; }
            .upload-section { padding: 40px; }
            .upload-icon { font-size: 4em; }
            .file-input-button { padding: 15px 30px; font-size: 1.1em; }
            .results-section { grid-template-columns: 1fr; gap: 30px; }
            .stats-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
            .stat-number { font-size: 2.2em; }
            .loading-spinner { width: 60px; height: 60px; }
            .loading-overlay p { font-size: 1.2em; }
            .radar-sweep { width: 600px; height: 600px; }
        }

        @media (max-width: 768px) {
            h1 { font-size: 2em; letter-spacing: 2px; }
            .subtitle { font-size: 1em; }
            .upload-section { padding: 30px; }
            .upload-icon { font-size: 3.5em; }
            .file-input-button { padding: 12px 25px; font-size: 1em; }
            .stats-grid { grid-template-columns: 1fr; }
            .stat-number { font-size: 2.2em; }
            .image-container img { max-width: 100%; }
            .radar-sweep { width: 400px; height: 400px; }
        }

        @media (max-width: 480px) {
            .container { padding: 10px; }
            h1 { font-size: 1.8em; letter-spacing: 1px; }
            .subtitle { font-size: 0.9em; }
            .upload-section { padding: 20px; }
            .upload-icon { font-size: 3em; }
            .file-input-button { padding: 10px 20px; font-size: 0.9em; }
            .stats-panel h3 { font-size: 1.5em; }
            .stat-item { padding: 15px; }
            .stat-number { font-size: 1.8em; }
            .loading-spinner { width: 40px; height: 40px; }
            .loading-overlay p { font-size: 1em; }
            .radar-sweep { width: 300px; height: 300px; }
        }
    </style>
</head>
<body>
    <!-- Blurry Aircraft Background Animation -->
    <div class="aircraft-particles-bg">
        <div class="particle particle-1"></div>
        <div class="particle particle-2"></div>
        <div class="particle particle-3"></div>
        <div class="particle particle-4"></div>
        <div class="particle particle-5"></div>
    </div>

    <!-- Background Radar Animation -->
    <div class="radar-bg">
        <div class="radar-sweep"></div>
    </div>

    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>AeroVision</h1>
            <p class="subtitle">Advanced SAR Aircraft Detection & Analysis</p>
        </div>

        <!-- Image Upload Section -->
        <div class="upload-section" id="uploadSection">
            <div class="upload-icon">✈️</div>
            <h3>Upload SAR Image for Detection</h3>
            <p>Drag and drop your SAR image file here, or click the button below to select.</p>
            <div class="file-input-wrapper">
                <input type="file" id="fileInput" class="file-input" accept=".png,.jpg,.jpeg,.bmp,.tiff,.tif">
                <button class="file-input-button" id="uploadButton">Select SAR Image</button>
            </div>
            <div class="progress-bar" id="progressBar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>

        <!-- Error Message Display -->
        <div class="error-message" id="errorMessage"></div>

        <!-- Results Display Section -->
        <div class="results-section" id="resultsSection">
            <div class="image-container">
                <h3>Original Image</h3>
                <img id="originalImage" src="" alt="Original SAR Image" onerror="this.onerror=null; this.src='https://placehold.co/400x400/333/FFF?text=Image+Load+Error'; console.error('Error loading original image');">
                <button class="download-button" id="downloadOriginal">Download Original</button>
            </div>
            <div class="image-container">
                <h3>Detected Output</h3>
                <img id="resultImage" src="" alt="Detection Results" onerror="this.onerror=null; this.src='https://placehold.co/400x400/333/FFF?text=Image+Load+Error'; console.error('Error loading result image');">
                <button class="download-button" id="downloadResult">Download Result</button>
            </div>
        </div>

        <!-- Statistics Panel -->
        <div class="stats-panel" id="statsPanel">
            <h3>Analysis Summary</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number" id="detectionCount">0</div>
                    <div class="stat-label">Aircraft Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="confidence">0%</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="processingTime">0s</div>
                    <div class="stat-label">Inference Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="threatLevel">LOW</div>
                    <div class="stat-label">Threat Level</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div>
            <div class="loading-spinner"></div>
            <p>Processing Image...</p>
            <p style="font-size:0.9em; margin-top: 10px; color: #AAAAAA;">Analyzing SAR data for aircraft signatures.</p>
        </div>
    </div>

    <script>
        // Get DOM elements
        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const errorMessage = document.getElementById('errorMessage');
        const resultsSection = document.getElementById('resultsSection');
        const statsPanel = document.getElementById('statsPanel');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const originalImage = document.getElementById('originalImage');
        const resultImage = document.getElementById('resultImage');
        const downloadOriginalButton = document.getElementById('downloadOriginal');
        const downloadResultButton = document.getElementById('downloadResult');
        const uploadButton = document.getElementById('uploadButton'); // Reference to the upload button
        
        // Removed JSON viewer related elements
        // const jsonViewerContainer = document.getElementById('jsonViewerContainer');
        // const toggleJsonViewerButton = document.getElementById('toggleJsonViewer');
        // const jsonOutput = document.getElementById('jsonOutput');

        let currentOriginalFileName = ''; // To store original filename for download
        let currentResultFileName = '';   // To store result filename for download
        let currentDetectionData = null;  // To store raw detection data for JSON viewer (still kept for potential future use or debugging)

        // Event listener for the custom file input button
        uploadButton.addEventListener('click', () => {
            fileInput.click(); // Trigger the hidden file input click
        });

        // Drag and drop event listeners for the upload section
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault(); // Prevent default drag behavior
            uploadSection.classList.add('dragover'); // Add 'dragover' class for visual feedback
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault(); // Prevent default drag behavior
            uploadSection.classList.remove('dragover'); // Remove 'dragover' class
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault(); // Prevent default drop behavior (e.g., opening file in browser)
            uploadSection.classList.remove('dragover'); // Remove 'dragover' class
            const files = e.dataTransfer.files; // Get dropped files
            if (files.length > 0) {
                handleFile(files[0]); // Process the first dropped file
            }
        });

        // Event listener for when a file is selected via the input
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]); // Process the selected file
            }
        });

        // Function to handle the file processing (upload and inference)
        function handleFile(file) {
            console.log('handleFile: File selected', file.name, file.type);
            // Validate file type
            const allowedTypes = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff'];
            if (!allowedTypes.includes(file.type)) {
                showError('Invalid file type. Please select a SAR image (PNG, JPG, JPEG, BMP, TIFF, TIF).');
                return;
            }

            // Prepare form data for upload
            const formData = new FormData();
            formData.append('file', file);

            // Reset UI elements and show loading state
            errorMessage.style.display = 'none';
            resultsSection.style.display = 'none';
            statsPanel.style.display = 'none';
            // Removed JSON viewer related display resets
            // jsonViewerContainer.style.display = 'none'; 
            // jsonOutput.style.display = 'none'; 
            // toggleJsonViewerButton.textContent = 'Show Raw JSON'; 

            loadingOverlay.style.display = 'flex';
            progressBar.style.display = 'block';
            progressFill.style.width = '0%'; // Reset progress bar
            uploadButton.classList.add('pulsing'); // Add pulsing animation to button

            // Simulate progress for better UX during upload (actual progress is hard to track with fetch)
            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 90) { // Cap simulated progress before server response
                    progress += Math.random() * 5; // Simulate varying progress speed
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                }
            }, 150); // Update every 150ms

            const startTime = Date.now(); // Record start time for processing time calculation

            // Send the file to the backend
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('handleFile: Fetch response received', response.status);
                // Check if the response is OK (status 200-299)
                if (!response.ok) {
                    // If not OK, parse error message from server or throw generic error
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || `Server responded with status ${response.status}`);
                    });
                }
                return response.json(); // Parse JSON response
            })
            .then(data => {
                console.log('handleFile: JSON data received', data);
                clearInterval(progressInterval); // Stop simulated progress
                progressFill.style.width = '100%'; // Set to 100% on success
                uploadButton.classList.remove('pulsing'); // Remove pulsing animation

                // Small delay to show 100% progress before hiding loading overlay
                setTimeout(() => {
                    loadingOverlay.style.display = 'none';
                    progressBar.style.display = 'none';
                }, 500); // Half-second delay

                if (data.success) {
                    currentDetectionData = data.detections; // Store raw data
                    showResults(data, Date.now() - startTime); // Display results
                } else {
                    showError(data.error || 'An unknown error occurred during processing.');
                }
            })
            .catch(error => {
                console.error('handleFile: Fetch error caught', error);
                clearInterval(progressInterval); // Stop simulated progress on error
                uploadButton.classList.remove('pulsing'); // Remove pulsing animation
                loadingOverlay.style.display = 'none';
                progressBar.style.display = 'none';
                showError('Network error or processing failed: ' + error.message);
            });
        }

        // Function to display detection results and statistics
        function showResults(data, processingTime) {
            console.log('showResults: Displaying results with data', data);
            // Set image sources using base64 data
            originalImage.src = 'data:image/jpeg;base64,' + data.original_image;
            resultImage.src = 'data:image/jpeg;base64,' + data.result_image;
            
            // Store filenames for download buttons
            currentOriginalFileName = `original_${data.filename}`;
            currentResultFileName = `detected_${data.filename}`; // More descriptive name

            // Show result sections
            resultsSection.style.display = 'grid';
            statsPanel.style.display = 'block';
            // Removed JSON viewer related display show
            // jsonViewerContainer.style.display = 'block'; 

            // Update statistics in the control panel
            document.getElementById('detectionCount').textContent = data.detection_count;
            
            // Calculate and display average confidence
            const avgConfidence = data.detections.length > 0 
                ? data.detections.reduce((sum, det) => sum + det.confidence, 0) / data.detections.length
                : 0;
            document.getElementById('confidence').textContent = `${Math.round(avgConfidence * 100)}%`;
            
            // Display processing time
            document.getElementById('processingTime').textContent = `${(processingTime / 1000).toFixed(2)}s`;
            
            // Determine and display threat level based on detection count
            const threatLevelElement = document.getElementById('threatLevel');
            let threatLevel = 'LOW';
            if (data.detection_count >= 5) {
                threatLevel = 'HIGH';
            } else if (data.detection_count >= 2) {
                threatLevel = 'MEDIUM';
            }
            threatLevelElement.textContent = threatLevel;
            // Add/remove classes for threat level styling
            threatLevelElement.classList.remove('LOW', 'MEDIUM', 'HIGH');
            threatLevelElement.classList.add(threatLevel);
            
            // Scroll to results for better UX
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // Function to display error messages
        function showError(message) {
            console.error('showError:', message);
            errorMessage.textContent = `ERROR: ${message}`;
            errorMessage.style.display = 'block';
            // Ensure other sections are hidden on error
            progressBar.style.display = 'none';
            loadingOverlay.style.display = 'none';
            resultsSection.style.display = 'none';
            statsPanel.style.display = 'none';
            // Removed JSON viewer related display hide
            // jsonViewerContainer.style.display = 'none';
            uploadButton.classList.remove('pulsing'); // Remove pulsing animation
        }

        // --- Download Functionality ---
        function downloadImage(imgElement, filename) {
            console.log('downloadImage: Attempting to download', filename);
            const link = document.createElement('a');
            link.href = imgElement.src; // Use the base64 data URL
            link.download = filename; // Set the desired filename for download
            document.body.appendChild(link); // Append to body (required for Firefox)
            link.click(); // Programmatically click the link to trigger download
            document.body.removeChild(link); // Clean up the link element
        }

        downloadOriginalButton.addEventListener('click', () => {
            if (originalImage.src && originalImage.src !== window.location.href) { // Ensure image is loaded
                downloadImage(originalImage, currentOriginalFileName);
            } else {
                showError('No original image available for download.');
            }
        });

        downloadResultButton.addEventListener('click', () => {
            if (resultImage.src && resultImage.src !== window.location.href) { // Ensure image is loaded
                downloadImage(resultImage, currentResultFileName);
            } else {
                showError('No result image available for download.');
            }
        });

        // --- JSON Viewer Toggle Functionality ---
        // Removed JSON Viewer Toggle Functionality as per request
        /*
        toggleJsonViewerButton.addEventListener('click', () => {
            if (jsonOutput.style.display === 'none') {
                if (currentDetectionData && currentDetectionData.length > 0) { // Check if data exists AND is not empty
                    jsonOutput.textContent = JSON.stringify(currentDetectionData, null, 2); // Pretty print JSON
                    jsonOutput.style.display = 'block';
                    toggleJsonViewerButton.textContent = 'Hide Raw JSON';
                } else {
                    // Explicit message if no detections or data is null/empty
                    jsonOutput.textContent = 'No raw detection data available for this image or no aircraft were detected.';
                    jsonOutput.style.display = 'block';
                    toggleJsonViewerButton.textContent = 'Hide Raw JSON';
                }
            } else {
                jsonOutput.style.display = 'none';
                toggleJsonViewerButton.textContent = 'Show Raw JSON';
            }
        });
        */
    </script>
</body>
</html>'''
    
    # Write the HTML content to index.html in the templates directory
    # This ensures the template is always up-to-date when the Flask app runs
    try:
        with open(os.path.join(basedir, 'templates', 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("✅ index.html template created/updated successfully.")
    except IOError as e:
        print(f"❌ Error writing index.html: {e}")

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file uploads, performs aircraft detection, and returns results.
    """
    if not MODEL_LOADED:
        return jsonify({'success': False, 'error': 'AI model not loaded. System is offline.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request. Please select an image.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file. Please choose an image to upload.'}), 400
    
    if file and allowed_file(file.filename):
        # Securely save the uploaded file with a timestamp to prevent overwrites
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # Add microseconds for higher uniqueness
        filename_with_timestamp = f"{timestamp}_{filename}" 
        filepath = os.path.join(UPLOAD_FOLDER, filename_with_timestamp)

        try:
            file.save(filepath)
            print(f"✅ File saved: {filepath}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to save uploaded file: {str(e)}'}), 500
        
        try:
            # Run YOLO inference on the uploaded image
            # conf: confidence threshold, iou: Intersection Over Union threshold for NMS
            results = model(filepath, conf=0.25, iou=0.7, save=False) # Adjusted conf for potentially better detection
            
            # Get the image with bounding boxes drawn by YOLO
            # results[0].plot() returns a NumPy array (OpenCV BGR format)
            result_img_np = results[0].plot() 
            
            # Define paths for saving the result image
            name, ext = os.path.splitext(filename_with_timestamp)
            # Save results as JPEG for consistent web display and smaller size
            result_filename = f"detected_{name}.jpg" # Changed to 'detected_' for clarity
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            
            # Save the processed image using OpenCV
            cv2.imwrite(result_path, result_img_np)
            print(f"✅ Result image saved: {result_path}")
            
            # Extract detection information
            detections = []
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'confidence': float(box.conf[0]), # Confidence score
                        'class_id': int(box.cls[0]),     # Detected class ID
                        'bbox': box.xyxy[0].tolist()     # Bounding box coordinates [x1, y1, x2, y2]
                    }
                    detections.append(detection)
            
            # Encode both original and result images to base64 for direct display in HTML
            original_b64 = encode_image_to_base64(filepath)
            result_b64 = encode_image_to_base64(result_path)

            if original_b64 is None or result_b64 is None:
                return jsonify({'success': False, 'error': 'Failed to encode images to base64.'}), 500
            
            # Return JSON response with detection data and base64 images
            return jsonify({
                'success': True,
                'original_image': original_b64,
                'result_image': result_b64,
                'detections': detections,
                'detection_count': len(detections),
                'filename': filename_with_timestamp # Pass original filename for client-side download naming
            })
            
        except Exception as e:
            # Log the full traceback for detailed server-side debugging
            import traceback
            traceback.print_exc() 
            return jsonify({'success': False, 'error': f'Detection process failed: {str(e)}. Please try another image.'}), 500
    
    return jsonify({'success': False, 'error': 'Invalid file type. Please upload a supported image format.'}), 400

# Routes to serve uploaded and result images directly (though base64 is used for display)
@app.route('/results/<filename>')
def get_result(filename):
    """Serves result images from the RESULTS_FOLDER."""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    """Serves original uploaded images from the UPLOAD_FOLDER."""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    print("🚀 SAR Aircraft Detection System Starting...")
    print("📡 Access the application at: http://localhost:5000")
    # Run the Flask application
    # debug=True provides auto-reloading and a debugger, but should be False in production
    app.run(debug=True, host='0.0.0.0', port=5000)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import threading
import os
import time
from datetime import datetime
import asyncio
import uvicorn
import json
import urllib.parse
import urllib.request
from io import BytesIO
from models import UnifiedSignLanguageDetector
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

app = FastAPI(title="Unified Sign Language Detection API", version="3.0.0")

# Configure Gemini if API key is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and HAS_GENAI:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

# Add CORS middleware for React app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DETECTED_TEXT_FILE = "detected_text.txt"
DETECTION_LOG_FILE = "sign_language_detections.txt"

# Global variables
detection_active = False
detection_thread = None
detector = UnifiedSignLanguageDetector()
detection_status = {
    "active": False,
    "language": None,
    "word_buffer": "",
    "sentence_buffer": "",
    "last_detected_char": "?",
    "confidence": 0.0,
    "session_id": None,
    "completed": False,
    "final_sentence": "",
    "detection_progress": 0.0,
    "auto_detection_enabled": True
}

# Request models
class DetectionRequest(BaseModel):
    language: str = "ASL"  # ASL, ISL, or TSL/TAMIL

def log_detection(detection_data):
    """Log detection to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log to text file
    with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {detection_data}\n")

def process_camera_opencv(language="ASL"):
    """Unified camera processing for ASL, ISL, and Tamil"""
    global detection_active, detection_status, detector
    
    # Initialize detector for specified language
    if not detector.initialize(language):
        print(f"Failed to load {language} model")
        return False
    
    detector.cap = cv2.VideoCapture(0)
    if not detector.cap.isOpened():
        print("Cannot access the camera")
        return False
    
    # Set frame dimensions based on language
    frame_width = 1280 if language in ["ISL", "TSL", "TAMIL"] else 1920
    frame_height = 720 if language in ["ISL", "TSL", "TAMIL"] else 1080
    detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_name = f'{language} Character Detection - Unified API'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    # Get language-specific instructions
    instructions = detector.get_instructions()
    
    word_buffer = ""
    sentence_buffer = ""
    detector.session_start_time = datetime.now().isoformat()
    detection_active = True
    
    # Initialize log file
    with open(DETECTION_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== {language} Character Detection Session Started: {detector.session_start_time} ===\n")
    
    # Reset detection status
    detection_status.update({
        "active": True,
        "language": language,
        "word_buffer": "",
        "sentence_buffer": "",
        "last_detected_char": "?",
        "confidence": 0.0,
        "session_id": detector.session_start_time,
        "completed": False,
        "final_sentence": "",
        "detection_progress": 0.0,
        "auto_detection_enabled": True  # All languages now use auto-detection
    })
    
    while detection_active:
        ret, frame = detector.cap.read()
        if not ret:
            print("Failed to read from camera")
            break
        
        # Mirror the frame for more intuitive interaction (except ASL)
        if language in ["ISL", "TSL", "TAMIL"]:
            frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Add guidance box
        box_size = int(min(width, height) * 0.7)
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(frame, 
                      (center_x - box_size//2, center_y - box_size//2),
                      (center_x + box_size//2, center_y + box_size//2), 
                      (0, 255, 0), 3)
        
        # Add overlay for status info based on language
        if language in ["ISL", "TSL", "TAMIL"]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            font_scale = 0.6
            line_spacing = 25
            start_y = 30
        else:
            # ASL style overlay
            overlay = np.zeros((200, width, 3), dtype=np.uint8)
            frame[0:200, 0:width] = cv2.addWeighted(overlay, 0.5, frame[0:200, 0:width], 0.5, 0)
            font_scale = 0.8
            line_spacing = 40
            start_y = 40
        
        # Instructions overlay
        for idx, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + idx * line_spacing), 
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
        
        # Process frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.current_detector.hands.process(frame_rgb)
        
        # Get detection results
        detected_char, confidence, extra_info = detector.process_frame(frame)
        
        # Draw hand landmarks
        detector.draw_landmarks(frame, results)
        
        # Display detected character and confidence
        if language in ["ISL", "TSL", "TAMIL"]:
            cv2.putText(frame, f"Detected: {detected_char} ({confidence:.1f}%)", 
                        (10, height - 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, 
                        (0, 255, 0) if detected_char != '?' else (100, 100, 100), 2)
        else:
            if detected_char != '?':
                cv2.putText(frame, f"Detected: {detected_char}", (10, 250), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Handle auto-detection progress bar for all languages
        if extra_info and extra_info.get("detection_progress", 0) > 0:
            progress = extra_info["detection_progress"]
            bar_width = 200
            bar_height = 20
            bar_x = width - 220
            bar_y = height - 100
            
            # Draw progress bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            # Show progress text
            progress_text = f"Detecting '{detected_char}': {int(progress * 100)}%"
            cv2.putText(frame, progress_text, (bar_x - 50, bar_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display current word and sentence
        word_y = height - 50 if language in ["ISL", "TSL", "TAMIL"] else 300
        sentence_y = height - 20 if language in ["ISL", "TSL", "TAMIL"] else 350
        font_size = 0.7 if language in ["ISL", "TSL", "TAMIL"] else 1
        
        cv2.putText(frame, f"Word: {word_buffer}", (10, word_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence_buffer}", (10, sentence_y), 
                    cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 255, 255), 2)
        
        # Handle automatic detection for all languages
        if extra_info and extra_info.get("should_auto_detect", False):
            word_buffer += detected_char
            log_detection(f"Auto-added '{detected_char}' to word. Current word: '{word_buffer}'")
            
            # Flash effect
            flash = np.ones_like(frame) * 255
            cv2.imshow(window_name, flash)
            cv2.waitKey(50)
        
        # Update global status for API access
        detection_status.update({
            "active": True,
            "language": language,
            "word_buffer": word_buffer,
            "sentence_buffer": sentence_buffer,
            "last_detected_char": detected_char,
            "confidence": confidence,
            "session_id": detector.session_start_time,
            "completed": False,
            "final_sentence": "",
            "detection_progress": extra_info.get("detection_progress", 0.0) if extra_info else 0.0,
            "auto_detection_enabled": True
        })
        
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER - Add word to sentence
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                
                log_detection(f"Added word '{word_buffer}' to sentence. Current sentence: '{sentence_buffer}'")
                word_buffer = ""
                
        elif key == ord('q'):  # Q - Finish sentence detection
            # Add the last word if any
            if word_buffer.strip():
                if sentence_buffer:
                    sentence_buffer += ' ' + word_buffer.strip()
                else:
                    sentence_buffer = word_buffer.strip()
                word_buffer = ""
            
            # Save sentence to text file
            with open(DETECTED_TEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(sentence_buffer)
            
            log_detection(f"Session completed. Final sentence: '{sentence_buffer}'")
            
            # Update final status
            detection_status.update({
                "active": False,
                "language": language,
                "word_buffer": word_buffer,
                "sentence_buffer": sentence_buffer,
                "last_detected_char": detected_char,
                "confidence": confidence,
                "completed": True,
                "final_sentence": sentence_buffer,
                "detection_progress": 0.0
            })
            
            detection_active = False
            break
    
    # Cleanup
    detector.cleanup()
    
    # Log session end
    with open(DETECTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    return True

@app.post("/start_detection")
async def start_detection(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Start sign language character detection with OpenCV popup window"""
    global detection_thread, detection_active
    
    # Validate language
    language = request.language.upper()
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    if detection_active:
        return JSONResponse(
            status_code=400,
            content={"message": f"{language} detection is already running"}
        )
    
    # Start detection in background thread
    detection_thread = threading.Thread(
        target=process_camera_opencv,
        args=(language,),
        daemon=True
    )
    detection_thread.start()
    
    # Wait a moment for detection to start
    await asyncio.sleep(1)
    
    # Get detector info
    detector_info = detector.get_detector_info()
    
    return JSONResponse(
        content={
            "message": f"{language} character detection started successfully",
            "language": language,
            "instructions": detector.get_instructions(),
            "session_id": detector.session_start_time,
            "status_url": "/detection_status",
            "detector_info": detector_info
        }
    )

@app.post("/stop_detection")
async def stop_detection():
    """Stop sign language character detection"""
    global detection_active, detection_thread
    
    if not detection_active:
        return JSONResponse(
            status_code=400,
            content={"message": "Detection is not running"}
        )
    
    detection_active = False
    
    # Wait for thread to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=5)
    
    return JSONResponse(content={"message": "Character detection stopped successfully"})

@app.get("/detection_status")
async def get_detection_status():
    """Get current character detection status and text output"""
    global detection_status
    
    return JSONResponse(content=detection_status)

@app.get("/get_detected_text")
async def get_detected_text():
    """Get the final detected sentence as text"""
    try:
        if os.path.exists(DETECTED_TEXT_FILE):
            with open(DETECTED_TEXT_FILE, 'r', encoding='utf-8') as f:
                detected_sentence = f.read().strip()
            
            return JSONResponse(content={
                "text": detected_sentence,
                "available": True,
                "language": detection_status.get("language", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return JSONResponse(content={
                "text": "",
                "available": False,
                "language": detection_status.get("language", "Unknown"),
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error reading detected text: {str(e)}"}
        )

@app.get("/detection_log")
async def get_detection_log():
    """Get sign language character detection log"""
    try:
        with open(DETECTION_LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
        return JSONResponse(content={"log": log_content})
    except FileNotFoundError:
        return JSONResponse(content={"log": ""})

@app.delete("/clear_session")
async def clear_session():
    """Clear current session data"""
    global detection_status
    
    try:
        # Clear files
        if os.path.exists(DETECTED_TEXT_FILE):
            os.remove(DETECTED_TEXT_FILE)
        
        # Reset status
        detection_status = {
            "active": False,
            "language": None,
            "word_buffer": "",
            "sentence_buffer": "",
            "last_detected_char": "?",
            "confidence": 0.0,
            "session_id": None,
            "completed": False,
            "final_sentence": "",
            "detection_progress": 0.0,
            "auto_detection_enabled": True
        }
        
        return JSONResponse(content={"message": "Session cleared successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error clearing session: {str(e)}"}
        )

@app.get("/model_info/{language}")
async def get_model_info(language: str):
    """Get information about the specified language model"""
    language = language.upper()
    
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    # Temporarily initialize detector to get info
    temp_detector = UnifiedSignLanguageDetector()
    if temp_detector.initialize(language):
        info = temp_detector.get_detector_info()
        temp_detector.cleanup()
        return JSONResponse(content=info)
    else:
        return JSONResponse(
            status_code=500,
            content={"message": f"Could not load {language} model information"}
        )

@app.get("/supported_languages")
async def get_supported_languages():
    """Get list of supported sign languages"""
    return JSONResponse(content={
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "ASL": {
            "name": "American Sign Language",
            "model_type": "Random Forest Classifier",
            "capture_method": "Automatic (hold for 1.5-2 seconds)",
            "hands_required": "Single hand"
        },
        "ISL": {
            "name": "Indian Sign Language", 
            "model_type": "LSTM Neural Network",
            "capture_method": "Automatic (hold for 2-3 seconds)",
            "hands_required": "Both hands supported"
        },
        "TSL": {
            "name": "Tamil Sign Language",
            "model_type": "LSTM Neural Network", 
            "capture_method": "Automatic (hold for 2-3 seconds)",
            "hands_required": "Both hands supported"
        },
        "TAMIL": {
            "name": "Tamil Sign Language (alias for TSL)",
            "model_type": "LSTM Neural Network",
            "capture_method": "Automatic (hold for 2-3 seconds)", 
            "hands_required": "Both hands supported"
        }
    })

@app.get("/word_formation/{language}")
async def get_word_formation_exercises(language: str):
    """Get available word formation exercises for the specified language"""
    language = language.upper()
    
    if language not in ["ASL", "ISL", "TSL", "TAMIL"]:
        raise HTTPException(status_code=400, detail="Language must be 'ASL', 'ISL', 'TSL', or 'TAMIL'")
    
    # Common words that can be formed (you can expand this based on your models)
    common_words = {
        "ASL": {
            "HELLO": ["H", "E", "L", "L", "O"],
            "WORLD": ["W", "O", "R", "L", "D"],
            "LOVE": ["L", "O", "V", "E"],
            "PEACE": ["P", "E", "A", "C", "E"]
        },
        "ISL": {
            "HELLO": ["H", "E", "L", "L", "O"],
            "NAMASTE": ["N", "A", "M", "A", "S", "T", "E"],
            "INDIA": ["I", "N", "D", "I", "A"]
        },
        "TSL": {
            "VANAKKAM": ["வ", "ண", "க", "க", "ம"],
            "NANDRI": ["ந", "ன", "ற", "ி"],
            "TAMIL": ["த", "மி", "ழ"]
        },
        "TAMIL": {
            "VANAKKAM": ["வ", "ண", "க", "க", "ம"],
            "NANDRI": ["ந", "ன", "ற", "ி"],
            "TAMIL": ["த", "மி", "ழ"]
        }
    }
    
    return JSONResponse(content={
        "language": language,
        "available_words": common_words.get(language, {}),
        "instructions": f"Practice forming words in {language} by holding each character sign for the required duration"
    })

@app.get("/statistics")
async def get_detection_statistics():
    """Get detection statistics from the current session"""
    global detection_status
    
    stats = {
        "current_session": {
            "active": detection_status["active"],
            "language": detection_status.get("language"),
            "characters_detected": len(detection_status.get("word_buffer", "") + detection_status.get("sentence_buffer", "")),
            "words_formed": len(detection_status.get("sentence_buffer", "").split()) if detection_status.get("sentence_buffer") else 0,
            "session_duration": None
        },
        "system_info": {
            "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
            "detection_method": "Real-time camera with MediaPipe hand tracking",
            "api_version": "3.0.0"
        }
    }
    
    # Calculate session duration if active
    if detection_status["active"] and detection_status.get("session_id"):
        try:
            session_start = datetime.fromisoformat(detection_status["session_id"])
            duration = datetime.now() - session_start
            stats["current_session"]["session_duration"] = str(duration).split('.')[0]  # Remove microseconds
        except:
            pass
    
    return JSONResponse(content=stats)

class ChatProcessRequest(BaseModel):
    text: str
    target_lang: str
    tone: str = "Professional"
    source_lang: str | None = None
    target_lang_code: str | None = None
    target_lang_short: str | None = None
    source_lang_code: str | None = None
    source_lang_short: str | None = None
    emotion: str | None = None
    input_mode: str | None = None
    instruction: str | None = None
    cognitive_load: str | None = None
    sarcasm_detected: bool | None = None
    relationship_state: str | None = None
    code_switching_detected: bool | None = None
    paralinguistic_signals: dict | None = None
    assertiveness_level: int | None = None
    research_signals: dict | None = None


class SpeechRequest(BaseModel):
    text: str
    language: str | None = None
    language_code: str | None = None


LANGUAGE_NAME_MAP = {
    "en": "English",
    "en-us": "English",
    "english": "English",
    "es": "Spanish",
    "es-es": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "fr-fr": "French",
    "french": "French",
    "de": "German",
    "de-de": "German",
    "german": "German",
    "hi": "Hindi",
    "hi-in": "Hindi",
    "hindi": "Hindi",
    "ta": "Tamil",
    "ta-in": "Tamil",
    "tamil": "Tamil",
    "te": "Telugu",
    "te-in": "Telugu",
    "telugu": "Telugu",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
}


def normalize_language_name(value):
    if not value:
        return None

    normalized = str(value).strip().lower()
    return LANGUAGE_NAME_MAP.get(normalized, str(value).strip())


LANGUAGE_CODE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Japanese": "ja",
}


def language_to_code(value):
    normalized = normalize_language_name(value)
    if not normalized:
        return "en"
    return LANGUAGE_CODE_MAP.get(normalized, str(value).strip().lower().split("-")[0])


def apply_tone_locally(text, tone):
    if not text:
        return text

    stripped = text.strip()
    if tone == "Professional":
        refined = stripped.replace("can u", "could you").replace("help me", "assist me")
        if not refined.lower().startswith(("please", "kindly", "dear team", "good day")):
            refined = f"Please note: {refined}"
        if not refined.endswith((".", "!", "?")):
            refined += "."
        return f"{refined} Thank you."
    if tone == "Casual":
        relaxed = stripped.replace("Hello", "Hey").replace("Greetings", "Hey").replace("I would like to", "I want to")
        if not relaxed.lower().startswith(("hey", "hi")):
            relaxed = f"Hey, {relaxed}"
        if not relaxed.endswith(("!", ".")):
            relaxed += "!"
        return relaxed
    if tone == "Empathetic":
        return f"I understand how this may feel. {stripped} I am here to help."
    if tone == "Inclusive":
        updated = stripped.replace("guys", "everyone").replace("he or she", "they").replace("chairman", "chairperson")
        return updated
    return stripped


def simplify_text_for_load(text, cognitive_load):
    if not text:
        return text

    if cognitive_load == "high":
        text = text.replace("therefore", "so").replace("approximately", "about").replace("regarding", "about")
        text = text.replace(",", ".")
        parts = [part.strip() for part in text.split(".") if part.strip()]
        return ". ".join(part[:110].strip() for part in parts[:3]) + ("." if parts else "")

    if cognitive_load == "medium":
        return text.replace("however", "but").replace("regarding", "about").replace("utilize", "use")

    return text


def apply_assertiveness_locally(text, level):
    if level is None or level == 0 or not text:
        return text

    updated = text
    if level > 0:
        updated = updated.replace("maybe", "").replace("perhaps", "").replace("possibly", "")
        updated = updated.replace("I think", "I recommend").replace("could we", "let us").replace("can we", "let us")
        if level >= 2 and not updated.lower().startswith(("please", "let us", "we should")):
            updated = f"We should {updated[:1].lower() + updated[1:]}" if updated else updated
    else:
        updated = updated.replace("We should", "Perhaps we should").replace("let us", "could we")
        if not updated.lower().startswith(("perhaps", "maybe", "please")):
            updated = f"Perhaps {updated[:1].lower() + updated[1:]}" if updated else updated

    return " ".join(updated.split())


def apply_relationship_style(text, relationship_state):
    if not text:
        return text

    if relationship_state == "friend":
        return text.replace("Thank you.", "Thanks!").replace("Please note:", "Just so you know,")
    if relationship_state == "colleague":
        return text.replace("Hey,", "Hello team,")
    return text


def translate_with_mymemory(text, source_code, target_code):
    query = urllib.parse.urlencode({
        "q": text,
        "langpair": f"{source_code}|{target_code}",
    })
    url = f"https://api.mymemory.translated.net/get?{query}"

    with urllib.request.urlopen(url, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))

    translated = payload.get("responseData", {}).get("translatedText", "").strip()
    if not translated:
        raise ValueError("Empty translation from MyMemory")
    return translated


def text_to_speech_stream(text, language_code):
    if not HAS_GTTS:
        raise RuntimeError("gTTS is not installed in the Python API environment.")

    audio_buffer = BytesIO()
    tts = gTTS(text=text, lang=language_code)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

@app.post("/process_chat")
async def process_chat(request: ChatProcessRequest):
    """Process chat text with LLM for translation and tone modification"""
    source_language = (
        normalize_language_name(request.source_lang)
        or normalize_language_name(request.source_lang_code)
        or normalize_language_name(request.source_lang_short)
        or "English"
    )
    target_language = (
        normalize_language_name(request.target_lang)
        or normalize_language_name(request.target_lang_code)
        or normalize_language_name(request.target_lang_short)
        or "English"
    )

    if not model:
        source_code = language_to_code(source_language)
        target_code = language_to_code(target_language)
        processed_text = apply_tone_locally(request.text, request.tone)
        processed_text = simplify_text_for_load(processed_text, request.cognitive_load)
        processed_text = apply_assertiveness_locally(processed_text, request.assertiveness_level)
        processed_text = apply_relationship_style(processed_text, request.relationship_state)
        applied_controls = {
            "tone": request.tone,
            "cognitive_load": request.cognitive_load or "low",
            "assertiveness_level": request.assertiveness_level if request.assertiveness_level is not None else 0,
            "relationship_state": request.relationship_state or "stranger",
            "sarcasm_detected": bool(request.sarcasm_detected),
            "code_switching_detected": bool(request.code_switching_detected),
        }

        if source_code == target_code:
            return JSONResponse(content={
                "original_text": request.text,
                "processed_text": processed_text,
                "tone": request.tone,
                "target_lang": target_language,
                "source_lang": source_language,
                "method": "local_tone_only",
                "applied_controls": applied_controls,
            })

        try:
            translated_text = translate_with_mymemory(processed_text, source_code, target_code)
            return JSONResponse(content={
                "original_text": request.text,
                "processed_text": translated_text,
                "tone": request.tone,
                "target_lang": target_language,
                "source_lang": source_language,
                "method": "mymemory_free",
                "applied_controls": applied_controls,
            })
        except Exception as translation_error:
            return JSONResponse(
                status_code=503,
                content={
                    "message": f"Free translation fallback failed: {str(translation_error)}",
                    "original_text": request.text,
                    "processed_text": processed_text,
                    "tone": request.tone,
                    "target_lang": target_language,
                    "source_lang": source_language,
                    "method": "unavailable"
                }
            )

        # Advanced Mock Logic for demo purposes when Gemini is not available
        text = request.text
        tone = request.tone
        target = request.target_lang.split('-')[0].upper()
        
        # 1. Apply Tone Modification (Simulated)
        processed = text
        if tone == "Professional":
            processed = f"Greetings, I would like to inform you that: {text}. Thank you for your consideration."
            processed = processed.replace("help me", "provide assistance").replace("can u", "would you be able to")
        elif tone == "Casual":
            processed = f"Hey! So, {text}. Cheers! ✌️"
            processed = processed.replace("assistance", "help").replace("Hello", "Hi there")
        elif tone == "Empathetic":
            processed = f"I completely understand your situation. {text}. I'm here to support you in any way I can."
        elif tone == "Inclusive":
            processed = text.replace("guys", "everyone").replace("he or she", "they").replace("chairman", "chairperson")
            processed = f"Hello everyone, {processed}"

        # 2. Simulated Translation Mapping (Common phrases)
        translations = {
            "HI": {"ES": "Hola", "FR": "Bonjour", "HI": "नमस्ते", "DE": "Hallo", "JA": "こんにちは"},
            "THANK YOU": {"ES": "Gracias", "FR": "Merci", "HI": "धन्यवाद", "DE": "Danke", "JA": "ありがとう"},
            "YES": {"ES": "Sí", "FR": "Oui", "HI": "हाँ", "DE": "Ja", "JA": "はい"},
            "NO": {"ES": "No", "FR": "Non", "HI": "नहीं", "DE": "Nein", "JA": "いいえ"}
        }
        
        # Simple word-for-word replacement for a "demo" feel if not English
        final_text = processed
        if target != "EN":
            # Just a demo effect to show it 'processed' it
            final_text = f"[{target}] {processed}"
        
        return JSONResponse(content={
            "original_text": request.text,
            "processed_text": final_text,
            "tone": request.tone,
            "target_lang": request.target_lang,
            "method": "advanced_simulation"
        })

    adaptive_notes = []
    if request.cognitive_load == "high":
        adaptive_notes.append("The receiver may be cognitively overloaded, so simplify wording and shorten sentence structure.")
    elif request.cognitive_load == "medium":
        adaptive_notes.append("Prefer moderate simplicity and avoid unnecessary complexity.")

    if request.sarcasm_detected:
        adaptive_notes.append("Sarcasm or irony may be present. Preserve intent carefully rather than translating literally.")

    if request.relationship_state:
        adaptive_notes.append(f"Conversation relationship state: {request.relationship_state}.")

    if request.code_switching_detected:
        adaptive_notes.append("The message may contain code-switching or mixed scripts.")

    if request.assertiveness_level is not None:
        adaptive_notes.append(f"Assertiveness calibration level: {request.assertiveness_level}.")

    if request.paralinguistic_signals:
        adaptive_notes.append(f"Paralinguistic cues: {request.paralinguistic_signals}.")

    prompt = request.instruction or f"""
    Act as an inclusive communication assistant.
    Transform the following text to have a '{request.tone}' tone.
    Ensure it uses inclusive language (gender-neutral, respectful).
    Then, translate the resulting text from {source_language} to {target_language}.
    {' '.join(adaptive_notes)}

    Text: {request.text}

    Provide only the translated text in the response, written in {target_language}.
    """
    
    try:
        response = model.generate_content(prompt)
        return JSONResponse(content={
            "original_text": request.text,
            "processed_text": response.text.strip(),
            "tone": request.tone,
            "target_lang": target_language,
            "source_lang": source_language,
            "method": "gemini",
            "applied_controls": {
                "tone": request.tone,
                "cognitive_load": request.cognitive_load or "low",
                "assertiveness_level": request.assertiveness_level if request.assertiveness_level is not None else 0,
                "relationship_state": request.relationship_state or "stranger",
                "sarcasm_detected": bool(request.sarcasm_detected),
                "code_switching_detected": bool(request.code_switching_detected),
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing with LLM: {str(e)}"}
        )


@app.post("/speak_text")
async def speak_text(request: SpeechRequest):
    """Generate spoken audio for translated text."""
    if not request.text or not request.text.strip():
        return JSONResponse(status_code=400, content={"message": "Text is required for speech."})

    language_code = (
        language_to_code(request.language_code)
        if request.language_code
        else language_to_code(request.language)
    )

    try:
        audio_stream = text_to_speech_stream(request.text.strip(), language_code)
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=translation.mp3"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error generating speech audio: {str(e)}"}
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Unified Sign Language Detection API",
        "version": "3.0.0",
        "description": "Real-time ASL, ISL, and Tamil Sign Language character detection with OpenCV popup window",
        "supported_languages": ["ASL", "ISL", "TSL", "TAMIL"],
        "endpoints": {
            "start_detection": "POST /start_detection - Start camera detection (specify language in body)",
            "stop_detection": "POST /stop_detection - Stop detection",
            "detection_status": "GET /detection_status - Get real-time status",
            "get_detected_text": "GET /get_detected_text - Get final sentence text",
            "detection_log": "GET /detection_log - Get detection log",
            "clear_session": "DELETE /clear_session - Clear session data",
            "model_info": "GET /model_info/{language} - Get model information",
            "supported_languages": "GET /supported_languages - Get supported languages info",
            "word_formation": "GET /word_formation/{language} - Get word formation exercises",
            "statistics": "GET /statistics - Get detection statistics"
        },
        "usage": {
            "All_Languages": {
                "controls": {
                    "ENTER": "Add current word to sentence",
                    "Q": "Complete sentence and save as text"
                },
                "method": "Automatic detection - hold sign for required duration"
            }
        },
        "features": {
            "unified_api": "Single API for ASL, ISL, and Tamil Sign Language",
            "language_switching": "Switch between different sign languages",
            "real_time_status": "Live detection status and progress",
            "text_output": "Detected sentences saved as text files",
            "automatic_detection": "All languages use automatic character detection",
            "progress_tracking": "Visual progress bars for character detection",
            "word_formation": "Support for word formation exercises"
        },
        "auto_detection_info": {
            "ASL": "Hold sign for 1.5-2 seconds for automatic detection",
            "ISL": "Hold sign for 2-3 seconds for automatic detection", 
            "TSL/TAMIL": "Hold sign for 2-3 seconds for automatic detection"
        }
    }

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(DETECTION_LOG_FILE)), exist_ok=True)
    
    print("Starting Unified Sign Language Detection API...")
    print("Supported Languages: ASL (American), ISL (Indian), and TSL/Tamil")
    print("All languages now use automatic detection - no manual key presses needed!")
    print("The API will open OpenCV popup windows for camera detection")
    print("Text output will be available via /get_detected_text endpoint")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

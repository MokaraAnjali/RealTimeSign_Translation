# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import os
# import time
# from datetime import datetime
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# class ASLDetector:
#     def __init__(self):
#         self.model = None
#         self.mp_hands = None
#         self.mp_drawing = None
#         self.hands = None
#         self.labels_dict = {
#             0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
#             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
#             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
#         }
#         self.model_path = './model.p'
#         self.last_capture_time = 0
#         self.capture_interval = 1  # seconds
    
#     def load_model(self):
#         """Load the trained ASL model and initialize MediaPipe"""
#         try:
#             if not os.path.exists(self.model_path):
#                 print(f"ASL model file not found at {self.model_path}")
#                 return False
            
#             model_dict = pickle.load(open(self.model_path, 'rb'))
#             self.model = model_dict['model']
            
#             # Initialize MediaPipe Hands
#             self.mp_hands = mp.solutions.hands
#             self.mp_drawing = mp.solutions.drawing_utils
#             self.hands = self.mp_hands.Hands(
#                 static_image_mode=False, 
#                 min_detection_confidence=0.5, 
#                 max_num_hands=1
#             )
            
#             print("ASL model loaded successfully")
#             return True
#         except Exception as e:
#             print(f"Error loading ASL model: {e}")
#             return False
    
#     def process_hand_landmarks(self, hand_landmarks):
#         """Process hand landmarks into a feature vector for the ASL model"""
#         data_aux = []
#         x_ = []
#         y_ = []
        
#         for landmark in hand_landmarks.landmark:
#             x_.append(landmark.x)
#             y_.append(landmark.y)
        
#         for landmark in hand_landmarks.landmark:
#             data_aux.append(landmark.x - min(x_))
#             data_aux.append(landmark.y - min(y_))
        
#         # Ensure the feature vector has a fixed length
#         while len(data_aux) < 42:
#             data_aux.append(0)
#         return data_aux[:42]
    
#     def predict(self, hand_landmarks):
#         """Make prediction for ASL character"""
#         try:
#             features = self.process_hand_landmarks(hand_landmarks)
#             prediction = self.model.predict([np.asarray(features)])
#             predicted_index = int(prediction[0])
#             detected_char = self.labels_dict.get(predicted_index, '?')
#             return detected_char, 100.0  # ASL model doesn't provide confidence
#         except Exception as e:
#             print(f"ASL prediction error: {e}")
#             return '?', 0.0
    
#     def get_instructions(self):
#         """Get ASL-specific instructions"""
#         return [
#             "Place hand in green box",
#             "Press SPACE to capture each sign (letter)",
#             "Press ENTER to add the captured word to the sentence", 
#             "Press Q when the sentence is complete"
#         ]
    
#     def is_manual_capture(self):
#         """ASL uses manual capture (SPACE key)"""
#         return True


# class ISLDetector:
#     def __init__(self):
#         self.model = None
#         self.mp_hands = None
#         self.mp_drawing = None
#         self.hands = None
#         self.isl_labels_dict = {
#             0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
#             9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J',
#             19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T',
#             29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z',
#             35: ' '
#         }
#         self.model_path = './final_lstm_hand_model.keras'
        
#         # Automatic detection variables
#         self.last_prediction = None
#         self.prediction_counter = 0
#         self.required_consistent_frames = 30  # ~2 seconds at 15 FPS
#         self.last_added_char = None
#         self.cooldown_counter = 0
#         self.cooldown_frames = 15  # ~1 second cooldown
#         self.confidence_threshold = 60  # Min confidence percentage for ISL
    
#     def load_model(self):
#         """Load the trained ISL LSTM model and initialize MediaPipe"""
#         try:
#             if not os.path.exists(self.model_path):
#                 print(f"ISL model file not found at {self.model_path}")
#                 return False
            
#             self.model = load_model(self.model_path)
#             print("ISL model loaded successfully")
            
#             # Initialize MediaPipe Hands for both hands
#             self.mp_hands = mp.solutions.hands
#             self.mp_drawing = mp.solutions.drawing_utils
#             self.hands = self.mp_hands.Hands(
#                 static_image_mode=False, 
#                 min_detection_confidence=0.5, 
#                 max_num_hands=2  # ISL can use both hands
#             )
            
#             return True
#         except Exception as e:
#             print(f"Error loading ISL model: {e}")
#             return False
    
#     def process_isl_landmarks(self, multi_hand_landmarks):
#         """Process hand landmarks for ISL LSTM model"""
#         # Initialize arrays for both hands (filled with zeros)
#         left_hand = np.zeros(21 * 2)  # 42 features for left hand
#         right_hand = np.zeros(21 * 2)  # 42 features for right hand
        
#         if multi_hand_landmarks:
#             # Process each detected hand
#             for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
#                 # Extract hand keypoints
#                 hand_keypoints = []
#                 for landmark in hand_landmarks.landmark:
#                     hand_keypoints.append(landmark.x)
#                     hand_keypoints.append(landmark.y)
                
#                 # Ensure we have exactly 42 features for each hand
#                 hand_keypoints = np.array(hand_keypoints[:42])
#                 if len(hand_keypoints) < 42:
#                     # Pad if needed
#                     hand_keypoints = np.pad(hand_keypoints, (0, 42 - len(hand_keypoints)))
                    
#                 if hand_idx == 0:
#                     left_hand = hand_keypoints
#                 elif hand_idx == 1:
#                     right_hand = hand_keypoints
        
#         # Combine both hands' features
#         features = np.concatenate([left_hand, right_hand])
        
#         # Format for LSTM: reshape to (batch_size=1, time_steps=1, features=84)
#         features = np.array(features).reshape(1, 1, 84)
        
#         return features
    
#     def predict(self, multi_hand_landmarks):
#         """Make prediction for ISL character"""
#         try:
#             features = self.process_isl_landmarks(multi_hand_landmarks)
#             prediction = self.model.predict(features, verbose=0)
#             predicted_index = np.argmax(prediction[0])
#             confidence = float(prediction[0][predicted_index]) * 100
            
#             # Only consider predictions above threshold
#             if confidence >= self.confidence_threshold:
#                 detected_char = self.isl_labels_dict.get(predicted_index, '?')
#                 return detected_char, confidence
#             else:
#                 return '?', confidence
#         except Exception as e:
#             print(f"ISL prediction error: {e}")
#             return '?', 0.0
    
#     def should_auto_detect(self, detected_char):
#         """Check if character should be automatically detected"""
#         if self.cooldown_counter > 0:
#             self.cooldown_counter -= 1
#             return False, 0
#         elif detected_char != '?' and self.last_prediction == detected_char:
#             self.prediction_counter += 1
#             progress = min(self.prediction_counter / self.required_consistent_frames, 1.0)
            
#             # Check if we've had the same prediction for the required number of frames
#             if self.prediction_counter >= self.required_consistent_frames:
#                 # Reset counter and set cooldown
#                 self.prediction_counter = 0
#                 self.cooldown_counter = self.cooldown_frames
#                 return True, progress
#             return False, progress
#         else:
#             # Reset counter for new prediction
#             self.last_prediction = detected_char
#             self.prediction_counter = 1 if detected_char != '?' else 0
#             return False, 0
    
#     def get_instructions(self):
#         """Get ISL-specific instructions"""
#         return [
#             "ISL Detection - Place hands in green box",
#             "Hold the same sign for 2-3 seconds for automatic detection",
#             "Press ENTER to add the captured word to the sentence", 
#             "Press Q when the sentence is complete"
#         ]
    
#     def is_manual_capture(self):
#         """ISL uses automatic capture"""
#         return False


# class UnifiedSignLanguageDetector:
#     def __init__(self):
#         self.asl_detector = ASLDetector()
#         self.isl_detector = ISLDetector()
#         self.current_detector = None
#         self.language = None
#         self.cap = None
#         self.session_start_time = None
        
#     def initialize(self, language="ASL"):
#         """Initialize the detector for specified language"""
#         self.language = language.upper()
        
#         if self.language == "ASL":
#             self.current_detector = self.asl_detector
#             return self.asl_detector.load_model()
#         elif self.language == "ISL":
#             self.current_detector = self.isl_detector
#             return self.isl_detector.load_model()
#         else:
#             print(f"Unsupported language: {language}")
#             return False
    
#     def get_detector_info(self):
#         """Get information about current detector"""
#         if self.language == "ASL":
#             return {
#                 "language": "ASL (American Sign Language)",
#                 "model_type": "Random Forest Classifier",
#                 "hands_supported": "Single hand",
#                 "classes": len(self.asl_detector.labels_dict),
#                 "supported_characters": list(self.asl_detector.labels_dict.values()),
#                 "manual_capture": True,
#                 "auto_detection": False
#             }
#         elif self.language == "ISL":
#             return {
#                 "language": "ISL (Indian Sign Language)",
#                 "model_type": "LSTM Neural Network",
#                 "hands_supported": "Both hands (2 hands)",
#                 "classes": len(self.isl_detector.isl_labels_dict),
#                 "supported_characters": list(self.isl_detector.isl_labels_dict.values()),
#                 "confidence_threshold": self.isl_detector.confidence_threshold,
#                 "manual_capture": False,
#                 "auto_detection": True,
#                 "consistent_frames_required": self.isl_detector.required_consistent_frames
#             }
#         return {}
    
#     def process_frame(self, frame):
#         """Process a single frame and return detection results"""
#         if not self.current_detector:
#             return None, 0.0, None
        
#         # Process frame for hand detection
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.current_detector.hands.process(frame_rgb)
        
#         detected_char = '?'
#         confidence = 0.0
#         extra_info = {}
        
#         if results.multi_hand_landmarks:
#             if self.language == "ASL":
#                 # ASL uses single hand
#                 hand_landmarks = results.multi_hand_landmarks[0]
#                 detected_char, confidence = self.current_detector.predict(hand_landmarks)
#             elif self.language == "ISL":
#                 # ISL uses multiple hands
#                 detected_char, confidence = self.current_detector.predict(results.multi_hand_landmarks)
                
#                 # Check for auto-detection
#                 should_detect, progress = self.current_detector.should_auto_detect(detected_char)
#                 extra_info = {
#                     "should_auto_detect": should_detect,
#                     "detection_progress": progress,
#                     "cooldown_active": self.current_detector.cooldown_counter > 0
#                 }
        
#         return detected_char, confidence, extra_info
    
#     def draw_landmarks(self, frame, results):
#         """Draw hand landmarks on frame"""
#         if results.multi_hand_landmarks and self.current_detector:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.current_detector.mp_drawing.draw_landmarks(
#                     frame, 
#                     hand_landmarks, 
#                     self.current_detector.mp_hands.HAND_CONNECTIONS,
#                     self.current_detector.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
#                     self.current_detector.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
#                 )
    
#     def get_instructions(self):
#         """Get language-specific instructions"""
#         if self.current_detector:
#             return self.current_detector.get_instructions()
#         return []
    
#     def is_manual_capture(self):
#         """Check if current language uses manual capture"""
#         if self.current_detector:
#             return self.current_detector.is_manual_capture()
#         return True
    
#     def cleanup(self):
#         """Cleanup resources"""
#         if self.cap:
#             self.cap.release()
#         cv2.destroyAllWindows()



import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import json
from datetime import datetime
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: tensorflow not found. ISL and Tamil detectors will be disabled.")

class ASLDetector:
    def __init__(self):
        self.model = None
        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }
        self.model_path = './model.p'
        
        # Automatic detection variables (similar to ISL)
        self.last_prediction = None
        self.prediction_counter = 0
        self.required_consistent_frames = 25  # ~1.5-2 seconds at 15 FPS
        self.last_added_char = None
        self.cooldown_counter = 0
        self.cooldown_frames = 15  # ~1 second cooldown
        self.confidence_threshold = 80  # Higher threshold for ASL
    
    def load_model(self):
        """Load the trained ASL model and initialize MediaPipe"""
        try:
            if not os.path.exists(self.model_path):
                print(f"ASL model file not found at {self.model_path}")
                return False
            
            model_dict = pickle.load(open(self.model_path, 'rb'))
            self.model = model_dict['model']
            
            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, 
                min_detection_confidence=0.5, 
                max_num_hands=1
            )
            
            print("ASL model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading ASL model: {e}")
            return False
    
    def process_hand_landmarks(self, hand_landmarks):
        """Process hand landmarks into a feature vector for the ASL model"""
        data_aux = []
        x_ = []
        y_ = []
        
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)
        
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))
        
        # Ensure the feature vector has a fixed length
        while len(data_aux) < 42:
            data_aux.append(0)
        return data_aux[:42]
    
    def predict(self, hand_landmarks):
        """Make prediction for ASL character"""
        try:
            features = self.process_hand_landmarks(hand_landmarks)
            prediction = self.model.predict([np.asarray(features)])
            predicted_index = int(prediction[0])
            detected_char = self.labels_dict.get(predicted_index, '?')
            
            # Use probability score as confidence if available
            try:
                probabilities = self.model.predict_proba([np.asarray(features)])
                confidence = float(np.max(probabilities)) * 100
            except:
                confidence = 85.0  # Default confidence for ASL
                
            return detected_char, confidence
        except Exception as e:
            print(f"ASL prediction error: {e}")
            return '?', 0.0
    
    def should_auto_detect(self, detected_char, confidence):
        """Check if character should be automatically detected"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0
        elif (detected_char != '?' and 
              confidence >= self.confidence_threshold and 
              self.last_prediction == detected_char):
            self.prediction_counter += 1
            progress = min(self.prediction_counter / self.required_consistent_frames, 1.0)
            
            # Check if we've had the same high-confidence prediction for required frames
            if self.prediction_counter >= self.required_consistent_frames:
                # Reset counter and set cooldown
                self.prediction_counter = 0
                self.cooldown_counter = self.cooldown_frames
                return True, progress
            return False, progress
        else:
            # Reset counter for new prediction
            self.last_prediction = detected_char
            self.prediction_counter = 1 if (detected_char != '?' and confidence >= self.confidence_threshold) else 0
            return False, 0
    
    def get_instructions(self):
        """Get ASL-specific instructions"""
        return [
            "ASL Detection - Place hand in green box",
            "Hold the same sign for 1.5-2 seconds for automatic detection",
            "Press ENTER to add the captured word to the sentence", 
            "Press Q when the sentence is complete"
        ]
    
    def is_manual_capture(self):
        """ASL now uses automatic capture"""
        return False


class ISLDetector:
    def __init__(self):
        self.model = None
        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        self.isl_labels_dict = {
            0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
            9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J',
            19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T',
            29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z',
            35: ' '
        }
        self.model_path = './final_lstm_hand_model.keras'
        
        # Automatic detection variables
        self.last_prediction = None
        self.prediction_counter = 0
        self.required_consistent_frames = 30  # ~2 seconds at 15 FPS
        self.last_added_char = None
        self.cooldown_counter = 0
        self.cooldown_frames = 15  # ~1 second cooldown
        self.confidence_threshold = 60  # Min confidence percentage for ISL
    
    def load_model(self):
        """Load the trained ISL LSTM model and initialize MediaPipe"""
        try:
            if not os.path.exists(self.model_path):
                print(f"ISL model file not found at {self.model_path}")
                return False
            
            if not HAS_TF:
                print("ISL model requires tensorflow which is not installed.")
                return False
            self.model = load_model(self.model_path)
            print("ISL model loaded successfully")
            
            # Initialize MediaPipe Hands for both hands
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, 
                min_detection_confidence=0.5, 
                max_num_hands=2  # ISL can use both hands
            )
            
            return True
        except Exception as e:
            print(f"Error loading ISL model: {e}")
            return False
    
    def process_isl_landmarks(self, multi_hand_landmarks):
        """Process hand landmarks for ISL LSTM model"""
        # Initialize arrays for both hands (filled with zeros)
        left_hand = np.zeros(21 * 2)  # 42 features for left hand
        right_hand = np.zeros(21 * 2)  # 42 features for right hand
        
        if multi_hand_landmarks:
            # Process each detected hand
            for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
                # Extract hand keypoints
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.append(landmark.x)
                    hand_keypoints.append(landmark.y)
                
                # Ensure we have exactly 42 features for each hand
                hand_keypoints = np.array(hand_keypoints[:42])
                if len(hand_keypoints) < 42:
                    # Pad if needed
                    hand_keypoints = np.pad(hand_keypoints, (0, 42 - len(hand_keypoints)))
                    
                if hand_idx == 0:
                    left_hand = hand_keypoints
                elif hand_idx == 1:
                    right_hand = hand_keypoints
        
        # Combine both hands' features
        features = np.concatenate([left_hand, right_hand])
        
        # Format for LSTM: reshape to (batch_size=1, time_steps=1, features=84)
        features = np.array(features).reshape(1, 1, 84)
        
        return features
    
    def predict(self, multi_hand_landmarks):
        """Make prediction for ISL character"""
        try:
            features = self.process_isl_landmarks(multi_hand_landmarks)
            prediction = self.model.predict(features, verbose=0)
            predicted_index = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_index]) * 100
            
            # Only consider predictions above threshold
            if confidence >= self.confidence_threshold:
                detected_char = self.isl_labels_dict.get(predicted_index, '?')
                return detected_char, confidence
            else:
                return '?', confidence
        except Exception as e:
            print(f"ISL prediction error: {e}")
            return '?', 0.0
    
    def should_auto_detect(self, detected_char, confidence):
        """Check if character should be automatically detected"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0
        elif detected_char != '?' and self.last_prediction == detected_char:
            self.prediction_counter += 1
            progress = min(self.prediction_counter / self.required_consistent_frames, 1.0)
            
            # Check if we've had the same prediction for the required number of frames
            if self.prediction_counter >= self.required_consistent_frames:
                # Reset counter and set cooldown
                self.prediction_counter = 0
                self.cooldown_counter = self.cooldown_frames
                return True, progress
            return False, progress
        else:
            # Reset counter for new prediction
            self.last_prediction = detected_char
            self.prediction_counter = 1 if detected_char != '?' else 0
            return False, 0
    
    def get_instructions(self):
        """Get ISL-specific instructions"""
        return [
            "ISL Detection - Place hands in green box",
            "Hold the same sign for 2-3 seconds for automatic detection",
            "Press ENTER to add the captured word to the sentence", 
            "Press Q when the sentence is complete"
        ]
    
    def is_manual_capture(self):
        """ISL uses automatic capture"""
        return False


class TSLDetector:
    def __init__(self):
        self.model = None
        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        self.tamil_labels = None
        self.model_path = './best_lstm_model.keras'
        self.labels_path = './tamil_labels.json'
        
        # Automatic detection variables (similar to ISL)
        self.last_prediction = None
        self.prediction_counter = 0
        self.required_consistent_frames = 30  # ~2 seconds at 15 FPS
        self.last_added_char = None
        self.cooldown_counter = 0
        self.cooldown_frames = 15  # ~1 second cooldown
        self.confidence_threshold = 65  # Min confidence percentage for TSL
    
    def load_model(self):
        """Load the trained Tamil LSTM model and initialize MediaPipe"""
        try:
            # Load the Tamil LSTM model
            if not os.path.exists(self.model_path):
                print(f"Tamil model file not found at {self.model_path}")
                return False
            
            if not HAS_TF:
                print("Tamil model requires tensorflow which is not installed.")
                return False
            self.model = load_model(self.model_path)
            print("Tamil model loaded successfully")
            
            # Load Tamil labels
            if not os.path.exists(self.labels_path):
                print(f"Tamil labels file not found at {self.labels_path}")
                return False
                
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.tamil_labels = json.load(f)
            
            # Convert string keys to integers
            self.tamil_labels = {int(k): v for k, v in self.tamil_labels.items()}
            print(f"Tamil labels loaded: {len(self.tamil_labels)} characters")
            
            # Initialize MediaPipe Hands for both hands (Tamil uses both hands)
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, 
                min_detection_confidence=0.5, 
                max_num_hands=2
            )
            
            return True
        except Exception as e:
            print(f"Error loading Tamil model: {e}")
            return False
    
    def process_tamil_landmarks(self, multi_hand_landmarks):
        """Process hand landmarks for Tamil LSTM model (similar to ISL)"""
        # Initialize arrays for both hands (filled with zeros)
        left_hand = np.zeros(21 * 2)  # 42 features for left hand
        right_hand = np.zeros(21 * 2)  # 42 features for right hand
        
        if multi_hand_landmarks:
            # Process each detected hand
            for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
                # Extract hand keypoints
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.append(landmark.x)
                    hand_keypoints.append(landmark.y)
                
                # Ensure we have exactly 42 features for each hand
                hand_keypoints = np.array(hand_keypoints[:42])
                if len(hand_keypoints) < 42:
                    # Pad if needed
                    hand_keypoints = np.pad(hand_keypoints, (0, 42 - len(hand_keypoints)))
                    
                if hand_idx == 0:
                    left_hand = hand_keypoints
                elif hand_idx == 1:
                    right_hand = hand_keypoints
        
        # Combine both hands' features
        features = np.concatenate([left_hand, right_hand])
        
        # Format for LSTM: reshape to (batch_size=1, time_steps=1, features=84)
        features = np.array(features).reshape(1, 1, 84)
        
        return features
    
    def predict(self, multi_hand_landmarks):
        """Make prediction for Tamil character"""
        try:
            features = self.process_tamil_landmarks(multi_hand_landmarks)
            prediction = self.model.predict(features, verbose=0)
            predicted_index = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_index]) * 100
            
            # Only consider predictions above threshold
            if confidence >= self.confidence_threshold:
                if predicted_index in self.tamil_labels:
                    tamil_info = self.tamil_labels[predicted_index]
                    # Return Tamil character and pronunciation
                    detected_char = tamil_info.get("tamil", "?")
                    pronunciation = tamil_info.get("pronunciation", "")
                    return f"{detected_char}({pronunciation})", confidence
                else:
                    return '?', confidence
            else:
                return '?', confidence
        except Exception as e:
            print(f"Tamil prediction error: {e}")
            return '?', 0.0
    
    def should_auto_detect(self, detected_char, confidence):
        """Check if character should be automatically detected"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, 0
        elif detected_char != '?' and self.last_prediction == detected_char:
            self.prediction_counter += 1
            progress = min(self.prediction_counter / self.required_consistent_frames, 1.0)
            
            # Check if we've had the same prediction for the required number of frames
            if self.prediction_counter >= self.required_consistent_frames:
                # Reset counter and set cooldown
                self.prediction_counter = 0
                self.cooldown_counter = self.cooldown_frames
                return True, progress
            return False, progress
        else:
            # Reset counter for new prediction
            self.last_prediction = detected_char
            self.prediction_counter = 1 if detected_char != '?' else 0
            return False, 0
    
    def get_instructions(self):
        """Get Tamil-specific instructions"""
        return [
            "Tamil Sign Language Detection - Place hands in green box",
            "Hold the same sign for 2-3 seconds for automatic detection",
            "Press ENTER to add the captured word to the sentence", 
            "Press Q when the sentence is complete"
        ]
    
    def is_manual_capture(self):
        """Tamil uses automatic capture"""
        return False


class UnifiedSignLanguageDetector:
    def __init__(self):
        self.asl_detector = ASLDetector()
        self.isl_detector = ISLDetector()
        self.tsl_detector = TSLDetector()
        self.current_detector = None
        self.language = None
        self.cap = None
        self.session_start_time = None
        
    def initialize(self, language="ASL"):
        """Initialize the detector for specified language"""
        self.language = language.upper()
        
        if self.language == "ASL":
            self.current_detector = self.asl_detector
            return self.asl_detector.load_model()
        elif self.language == "ISL":
            self.current_detector = self.isl_detector
            return self.isl_detector.load_model()
        elif self.language == "TSL" or self.language == "TAMIL":
            self.current_detector = self.tsl_detector
            return self.tsl_detector.load_model()
        else:
            print(f"Unsupported language: {language}")
            return False
    
    def get_detector_info(self):
        """Get information about current detector"""
        if self.language == "ASL":
            return {
                "language": "ASL (American Sign Language)",
                "model_type": "Random Forest Classifier",
                "hands_supported": "Single hand",
                "classes": len(self.asl_detector.labels_dict),
                "supported_characters": list(self.asl_detector.labels_dict.values()),
                "manual_capture": False,
                "auto_detection": True,
                "confidence_threshold": self.asl_detector.confidence_threshold,
                "consistent_frames_required": self.asl_detector.required_consistent_frames
            }
        elif self.language == "ISL":
            return {
                "language": "ISL (Indian Sign Language)",
                "model_type": "LSTM Neural Network",
                "hands_supported": "Both hands (2 hands)",
                "classes": len(self.isl_detector.isl_labels_dict),
                "supported_characters": list(self.isl_detector.isl_labels_dict.values()),
                "confidence_threshold": self.isl_detector.confidence_threshold,
                "manual_capture": False,
                "auto_detection": True,
                "consistent_frames_required": self.isl_detector.required_consistent_frames
            }
        elif self.language == "TSL" or self.language == "TAMIL":
            tamil_chars = []
            if self.tsl_detector.tamil_labels:
                tamil_chars = [info.get("tamil", "?") for info in self.tsl_detector.tamil_labels.values()]
            
            return {
                "language": "TSL (Tamil Sign Language)",
                "model_type": "LSTM Neural Network",
                "hands_supported": "Both hands (2 hands)",
                "classes": len(self.tsl_detector.tamil_labels) if self.tsl_detector.tamil_labels else 0,
                "supported_characters": tamil_chars,
                "confidence_threshold": self.tsl_detector.confidence_threshold,
                "manual_capture": False,
                "auto_detection": True,
                "consistent_frames_required": self.tsl_detector.required_consistent_frames
            }
        return {}
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        if not self.current_detector:
            return None, 0.0, None
        
        # Process frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.current_detector.hands.process(frame_rgb)
        
        detected_char = '?'
        confidence = 0.0
        extra_info = {}
        
        if results.multi_hand_landmarks:
            if self.language == "ASL":
                # ASL uses single hand
                hand_landmarks = results.multi_hand_landmarks[0]
                detected_char, confidence = self.current_detector.predict(hand_landmarks)
                
                # Check for auto-detection
                should_detect, progress = self.current_detector.should_auto_detect(detected_char, confidence)
                extra_info = {
                    "should_auto_detect": should_detect,
                    "detection_progress": progress,
                    "cooldown_active": self.current_detector.cooldown_counter > 0
                }
            elif self.language == "ISL":
                # ISL uses multiple hands
                detected_char, confidence = self.current_detector.predict(results.multi_hand_landmarks)
                
                # Check for auto-detection
                should_detect, progress = self.current_detector.should_auto_detect(detected_char, confidence)
                extra_info = {
                    "should_auto_detect": should_detect,
                    "detection_progress": progress,
                    "cooldown_active": self.current_detector.cooldown_counter > 0
                }
            elif self.language == "TSL" or self.language == "TAMIL":
                # Tamil uses multiple hands (similar to ISL)
                detected_char, confidence = self.current_detector.predict(results.multi_hand_landmarks)
                
                # Check for auto-detection
                should_detect, progress = self.current_detector.should_auto_detect(detected_char, confidence)
                extra_info = {
                    "should_auto_detect": should_detect,
                    "detection_progress": progress,
                    "cooldown_active": self.current_detector.cooldown_counter > 0
                }
        
        return detected_char, confidence, extra_info
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks and self.current_detector:
            for hand_landmarks in results.multi_hand_landmarks:
                self.current_detector.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.current_detector.mp_hands.HAND_CONNECTIONS,
                    self.current_detector.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.current_detector.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
    
    def get_instructions(self):
        """Get language-specific instructions"""
        if self.current_detector:
            return self.current_detector.get_instructions()
        return []
    
    def is_manual_capture(self):
        """Check if current language uses manual capture"""
        if self.current_detector:
            return self.current_detector.is_manual_capture()
        return False  # All languages now use automatic detection
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
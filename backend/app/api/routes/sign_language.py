"""
Sign Language API Routes
Endpoints for sign language detection and interpretation.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from pydantic import BaseModel
import base64
import tempfile
import os

from app.models.schemas import (
    SignToIntentRequest,
    SignToIntentResponse,
)
from app.services import (
    get_vision_service,
    get_openai_service,
    get_user_service,
)
from app.services.gesture_sequence_service import get_gesture_session, clear_gesture_session

# Gesture to text/intent mapping - CLEAR AND CONSISTENT
GESTURE_MAPPINGS = {
    "Thumb_Up": {"text": "Good", "intent": "positive"},
    "Thumb_Down": {"text": "No", "intent": "negative"},
    "Open_Palm": {"text": "Hello", "intent": "greeting"},
    "Closed_Fist": {"text": "Yes", "intent": "affirmative"},
    "Victory": {"text": "Peace", "intent": "positive"},
    "Pointing_Up": {"text": "Look", "intent": "attention"},
    "ILoveYou": {"text": "I love you", "intent": "affection"},
    "None": {"text": "No clear gesture detected", "intent": "no_gesture"},
}

router = APIRouter(prefix="/azure", tags=["Sign Language"])


class VideoGestureRequest(BaseModel):
    """Request model for video gesture processing."""
    video_base64: str
    user_id: Optional[str] = None


@router.post("/sign-to-intent", response_model=SignToIntentResponse)
async def sign_to_intent(request: SignToIntentRequest):
    """
    Recognize sign language gestures and convert to intent/text.
    
    Supports continuous gesture detection by maintaining session state.
    Use session_id to accumulate gestures over multiple frames.
    
    This endpoint:
    1. Analyzes the image/video using MediaPipe for hand/face detection
    2. Extracts gesture features and landmarks
    3. Accumulates gestures in a session for sentence formation
    4. Uses AI to interpret gestures into natural language
    
    Accepts either:
    - image_base64: Single frame for gesture detection
    - video_base64: Video for sequence analysis (future)
    """
    vision_service = get_vision_service()
    openai_service = get_openai_service()
    user_service = get_user_service()
    
    try:
        print(f"Sign-to-intent request received. Image: {len(request.image_base64) if request.image_base64 else 0} chars")
        
        if not request.image_base64 and not request.video_base64:
            raise HTTPException(
                status_code=400,
                detail="Either image_base64 or video_base64 must be provided"
            )
        
        # Currently support single frame analysis
        if request.video_base64:
            raise HTTPException(
                status_code=501,
                detail="Video analysis not yet implemented. Please use image_base64."
            )
        
        # Get or create gesture session for continuous detection
        session_id = request.user_id or "default_session"
        gesture_session = get_gesture_session(session_id)
        
        # Step 1: Extract hand and face landmarks with combined gesture recognition
        print("Calling vision_service.analyze_sign_language_frame...")
        analysis = await vision_service.analyze_sign_language_frame(
            request.image_base64
        )
        
        # Log recognition method and results
        recognition_method = analysis.get("recognition_method", "none")
        primary_text = analysis.get("primary_text", "No gesture detected")
        asl_letter = analysis.get("asl_letter")
        
        print(f"Analysis result: method={recognition_method}, text='{primary_text}'")
        if asl_letter:
            print(f"ASL letter detected: {asl_letter.get('letter')} ({asl_letter.get('confidence', 0):.2%})")
        
        if analysis.get("error"):
            print(f"Analysis error: {analysis.get('error')}")
            raise HTTPException(
                status_code=503,
                detail=f"Vision service unavailable: {analysis.get('error')}"
            )

        hands_data = analysis.get("hands", {})
        face_data = analysis.get("face", {})
        combined_gestures = analysis.get("combined_gestures", [])
        
        print(f"Hands detected: {hands_data.get('hands_detected', False)}, combined_gestures: {combined_gestures}")
        
        # Check if any gestures were detected (from either MediaPipe or ASL CNN)
        if not combined_gestures and not hands_data.get("hands_detected"):
            # No gesture but maintain session
            summary = gesture_session.get_gesture_summary()
            print("No hands/gestures detected in frame")
            return SignToIntentResponse(
                intent="no_gesture" if not summary["sentence"] else "continuing",
                text=summary["sentence"] or "No hands detected. Please show your hand clearly in the camera with good lighting.",
                confidence=summary["confidence"],
                gestures_detected=summary["gestures"],
                hand_landmarks=None
            )
        
        # Extract gesture names from combined results
        gesture_names = [g.get("gesture", "Unknown") for g in combined_gestures if g.get("gesture")]
        real_gestures = [g for g in gesture_names if g and g != "None" and g != "Unknown"]
        
        print(f"Real gestures detected: {real_gestures}")
        
        # Add gestures to session for continuous detection
        if real_gestures:
            primary_gesture = real_gestures[0]
            primary_confidence = combined_gestures[0].get("confidence", 0.5) if combined_gestures else 0.5
            gesture_session.add_gesture(primary_gesture, primary_confidence)
            print(f"Added gesture to session: {primary_gesture} (confidence: {primary_confidence:.2%})")
        
        # Handle ASL letter separately for fingerspelling
        if asl_letter and asl_letter.get("letter") and asl_letter.get("confidence", 0) > 0.5:
            letter = asl_letter["letter"]
            if letter not in ["nothing"]:
                gesture_session.add_gesture(f"ASL_{letter}", asl_letter["confidence"])
        
        # Get accumulated sentence from gesture sequence
        summary = gesture_session.get_gesture_summary()
        
        # If no real gestures in current frame but we have accumulated sentence
        if not real_gestures and not asl_letter:
            if summary["sentence"]:
                return SignToIntentResponse(
                    intent="gesture_sequence",
                    text=summary["sentence"],
                    confidence=summary["confidence"],
                    gestures_detected=summary["gestures"],
                    hand_landmarks=None
                )
            else:
                return SignToIntentResponse(
                    intent="unclear_gesture",
                    text="Hand detected but gesture unclear. Try making a clearer sign.",
                    confidence=0.3,
                    gestures_detected=[],
                    hand_landmarks=None
                )
        
        # Get user personalization if available
        personalization = None
        if request.user_id:
            personalization = await user_service.get_personalization_data(
                request.user_id
            )
        
        # Determine the final text output
        if summary["sentence"]:
            text = summary["sentence"]
        elif primary_text and primary_text != "No gesture detected":
            text = primary_text
        elif real_gestures:
            text = GESTURE_MAPPINGS.get(
                real_gestures[0], 
                {"text": real_gestures[0].replace("_", " ").replace("ASL ", ""), "intent": "gesture"}
            )["text"]
        else:
            text = "Gesture detected"
        
        intent = "gesture_sequence" if summary["gesture_count"] > 1 else GESTURE_MAPPINGS.get(
            real_gestures[0] if real_gestures else "None", {"intent": "gesture"}
        ).get("intent", "gesture")
        
        # Prepare hand landmarks for response
        hand_landmarks_response = None
        if combined_gestures:
            hand_landmarks_response = {
                "hands_count": len(combined_gestures),
                "hands": [
                    {
                        "handedness": "Unknown",
                        "gesture": g.get("gesture"),
                        "text": g.get("text"),
                        "confidence": g.get("confidence", 0),
                        "source": g.get("source", "unknown")
                    }
                    for g in combined_gestures
                ]
            }
        
        return SignToIntentResponse(
            intent=intent,
            text=text,
            confidence=summary["confidence"] if summary["confidence"] > 0 else (combined_gestures[0].get("confidence", 0.5) if combined_gestures else 0.5),
            gestures_detected=summary["gestures"] if summary["gestures"] else real_gestures,
            hand_landmarks=hand_landmarks_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sign-to-intent/upload", response_model=SignToIntentResponse)
async def sign_to_intent_upload(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(default=None)
):
    """
    Recognize sign language from uploaded image file.
    """
    try:
        # Read and encode file
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Create request and process
        request = SignToIntentRequest(
            image_base64=image_base64,
            user_id=user_id
        )
        
        return await sign_to_intent(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sign-to-intent/clear-session")
async def clear_sign_session(session_id: str = Form(...)):
    """
    Clear accumulated gestures for a session.
    
    Use this when starting a new gesture sequence or resetting the session.
    """
    try:
        clear_gesture_session(session_id)
        return {"success": True, "message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sign-to-intent/video", response_model=SignToIntentResponse)
async def process_video_gestures(request: VideoGestureRequest):
    """
    Process a video file for continuous gesture recognition.
    
    Extracts frames from the video and processes each frame for gestures,
    then combines all detected gestures into a coherent sentence.
    
    This is the preferred method for capturing a sequence of sign language
    gestures that form a complete message.
    """
    vision_service = get_vision_service()
    
    try:
        import cv2
        import numpy as np
        
        # Decode video from base64
        video_bytes = base64.b64decode(request.video_base64)
        
        # Save to temporary file (OpenCV needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Open video file
            cap = cv2.VideoCapture(tmp_path)
            
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames - process 2 frames per second to balance accuracy and speed
            sample_rate = max(1, int(fps / 2))
            
            # Create a fresh session for this video
            session_id = request.user_id or f"video_{os.urandom(4).hex()}"
            clear_gesture_session(session_id)
            gesture_session = get_gesture_session(session_id)
            
            all_gestures = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Only process every nth frame
                if frame_count % sample_rate != 0:
                    continue
                
                # Encode frame as base64 for gesture recognition
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Analyze frame for gestures
                analysis = await vision_service.analyze_sign_language_frame(frame_base64)
                
                if analysis.get("error"):
                    continue  # Skip frames that fail to process
                
                hands_data = analysis.get("hands", {})
                
                if hands_data.get("hands_detected", 0) > 0:
                    gestures = hands_data.get("gestures", [])
                    real_gestures = [g for g in gestures if g and g != "None"]
                    
                    if real_gestures:
                        primary_gesture = real_gestures[0]
                        confidence = hands_data.get("hands", [{}])[0].get("confidence", 0.5)
                        
                        # Add to session for sequence building
                        gesture_session.add_gesture(primary_gesture, confidence)
                        all_gestures.extend(real_gestures)
                
                processed_frames += 1
            
            cap.release()
            
            # Get the accumulated sentence
            summary = gesture_session.get_gesture_summary()
            
            # Clean up session after processing
            clear_gesture_session(session_id)
            
            if not all_gestures:
                return SignToIntentResponse(
                    intent="no_gesture",
                    text="No gestures detected in video. Please try recording again with clearer hand movements.",
                    confidence=0.0,
                    gestures_detected=[],
                    hand_landmarks=None
                )
            
            # Build response
            sentence = summary.get("sentence", "")
            if not sentence:
                # Fallback: create sentence from unique gestures
                unique_gestures = list(dict.fromkeys(all_gestures))  # Preserve order, remove dups
                sentence_parts = []
                for g in unique_gestures:
                    mapping = GESTURE_MAPPINGS.get(g, {"text": g.replace("_", " ")})
                    sentence_parts.append(mapping["text"])
                sentence = " → ".join(sentence_parts)
            
            return SignToIntentResponse(
                intent="video_gesture_sequence",
                text=sentence,
                confidence=summary.get("confidence", 0.7),
                gestures_detected=summary.get("gestures", list(dict.fromkeys(all_gestures))),
                hand_landmarks={
                    "video_stats": {
                        "total_frames": total_frames,
                        "processed_frames": processed_frames,
                        "gestures_found": len(all_gestures)
                    }
                }
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@router.post("/analyze-gesture")
async def analyze_gesture(
    image_base64: str = Form(...),
    include_landmarks: bool = Form(default=False)
):
    """
    Low-level gesture analysis endpoint.
    
    Returns raw hand and face landmark data without interpretation.
    Useful for debugging and custom processing.
    """
    vision_service = get_vision_service()
    
    try:
        # Get raw analysis
        hand_analysis = await vision_service.extract_hand_landmarks(image_base64)
        face_analysis = await vision_service.extract_face_data(image_base64)
        
        response = {
            "hands_detected": hand_analysis.get("hands_detected", 0),
            "gestures": hand_analysis.get("gestures", []),
            "faces_detected": face_analysis.get("faces_detected", 0),
            "expression": (
                face_analysis.get("faces", [{}])[0].get("expression")
                if face_analysis.get("faces") else None
            )
        }
        
        if include_landmarks:
            response["hand_landmarks"] = hand_analysis.get("hands", [])
            response["face_data"] = face_analysis.get("faces", [])
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-face")
async def analyze_face(image_base64: str = Form(...)):
    """
    Analyze facial expression from an image.
    
    Returns detected faces with expressions for emotion detection.
    """
    vision_service = get_vision_service()
    
    try:
        face_analysis = await vision_service.extract_face_data(image_base64)
        
        return {
            "faces_detected": face_analysis.get("faces_detected", 0),
            "faces": face_analysis.get("faces", []),
            "error": face_analysis.get("error")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import sys
import time
import argparse
from collections import deque, Counter
import cv2
import numpy as np
import joblib

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    mp_hands = None
    mp_draw = None
    print(f"Warning: MediaPipe solutions unavailable: {e}")

# Import FER (Facial Expression Recognition)
try:
    from fer.fer import FER
except Exception as e:
    FER = None
    print(f"Warning: FER unavailable: {e}")

# Import pyttsx3 for Text-to-Speech
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 160)
except Exception:
    tts_engine = None

# Model paths
MODEL_PATH = "data/models/landmarks_model.pkl"
ENC_PATH = "data/models/label_encoder.pkl"

SAY_MAP = {
    str(i): s for i, s in enumerate(["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"])
}


def normalize_hand(arr21x3):
    """Wrist-centered, scale-invariant normalization."""
    if np.allclose(arr21x3, 0):
        return arr21x3
    arr = arr21x3.copy()
    arr -= arr[0]
    scale = np.sqrt((arr**2).sum(axis=1)).mean()
    if scale < 1e-6:
        return arr
    return arr / scale


def predict_sign(feat, model, label_encoder):
    """Predicts gesture label and confidence."""
    if model is None or label_encoder is None:
        return "Model Not Loaded", 0.0
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
        else:
            idx = int(model.predict(feat)[0])
            conf = 1.0
        label = label_encoder.inverse_transform([idx])[0]
        return label, conf
    except Exception:
        return "Unknown", 0.0


def open_camera():
    """Tries opening webcam with multiple backends and indices."""
    for idx in [0, 1]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap
    return None


def main():
    parser = argparse.ArgumentParser(description="Real-Time Indian Sign Language & Emotion Recognition System")
    parser.add_argument("--mode", choices=["both", "sign", "emotion"], default="both",
                        help="Operation mode: 'both' (default), 'sign', or 'emotion'")
    args = parser.parse_args()

    mode = args.mode

    # Load Sign Model
    sign_model = None
    label_encoder = None
    if os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH):
        try:
            sign_model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENC_PATH)
            print("Loaded sign language recognition model.")
        except Exception as ex:
            print(f"Failed to load sign model: {ex}")
    else:
        print("Notice: Sign model not found. Run `python src/train.py` to record & train custom signs.")

    # Load Emotion Detector
    emotion_detector = None
    if FER is not None:
        try:
            print("Initializing Facial Expression Recognition...")
            emotion_detector = FER(mtcnn=False)
            print("FER initialized successfully.")
        except Exception as ex:
            print(f"FER initialization error: {ex}")

    # MediaPipe Hands
    hands_detector = None
    if mp_hands is not None:
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    cap = open_camera()
    if cap is None:
        print("Error: Could not open any webcam (tried indices 0 and 1).")
        return

    print("=" * 65)
    print("  Real-Time ISL & Emotion Recognition System is RUNNING")
    print("=" * 65)
    print("  Controls:")
    print("   [B] : Combined Mode (Sign + Emotion)")
    print("   [S] : Sign Language Mode only")
    print("   [E] : Facial Emotion Mode only")
    print("   [V] : Toggle Voice (TTS) output")
    print("   [Q] / [ESC] : Quit")
    print("=" * 65)

    voice_enabled = True
    SMOOTH_N = 7
    CONF_THRESH = 0.65
    VOTE_MIN_FRAC = 0.6
    pred_history = deque(maxlen=SMOOTH_N)
    last_spoken = ""
    last_say_time = 0.0
    SAY_COOLDOWN = 1.0

    # FPS Calculation
    t0 = time.time()
    frame_count = 0
    fps = 0.0

    # Emotion detect interval to maintain smooth 30 FPS
    emotion_interval = 4  # run FER every 4 frames
    cached_emotions = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed. Exiting.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_sign = "Detecting..."
        current_conf = 0.0

        # --- 1. Emotion Recognition ---
        if mode in ["both", "emotion"] and emotion_detector is not None:
            if frame_count % emotion_interval == 0:
                try:
                    cached_emotions = emotion_detector.detect_emotions(rgb)
                except Exception:
                    cached_emotions = []

            for r in cached_emotions:
                (x, y, ew, eh) = r.get("box", (0, 0, 0, 0))
                em_dict = r.get("emotions", {})
                if em_dict:
                    top_emotion = max(em_dict, key=em_dict.get)
                    em_score = em_dict[top_emotion]
                    cv2.rectangle(frame, (x, y), (x + ew, y + eh), (255, 120, 0), 2)
                    cv2.putText(
                        frame,
                        f"Face: {top_emotion.capitalize()} ({em_score:.2f})",
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 180, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # --- 2. Sign Language Recognition ---
        if mode in ["both", "sign"] and hands_detector is not None:
            results = hands_detector.process(rgb)
            left = np.zeros((21, 3), dtype=np.float32)
            right = np.zeros((21, 3), dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    handed = hd.classification[0].label
                    if handed == "Left":
                        left = coords
                    else:
                        right = coords
                    if mp_draw and mp_hands:
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                left_n = normalize_hand(left)
                right_n = normalize_hand(right)
                feat = np.concatenate([left_n.flatten(), right_n.flatten()])[None, :]

                if sign_model is not None and label_encoder is not None:
                    label_raw, conf = predict_sign(feat, sign_model, label_encoder)
                    pred_history.append((label_raw, conf))
                    labels_only = [p[0] for p in pred_history]
                    most, count = Counter(labels_only).most_common(1)[0]
                    frac = count / len(labels_only)

                    if conf >= CONF_THRESH and frac >= VOTE_MIN_FRAC:
                        current_sign = most
                        current_conf = conf
                    else:
                        current_sign = "Detecting..."
                        current_conf = conf

                    # TTS Speech Output
                    now = time.time()
                    if (
                        voice_enabled
                        and tts_engine is not None
                        and current_sign not in ["Detecting...", ""]
                        and current_sign != last_spoken
                        and (now - last_say_time) > SAY_COOLDOWN
                    ):
                        to_say = SAY_MAP.get(current_sign, current_sign)
                        try:
                            tts_engine.say(to_say)
                            tts_engine.runAndWait()
                        except Exception:
                            pass
                        last_spoken = current_sign
                        last_say_time = now
                else:
                    current_sign = "Hands Tracked"

        # --- 3. FPS Calc ---
        frame_count += 1
        if frame_count % 10 == 0:
            t1 = time.time()
            fps = 10 / (t1 - t0 + 1e-9)
            t0 = t1

        # --- 4. Modern HUD / Overlay ---
        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.line(frame, (0, 85), (w, 85), (0, 200, 255), 2)

        # Title & Mode
        mode_text = f"Mode: {mode.upper()}  |  Voice: {'ON' if voice_enabled else 'OFF'}  |  FPS: {fps:.1f}"
        cv2.putText(frame, mode_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Detected Sign
        sign_display = f"ISL Sign: {current_sign}"
        if current_conf > 0 and current_sign != "Detecting...":
            sign_display += f" ({current_conf:.2f})"
        cv2.putText(frame, sign_display, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA)

        # Controls hint at bottom
        cv2.putText(frame, "[B] Both  [S] Sign  [E] Emotion  [V] Voice  [Q] Quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Indian Sign Language & Face Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
            break
        elif key in [ord("b"), ord("B")]:
            mode = "both"
            print("Switched to COMBINED Mode.")
        elif key in [ord("s"), ord("S")]:
            mode = "sign"
            print("Switched to SIGN LANGUAGE Mode.")
        elif key in [ord("e"), ord("E")]:
            mode = "emotion"
            print("Switched to EMOTION Mode.")
        elif key in [ord("v"), ord("V")]:
            voice_enabled = not voice_enabled
            print(f"Voice output set to: {voice_enabled}")

    cap.release()
    cv2.destroyAllWindows()
    if hands_detector:
        hands_detector.close()


if __name__ == "__main__":
    main()

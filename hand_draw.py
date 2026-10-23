import os
import subprocess
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"
RAW_VIDEO = "recording.mp4"
WA_VIDEO = "whatsapp_video.mp4"


def make_detector():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

        def detect(frame):
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                return [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
            return None
        return detect

    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    start = time.time()

    def detect(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start) * 1000)
        res = landmarker.detect_for_video(img, ts)
        if res.hand_landmarks:
            return [(p.x, p.y) for p in res.hand_landmarks[0]]
        return None
    return detect


def convert_for_whatsapp():
    """recording.mp4 ko WhatsApp wale H.264 format mein badalta hai."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print("WhatsApp video nahi bani. Pehle chalao: pip install imageio-ffmpeg")
        return
    if not os.path.exists(RAW_VIDEO):
        return
    print("Converting for WhatsApp... thoda wait karo")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([
        ffmpeg, "-y", "-i", RAW_VIDEO,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "28", "-movflags", "+faststart", WA_VIDEO,
    ], capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(WA_VIDEO):
        size_mb = os.path.getsize(WA_VIDEO) / (1024 * 1024)
        print(f"Done: {WA_VIDEO} ({size_mb:.1f} MB) - ye WhatsApp pe bhejo")
    else:
        print("Convert nahi hua:")
        print(result.stderr[-300:])


# Toolbar buttons: (label, BGR color). color None = special button
BUTTONS = [
    ("Red", (0, 0, 255)),
    ("Orange", (0, 140, 255)),
    ("Yellow", (0, 220, 255)),
    ("Green", (0, 200, 0)),
    ("Blue", (255, 100, 0)),
    ("Purple", (200, 0, 160)),
    ("Eraser", None),
    ("Clear", None),
]
BAR_H = 60
PALM_ERASE_RADIUS = 45


def main():
    detect = make_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nahi khul rahi.")
        return

    writer = None
    recording = False
    canvas = None
    prev = None
    color = BUTTONS[0][1]
    tool = "pen"
    thickness = 6
    eraser_size = 40

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        bw = w // len(BUTTONS)

        pts = detect(frame)
        if pts:
            ix, iy = int(pts[8][0] * w), int(pts[8][1] * h)
            index_up = pts[8][1] < pts[6][1]
            middle_up = pts[12][1] < pts[10][1]
            ring_up = pts[16][1] < pts[14][1]
            pinky_up = pts[20][1] < pts[18][1]
            palm = index_up and middle_up and ring_up and pinky_up

            if palm:
                # Open palm: erase under the hand
                prev = None
                px, py = int(pts[9][0] * w), int(pts[9][1] * h)
                cv2.circle(canvas, (px, py), PALM_ERASE_RADIUS, (0, 0, 0), -1)
                cv2.circle(frame, (px, py), PALM_ERASE_RADIUS, (255, 255, 255), 2)
            elif index_up and middle_up:
                # Selection mode: pointer move, toolbar select
                prev = None
                cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)
                if iy < BAR_H:
                    name, col = BUTTONS[min(ix // bw, len(BUTTONS) - 1)]
                    if name == "Eraser":
                        tool = "eraser"
                    elif name == "Clear":
                        canvas[:] = 0
                    else:
                        tool = "pen"
                        color = col
            elif index_up and iy > BAR_H:
                # Draw / erase mode
                if tool == "eraser":
                    if prev is not None:
                        cv2.line(canvas, prev, (ix, iy), (0, 0, 0), eraser_size)
                    cv2.circle(frame, (ix, iy), eraser_size // 2, (255, 255, 255), 2)
                else:
                    if prev is not None:
                        cv2.line(canvas, prev, (ix, iy), color, thickness)
                    cv2.circle(frame, (ix, iy), thickness + 4, color, -1)
                prev = (ix, iy)
            else:
                prev = None
        else:
            prev = None

        # Overlay drawing on camera image
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        out = frame.copy()
        out[mask] = canvas[mask]

        # Toolbar on top
        for i, (name, col) in enumerate(BUTTONS):
            x1, x2 = i * bw, (i + 1) * bw
            fill = col if col is not None else (70, 70, 70)
            cv2.rectangle(out, (x1, 0), (x2, BAR_H), fill, -1)
            selected = (name == "Eraser" and tool == "eraser") or \
                       (col is not None and tool == "pen" and col == color)
            border = (255, 255, 255) if selected else (30, 30, 30)
            cv2.rectangle(out, (x1, 0), (x2, BAR_H), border, 4 if selected else 1)
            cv2.putText(out, name, (x1 + 6, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        cv2.putText(out, "1 finger: draw | 2: select | palm: erase | r:rec s:save q:quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Recording: save the frame first, then show the red dot only on screen
        if recording and writer is not None:
            writer.write(out)
            cv2.circle(out, (w - 25, BAR_H + 20), 8, (0, 0, 255), -1)

        cv2.imshow("AI Hand Drawing", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if not recording:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(RAW_VIDEO, fourcc, 20, (w, h))
                recording = True
                print("Recording started")
            else:
                recording = False
                writer.release()
                writer = None
                print("Saved:", RAW_VIDEO)
                convert_for_whatsapp()
        elif key == ord("c"):
            canvas[:] = 0
        elif key in (ord("+"), ord("=")):
            thickness = min(30, thickness + 2)
        elif key == ord("-"):
            thickness = max(2, thickness - 2)
        elif key == ord("s"):
            white = np.full_like(canvas, 255)
            white[mask] = canvas[mask]
            cv2.imwrite("drawing.png", white)
            print("Saved: drawing.png")

    if writer is not None:
        writer.release()
        convert_for_whatsapp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
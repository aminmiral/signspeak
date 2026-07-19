"""signspeak-track: live landmark visualizer to verify the setup. Press q to quit."""

import time

import cv2

from signspeak.capture import Camera
from signspeak.config import load_config
from signspeak.landmarks import HandTracker


def main() -> None:
    cfg = load_config()
    tracker = HandTracker(
        max_num_hands=cfg.hands.max_num_hands,
        min_detection_confidence=cfg.hands.min_detection_confidence,
        min_tracking_confidence=cfg.hands.min_tracking_confidence,
    )
    prev = time.time()
    with Camera(cfg.capture.device, cfg.capture.width, cfg.capture.height) as cam:
        for frame in cam.frames():
            tracker.extract(frame)
            tracker.draw(frame)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(
                frame,
                f"FPS: {fps:.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SignSpeak - tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

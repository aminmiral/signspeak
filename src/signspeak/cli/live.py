"""signspeak-live: live translation — signs -> text -> speech.

Keys: ENTER confirm candidate · x reject candidate · SPACE speak
      backspace undo · c clear · q quit
"""

import cv2

from signspeak.capture import Camera
from signspeak.config import ROOT, load_config
from signspeak.diagnostics import MESSAGES, Visibility, assess
from signspeak.features import vectorize
from signspeak.landmarks import HandTracker
from signspeak.predict import SignClassifier
from signspeak.sentence import SentenceBuilder
from signspeak.tts import speak_async

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def main() -> None:
    cfg = load_config()
    model_file = ROOT / cfg.paths.model_file
    if not model_file.exists():
        raise SystemExit("No trained model — run signspeak-train first.")

    classifier = SignClassifier.load(model_file)
    builder = SentenceBuilder(
        confidence_threshold=cfg.debounce.confidence_threshold,
        stable_frames=cfg.debounce.stable_frames,
        require_confirm=cfg.debounce.require_confirm,
    )
    tracker = HandTracker(
        max_num_hands=cfg.hands.max_num_hands,
        min_detection_confidence=cfg.hands.min_detection_confidence,
        min_tracking_confidence=cfg.hands.min_tracking_confidence,
    )

    with Camera(cfg.capture.device, cfg.capture.width, cfg.capture.height) as cam:
        for frame in cam.frames():
            hands = tracker.extract(frame)
            feats = vectorize(hands.left, hands.right)
            visibility = assess(hands.left, hands.right)

            if feats is not None:
                label, conf = classifier.predict(feats)
                builder.update(label, conf)
                tracker.draw(frame)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.0%})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    GREEN,
                    2,
                )
            else:
                builder.update(None)

            if visibility is not Visibility.OK:
                cv2.putText(
                    frame,
                    MESSAGES[visibility],
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    RED,
                    2,
                )

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
            if builder.pending:
                cv2.putText(
                    frame,
                    f"candidate: {builder.pending}  (ENTER = ok, x = no)",
                    (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    YELLOW,
                    2,
                )
            cv2.putText(
                frame,
                builder.text[-60:] or "(sign to build a sentence)",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2,
            )

            cv2.imshow("SignSpeak - live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 13:  # ENTER
                builder.confirm()
            if key == ord("x"):
                builder.reject()
            if key == ord(" ") and builder.words:
                speak_async(builder.text, backend=cfg.tts.backend, voice=cfg.tts.voice)
            if key == ord("c"):
                builder.clear()
            if key == 8:  # backspace
                builder.undo()

    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

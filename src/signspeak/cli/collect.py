"""signspeak-collect: record labeled landmark samples for one sign.

    signspeak-collect --label HELLO [--signer S01] [--lighting bright]

Press 's' to save a sample (hold for burst capture), 'q' to quit.
Only landmark geometry is saved — never images or video (RULES R-1/R-2).
"""

import argparse
import csv

import cv2

from signspeak.capture import Camera
from signspeak.config import ROOT, load_config
from signspeak.features import FEATURE_DIM, vectorize
from signspeak.landmarks import HandTracker

COLUMNS = [f"f{i}" for i in range(FEATURE_DIM)] + ["label", "signer", "lighting"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Sign name, e.g. HELLO")
    parser.add_argument("--signer", default="S01", help="Anonymous signer ID")
    parser.add_argument("--lighting", default="bright", choices=["bright", "dim"])
    args = parser.parse_args()
    label = args.label.strip().upper()

    cfg = load_config()
    data_file = ROOT / cfg.paths.data_file
    data_file.parent.mkdir(exist_ok=True)
    new_file = not data_file.exists()

    tracker = HandTracker(
        max_num_hands=cfg.hands.max_num_hands,
        min_detection_confidence=cfg.hands.min_detection_confidence,
        min_tracking_confidence=cfg.hands.min_tracking_confidence,
    )

    saved = 0
    with (
        open(data_file, "a", newline="") as f,
        Camera(cfg.capture.device, cfg.capture.width, cfg.capture.height) as cam,
    ):
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COLUMNS)

        for frame in cam.frames():
            hands = tracker.extract(frame)
            feats = vectorize(hands.left, hands.right)
            tracker.draw(frame)

            ok = feats is not None
            status = "hand OK - press 's' to save" if ok else "no hand detected"
            cv2.putText(
                frame,
                f"[{label}] saved: {saved}  {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if ok else (0, 0, 255),
                2,
            )
            cv2.imshow("SignSpeak - collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and feats is not None:
                writer.writerow(list(feats) + [label, args.signer, args.lighting])
                saved += 1

    tracker.close()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples for '{label}' -> {data_file}")


if __name__ == "__main__":
    main()

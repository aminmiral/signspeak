"""signspeak-record-seq: record dynamic-sign landmark sequences to NPZ.

    signspeak-record-seq --label HELLO [--signer S01] [--lighting bright]

Press 'r' to start a recording; the tool captures `window` frames (~1.2 s at
25 fps) and saves the body-frame feature sequence. Press 'q' to quit.
Only landmark geometry is stored — never video (RULES R-1/R-2).
"""

import argparse
import time

import cv2
import numpy as np

from signspeak.capture import Camera
from signspeak.config import ROOT, load_config
from signspeak.features import vectorize_holistic
from signspeak.holistic import HolisticTracker

WINDOW = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--signer", default="S01")
    parser.add_argument("--lighting", default="bright", choices=["bright", "dim"])
    args = parser.parse_args()
    label = args.label.strip().upper()

    cfg = load_config()
    out_dir = ROOT / "data" / "sequences" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = HolisticTracker()
    recording: list[np.ndarray] = []
    active = False
    saved = 0

    with Camera(cfg.capture.device, cfg.capture.width, cfg.capture.height) as cam:
        for frame in cam.frames():
            holistic = tracker.extract(frame)

            if active and holistic.usable:
                recording.append(
                    vectorize_holistic(
                        holistic.pose, holistic.left, holistic.right, holistic.face
                    )
                )
                if len(recording) == WINDOW:
                    stamp = int(time.time() * 1000)
                    path = out_dir / f"{args.signer}_{args.lighting}_{stamp}.npz"
                    np.savez_compressed(
                        path,
                        sequence=np.stack(recording),
                        label=label,
                        signer=args.signer,
                        lighting=args.lighting,
                    )
                    saved += 1
                    recording = []
                    active = False

            status = (
                f"RECORDING {len(recording)}/{WINDOW}"
                if active
                else "press 'r' to record"
            )
            color = (0, 0, 255) if active else (0, 255, 0)
            cv2.putText(
                frame,
                f"[{label}] saved: {saved}  {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.imshow("SignSpeak - record sequences", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not active:
                recording = []
                active = True

    tracker.close()
    cv2.destroyAllWindows()
    print(f"Saved {saved} sequences for '{label}' -> {out_dir}")


if __name__ == "__main__":
    main()

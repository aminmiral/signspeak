"""Command-line entry points (thin wrappers over the library modules).

Modules are not imported eagerly — each pulls in cv2/mediapipe, which must not
load on package import (RULES R-5). Entry points reference them directly, e.g.
`signspeak.cli.track:main`.
"""

__all__ = ["collect", "evaluate", "live", "record_seq", "track", "train"]

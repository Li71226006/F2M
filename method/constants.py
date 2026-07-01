from __future__ import annotations

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

FINGER_JOINT_PREFIXES = {
    "thumb": "THJ",
    "index": "FFJ",
    "middle": "MFJ",
    "ring": "RFJ",
    "little": "LFJ",
}

SHADOWHAND_FINGER_LINK_MAP = {
    0: ["thproximal", "thmiddle", "thdistal"],
    1: ["ffproximal", "ffmiddle", "ffdistal"],
    2: ["mfproximal", "mfmiddle", "mfdistal"],
    3: ["rfproximal", "rfmiddle", "rfdistal"],
    4: ["lfmetacarpal", "lfproximal", "lfmiddle", "lfdistal"],
}

MODES = {
    "5f_full": [1, 1, 1, 1, 1],
    "4f_no_little": [1, 1, 1, 1, 0],
    "3f_thumb_index_middle": [1, 1, 1, 0, 0],
    "2f_thumb_index": [1, 1, 0, 0, 0],
}

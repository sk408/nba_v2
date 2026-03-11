"""Shared threshold constants for winner/pick semantics."""

# Model side-pick threshold on game_score.
# Positive values favor HOME, negative values favor AWAY.
MODEL_PICK_EDGE_THRESHOLD = 0.0

# Actual game winner threshold on realized spread (home_score - away_score).
# Small margins within +/- threshold are treated as PUSH.
ACTUAL_WIN_THRESHOLD = 0.5

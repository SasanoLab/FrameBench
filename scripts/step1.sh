#!/bin/bash

LANGUAGE="${LANGUAGE:-ja}"
DATA_ROOT="${DATA_ROOT:-data}"

uv run python src/generation/1-1_frame_parse.py --data_root "$DATA_ROOT" --language "$LANGUAGE"
uv run python src/generation/1-2_lu_driven_edit.py --data_root "$DATA_ROOT" --language "$LANGUAGE"
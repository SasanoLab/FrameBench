#!/bin/bash

uv run python src/generation/1-1_frame_parse.py --data_root data --language ja
uv run python src/generation/1-2_lu_driven_edit.py --data_root data --language ja
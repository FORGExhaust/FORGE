"""Internal path utilities for locating package data.

Copyright 2025-2026 Chris Marsden

This file is part of FORGE.

FORGE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORGE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FORGE.  If not, see <http://www.gnu.org/licenses/>.
"""
from pathlib import Path


def data_dir_path() -> str:
    return str(Path(__file__).resolve().parent / "data")

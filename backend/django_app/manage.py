#!/usr/bin/env python3
"""
backend/django_app/manage.py

Usage (from this directory):
    python3 manage.py runserver 0.0.0.0:8002
"""

import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Is it installed? "
            "Run: pip install -r requirements.txt (from the project root)."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()

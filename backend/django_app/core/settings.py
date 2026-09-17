"""
backend/django_app/core/settings.py

Phase 8 optional web layer. Deliberately minimal — no database, no auth,
no DRF — this is a thin UI over the already-verified CLI (process_image),
not a separate application with its own data model. Per the SRS's own
framing, this stays secondary to the CUDA centerpiece, so it's kept as
simple as it can be while still satisfying FR: image upload, algorithm
selection, CPU/CUDA/Both execution, result display.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/django_app/
PROJECT_ROOT = BASE_DIR.parent.parent               # gpu-computer-vision/

# Dev-only secret key — this app is a local demo tool, not deployed
# anywhere with real users, so there's no production secret-management
# story needed here.
SECRET_KEY = "dev-only-not-for-production-gpu-cv-engine"
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"

# No database — nothing in this app is persisted beyond the filesystem
# (uploaded/processed images in MEDIA_ROOT). Django still wants a DATABASES
# setting defined even if unused.
DATABASES = {}

STATIC_URL = "static/"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Path to the compiled CLI binary from Phase 8 (cpp/cli/process_image.cpp).
# Built from the project root as `./process_image` per that file's own
# build instructions — this points there directly rather than requiring
# it to be copied anywhere.
PROCESS_IMAGE_BINARY = PROJECT_ROOT / "process_image"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

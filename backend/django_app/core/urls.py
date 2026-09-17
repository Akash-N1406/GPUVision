"""
backend/django_app/core/urls.py
"""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("", include("dashboard.urls")),
]

# Serves uploaded/processed images directly from MEDIA_ROOT during
# development. This app has no production deployment story (see
# settings.py's DEBUG/SECRET_KEY comments), so there's no separate static
# file server to hand this off to.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

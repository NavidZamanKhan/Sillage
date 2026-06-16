"""
Root URL configuration for Sillage 2.0 backend.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("recommendations.urls")),
]

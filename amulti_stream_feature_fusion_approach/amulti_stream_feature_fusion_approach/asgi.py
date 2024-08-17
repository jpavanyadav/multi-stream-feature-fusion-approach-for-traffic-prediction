"""
ASGI config for amulti_stream_feature_fusion_approach

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'amulti_stream_feature_fusion_approach.settings')

application = get_asgi_application()

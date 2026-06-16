"""
API view for /api/search/ — supports both GET and POST.
"""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import SearchSerializer
from .services import search_perfumes


class SearchAPIView(APIView):
    """
    Perfume recommendation search.

    GET  /api/search/?query=rose&gender=Man&limit=5
    POST /api/search/  {"query": "rose", "gender": "Man", "limit": 5}
    """

    def get(self, request):
        return self._handle(request.query_params)

    def post(self, request):
        return self._handle(request.data)

    def _handle(self, data):
        serializer = SearchSerializer(data=data)
        if not serializer.is_valid():
            return Response(
                {"error": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        query = serializer.validated_data["query"]
        gender = serializer.validated_data["gender"]
        limit = serializer.validated_data["limit"]

        try:
            result = search_perfumes(query=query, gender=gender, limit=limit)
        except RuntimeError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            return Response(
                {"error": f"Unexpected error: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(result)

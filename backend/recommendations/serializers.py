"""
Serializer for the /api/search/ endpoint.
"""

from rest_framework import serializers


class SearchSerializer(serializers.Serializer):
    query = serializers.CharField(
        required=True,
        allow_blank=True,
        help_text="Search query: a vibe, note, occasion, or perfume name.",
    )
    gender = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=True,
        default=None,
        help_text="Gender filter: 'Man' or 'Women'. Blank means no filter.",
    )
    limit = serializers.IntegerField(
        required=False,
        default=5,
        min_value=1,
        max_value=100,
        help_text="Number of results to return (1–100).",
    )

    def validate_query(self, value):
        return value.strip() if value else ""

    def validate_gender(self, value):
        if value is None:
            return None
        value = value.strip()
        return value if value else None

    def validate_limit(self, value):
        if value is None or value < 1:
            return 5
        return min(value, 100)

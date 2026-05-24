from enum import Enum


class LLMRoute(str, Enum):
    SUMMARY = "summary"
    BUSINESS = "business"
    FUNDAMENTALS = "fundamentals"
    EARNINGS = "earnings"
    NEWS = "news"

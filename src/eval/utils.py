from loguru import logger


class LangMap:
    """A map with supported languages."""

    map: dict[str, str] = {
        "en": "English",
        "es": "Spanish",
        "ru": "Russian",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ja": "Japanese",
        "he_IL": "Hebrew",
    }

    @classmethod
    def normalize_lang(cls, lang: str) -> str:
        """Extend language code."""
        if _lang := cls.map.get(lang, ""):
            return _lang

        logger.warning(
            f"Language '{lang}' not found in lang_map. "
            "Adapting using the raw value."
        )
        return lang

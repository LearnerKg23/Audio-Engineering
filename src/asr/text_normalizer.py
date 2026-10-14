"""
Indic and Bilingual Text Normalizer (Stage 3 ASR).

Implements rigorous orthographic and phonetic normalization for Indian languages
(Hindi, Hinglish, Marathi, etc.) as mandated by Milestone M6:
1. Unicode NFKC normalization.
2. Devanagari character and nukta harmonization (क़ -> क + ़ or composed).
3. Stripping Indian (danda, double danda) and Western punctuation.
4. Lowercasing Latin script for code-switched Hinglish.
5. Devanagari grapheme and word tokenization for accurate WER and CER computation.
"""

from dataclasses import dataclass
import re
from typing import List, Optional
import unicodedata


# Devanagari nukta mappings (composed <-> decomposed standard forms)
NUKTA_MAPPINGS = {
    "\u0958": "\u0915\u093c",  # क़ -> क + ़
    "\u0959": "\u0916\u093c",  # ख़ -> ख + ़
    "\u095a": "\u0917\u093c",  # ग़ -> ग + ़
    "\u095b": "\u091c\u093c",  # ज़ -> ज + ़
    "\u095c": "\u0921\u093c",  # ड़ -> ड + ़
    "\u095d": "\u0922\u093c",  # ढ़ -> ढ + ़
    "\u095e": "\u092b\u093c",  # फ़ -> फ + ़
    "\u095f": "\u092f\u093c",  # य़ -> य + ़
}

# Inverted mapping to composed form
NUKTA_COMPOSED_MAPPINGS = {v: k for k, v in NUKTA_MAPPINGS.items()}

# Punctuation to strip (Devanagari danda, double danda, Western punctuation, currency symbols, quotes)
PUNCTUATION_PATTERN = re.compile(
    r"[।॥.,!?;:\"'()\[\]{}—\-–/\\|*~`@#$%^&+=_<>«»“”‘’₹\u20b9]"
)

# Zero-Width characters to remove
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

# Whitespace cleaner
WHITESPACE_PATTERN = re.compile(r"\s+")


class IndicTextNormalizer:
    """
    Standardizes bilingual Hindi/English and Indian regional transcripts
    to prevent false error inflation in Word Error Rate (WER) and Character Error Rate (CER).
    """

    def __init__(
        self,
        compose_nuktas: bool = True,
        strip_punctuation: bool = True,
        lowercase_latin: bool = True,
        remove_zero_width: bool = True,
    ):
        self.compose_nuktas = compose_nuktas
        self.strip_punctuation = strip_punctuation
        self.lowercase_latin = lowercase_latin
        self.remove_zero_width = remove_zero_width

    @staticmethod
    def clean_vakyansh_ctc(raw_text: str) -> str:
        """
        Cleans Vakyansh / AI4Bharat CTC token delimiter artifacts (e.g., 's' boundaries).
        Reconstructs canonical Devanagari words from character sequence streams.
        """
        if not raw_text or " s " not in f" {raw_text} ":
            return raw_text

        cleaned = re.sub(r"\s*s\s+s\s*", " <SPACE> ", f" {raw_text} ")
        tokens = cleaned.split()
        words = []
        current_word_chars = []

        for token in tokens:
            if token == "<SPACE>":
                if current_word_chars:
                    words.append("".join(current_word_chars))
                    current_word_chars = []
            elif token == "s":
                continue
            else:
                current_word_chars.append(token)

        if current_word_chars:
            words.append("".join(current_word_chars))

        return " ".join(words)

    def normalize(self, text: Optional[str]) -> str:
        """
        Executes complete normalization pipeline on input text.
        
        Args:
            text: Raw input transcript string.
            
        Returns:
            Cleaned, canonicalized string.
        """
        if not text:
            return ""

        # Pre-clean Vakyansh / AI4Bharat CTC artifacts if present
        text = self.clean_vakyansh_ctc(str(text))

        # 1. Unicode NFKC normalization
        norm_text = unicodedata.normalize("NFKC", str(text))

        # 2. Remove zero-width formatting characters
        if self.remove_zero_width:
            norm_text = ZERO_WIDTH_PATTERN.sub("", norm_text)

        # 3. Nukta harmonization
        if self.compose_nuktas:
            for decomp, comp in NUKTA_COMPOSED_MAPPINGS.items():
                norm_text = norm_text.replace(decomp, comp)
        else:
            for comp, decomp in NUKTA_MAPPINGS.items():
                norm_text = norm_text.replace(comp, decomp)

        # 4. Strip punctuation (danda, commas, periods, quotation marks, dashes)
        if self.strip_punctuation:
            norm_text = PUNCTUATION_PATTERN.sub(" ", norm_text)

        # 5. Lowercase Latin characters for bilingual code-switching
        if self.lowercase_latin:
            norm_text = norm_text.lower()

        # 6. Normalize whitespace
        norm_text = WHITESPACE_PATTERN.sub(" ", norm_text).strip()

        return norm_text

    def tokenize_words(self, text: Optional[str]) -> List[str]:
        """
        Normalizes and splits text into word tokens for WER calculation.
        """
        clean_text = self.normalize(text)
        if not clean_text:
            return []
        return clean_text.split()

    def tokenize_characters(self, text: Optional[str]) -> List[str]:
        """
        Normalizes and extracts non-whitespace character tokens for CER calculation.
        Handles Devanagari characters, matras, and Latin letters consistently.
        """
        clean_text = self.normalize(text)
        if not clean_text:
            return []
        return [ch for ch in clean_text if not ch.isspace()]


# Module-level convenience functions
_DEFAULT_NORMALIZER = IndicTextNormalizer()


def normalize_indic_text(text: Optional[str]) -> str:
    """Convenience helper for standardizing Indic / Hinglish text."""
    return _DEFAULT_NORMALIZER.normalize(text)


def tokenize_indic_words(text: Optional[str]) -> List[str]:
    """Convenience helper for extracting word tokens."""
    return _DEFAULT_NORMALIZER.tokenize_words(text)


def tokenize_indic_characters(text: Optional[str]) -> List[str]:
    """Convenience helper for extracting character tokens."""
    return _DEFAULT_NORMALIZER.tokenize_characters(text)

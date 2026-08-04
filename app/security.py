import re
import logging
from typing import Optional, Tuple
from langsmith import traceable

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Handles prompt injection detection and input cleaning"""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions", #ignore all previous instructions
        r"forget\s+(all\s+)?previous", #forget all previous
        r"new\s+instructions:", #new instructions:
        r"system\s*prompt", #system prompt
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
        r"reveal\s+(your|the)\s+(system|instructions|prompt)",
        r"you\s+are\s+now\s+(DAN|jailbroken)",
    ]

    def __init__(self):
        self.patterns = []
        for p in self.INJECTION_PATTERNS:
            compiled = re.compile(p, re.IGNORECASE) #compiling tells python to prepare the regex for faster matching, and re.IGNORECASE makes it case-insensitive
            self.patterns.append(compiled)

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """Check if input is safe. Returns (is_safe, reason_if_blocked)"""
        for pattern in self.patterns:
            if pattern.search(text):
                return False, f"Blocked: potential prompt injection detected {pattern.pattern}"
        return True, None

    def clean(self, text: str) -> str:
        """Remove potentially dangerous content."""
        # Remove common injection delimiters
        text = re.sub(r"[-]{3,}", "", text) #replace 3 or more dashes with nothing
        text = re.sub(r"[=]{3,}", "", text) #replace 3 or more equal signs with nothing

        # Escape special characters that might confuse the model
        text = text.replace("{{", "{ {").replace("}}", "} }") #{{system}} becomes { {system} } 

        return text.strip() #remove leading and trailing whitespace


class PIIDetector:
    """
    Detect and mask personally identifiable information.
    Works on BOTH input (before LLM) and output (before client).
    """

    PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), # Aadhaar number format
        "credit_card": re.compile(
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        ),
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "aadhaar": "[AADHAAR REDACTED]",
        "credit_card": "[CARD REDACTED]",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII types present in text."""
        found = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches
        return found

    def mask(self, text: str) -> str:
        """Replace all PII with redaction markers."""
        masked = text
        for pii_type, pattern in self.PATTERNS.items(): # Find pattern in text → replace with replacement
            masked = pattern.sub(self.MASK_MAP[pii_type], masked) #pattern.sub(replacement, text)
        return masked


class OutputValidator:
    """
    Validate LLM output before returning to the client.
    Catches PII leakage and harmful content in responses.
    """

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.IGNORECASE), #patterns like "here's how to hack"
        re.compile(r"password\s+is\s+", re.IGNORECASE),#password is patterns like "password is 1234"
        re.compile(r"api[_\s]?key\s*[:=]", re.IGNORECASE), #API key patterns like "api_key: " or "api key = "
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        """
        Validate and clean output.
        Returns: (cleaned_output, list_of_warnings)
        """
        warnings = []

        # Check for PII leakage in output
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")

        # Check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = "[Response blocked: potentially harmful content]"
                warnings.append("Harmful content blocked")
                break

        return output, warnings


class SecurityPipeline:
    """
    Full security pipeline that processes input and output.
    This is the single class you wire into your API.
    """

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable(name="security_check_input")
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Process input through security checks.
        Returns: (is_allowed, cleaned_text, security_notes)
        """
        notes = []

        # Step 1: Check for injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        # Step 2: Clean input
        cleaned = self.sanitizer.clean(text)

        # Step 3: Mask PII before it reaches the LLM
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked: {list(pii_found.keys())}")

        return True, cleaned, notes

    @traceable(name="security_check_output")
    def check_output(self, text: str) -> tuple[str, list[str]]:
        """
        Validate output before returning to client.
        Returns: (cleaned_output, warnings)
        """
        return self.output_validator.validate(text)
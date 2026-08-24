"""
Regex Pattern Definitions for Scam Call Detection
Includes high-confidence regex patterns for India-specific scam vectors.
"""

import re
from typing import Dict, List, Pattern

# Regex patterns for complex scam phrases
PATTERNS: Dict[str, List[Pattern]] = {
    # 1. Digital Arrest / Law Enforcement Threat
    "digital_arrest_pattern": [
        re.compile(r"\b(under|placed on|issued)\s+(digital arrest|court order|warrant)\b", re.IGNORECASE),
        re.compile(r"\b(cbi|cyber crime|customs|police)\s+(enforcement|department|officer|branch)\b", re.IGNORECASE),
        re.compile(r"\b(parcel|package)\s+(containing|intercepted with)\s+(drugs|mdma|illegal|passports)\b", re.IGNORECASE),
        re.compile(r"\b(connect to|stay on)\s+(video call|skype|zoom)\b", re.IGNORECASE),
    ],

    # 2. UPI PIN Coercion
    "upi_coercion_pattern": [
        re.compile(r"\b(enter|input|type)\s+(your\s+)?(6-digit|4-digit)?\s*(upi pin|pin number)\b", re.IGNORECASE),
        re.compile(r"\b(pin|password)\s+is\s+mandatory\s+to\s+(receive|credit)\b", re.IGNORECASE),
        re.compile(r"\b(accept|tap)\s+(the\s+)?(collect request|payment request)\b", re.IGNORECASE),
    ],

    # 3. OTP & Credential Theft
    "otp_theft_pattern": [
        re.compile(r"\b(share|read out|tell me|verify)\s+(the\s+)?(6-digit|4-digit)?\s*(otp|verification code|one time password)\b", re.IGNORECASE),
        re.compile(r"\b(what is|provide)\s+(the\s+)?code\s+sent\s+to\s+your\s+phone\b", re.IGNORECASE),
    ],

    # 4. Remote Access / APK Download
    "remote_access_pattern": [
        re.compile(r"\b(download|install)\s+(the\s+)?(anydesk|teamviewer|quicksupport|rustdesk|apk)\b", re.IGNORECASE),
        re.compile(r"\b(share|provide)\s+(the\s+)?(9-digit|code|remote access)\b", re.IGNORECASE),
    ],

    # 5. Asset Freeze / SIM Deactivation Threat
    "account_freeze_pattern": [
        re.compile(r"\b(account|sim card|sim)\s+will be\s+(permanently\s+)?(blocked|frozen|terminated)\b", re.IGNORECASE),
        re.compile(r"\bwithin\s+(2|1|24|48)\s+hours?\s+due to\s+pending\s+kyc\b", re.IGNORECASE),
    ],

    # 6. Untraceable Payment Demand
    "payment_demand_pattern": [
        re.compile(r"\b(pay|transfer)\s+via\s+(gift cards?|wire|bitcoin|upi|crypto|itunes|target)\b", re.IGNORECASE),
        re.compile(r"\b(secret|police|rbi)\s+(verification account|clearance account)\b", re.IGNORECASE),
    ],

    # 7. Imminent Law Enforcement Action
    "imminent_arrest_pattern": [
        re.compile(r"\b(marshals|police|officers)\s+will\s+arrive\b", re.IGNORECASE),
        re.compile(r"\b(arrest warrant|lawsuit)\s+has been\s+filed\b", re.IGNORECASE),
    ]
}

# Pattern weights for risk scoring
PATTERN_WEIGHTS = {
    "digital_arrest_pattern": 15,
    "upi_coercion_pattern": 15,
    "otp_theft_pattern": 15,
    "remote_access_pattern": 12,
    "account_freeze_pattern": 10,
    "payment_demand_pattern": 12,
    "imminent_arrest_pattern": 10,
}

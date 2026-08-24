"""
Suspicious Word Dictionary & Category Definitions
Includes India-Specific Threat Vectors (Digital Arrest, UPI Coercion, Fake KYC, Hinglish terms)
"""

SCAM_WORDS = {
    # 1. Digital Arrest / Law Enforcement Impersonation (Weight: 10)
    "digital_arrest": {
        "cbi", "central bureau", "enforcement directorate", "customs", "narcotics", "mdma",
        "fedex", "parcel", "contraband", "video call", "digital arrest", "arrest warrant",
        "supreme court", "illegal materials", "cyber crime", "police station", "giraftari",
        "narcotic drugs", "border police", "criminal prosecution"
    },

    # 2. UPI Coercion / Payment Scam (Weight: 10)
    "upi_coercion": {
        "upi pin", "enter pin", "phonepe", "gpay", "google pay", "paytm", "qr code",
        "collect request", "scan qr", "receive money", "balance transfer", "army officer",
        "olx", "pin to credit", "enter pin now"
    },

    # 3. Fake KYC / SIM Block / Remote Access (Weight: 9)
    "fake_kyc": {
        "trai", "sim block", "sim deactivation", "aadhaar", "kyc update", "apk download",
        "anydesk", "teamviewer", "rustdesk", "quicksupport", "screen share", "banking app",
        "account frozen", "asset freeze", "pending kyc"
    },

    # 4. Threats & Legal Prosecution (Weight: 9)
    "threats": {
        "arrest", "warrant", "lawsuit", "legal action", "court", "jail", "police force",
        "marshals", "criminal", "prosecution", "penalty", "enforce", "prosecute"
    },

    # 5. Personal Info & OTP Credential Theft (Weight: 8)
    "personal_info": {
        "ssn", "social security", "password", "pin", "otp", "6-digit", "4-digit",
        "credentials", "cvv", "verification code", "one time password", "netbanking"
    },

    # 6. Financial Fraud & Untraceable Payment (Weight: 8)
    "financial": {
        "wire transfer", "gift card", "bitcoin", "itunes", "target card", "clearance fee",
        "transfer funds", "rbi account", "secret verification account", "penalty fee",
        "western union", "crypto", "pay cash"
    },

    # 7. Official Impersonation (Weight: 7)
    "impersonation": {
        "irs", "internal revenue", "microsoft", "government", "official", "officer",
        "inspector", "cyber branch", "agent", "enforcement officer", "customs officer"
    },

    # 8. Prize & Lottery Scams (Weight: 7)
    "prize_lottery": {
        "winner", "lottery", "prize", "congratulations", "claim reward", "jackpot",
        "lucky winner", "selected for prize"
    },

    # 9. Urgency & Time Pressure (Weight: 5)
    "urgency": {
        "immediately", "don't hang up", "act now", "2 hours", "45 minutes", "30 minutes",
        "emergency", "final warning", "right now", "do not disconnect"
    },

    # 10. Suspicious Software / Remote Control (Weight: 6)
    "suspicious_actions": {
        "remote access", "download", "virus", "malware", "apk", "install app",
        "remote connection", "clean infection", "remote utility"
    },

    # 11. General Sales / Pressure Tactics (Weight: 3 - Low weight to avoid telemarket false positives)
    "pressure_tactics": {
        "limited time", "expires today", "special offer", "discount", "secret",
        "don't miss out", "lock in rate", "exclusive offer"
    }
}

# Category Weights for Scorer
CATEGORY_WEIGHTS = {
    "digital_arrest": 10,
    "upi_coercion": 10,
    "fake_kyc": 9,
    "threats": 9,
    "personal_info": 8,
    "financial": 8,
    "impersonation": 7,
    "prize_lottery": 7,
    "urgency": 5,
    "suspicious_actions": 6,
    "pressure_tactics": 3,
}

# Key categories that mark a HARD SCAM VECTOR vs general sales call
HARD_SCAM_CATEGORIES = {
    "digital_arrest", "upi_coercion", "fake_kyc", "threats", "personal_info", "financial", "suspicious_actions"
}

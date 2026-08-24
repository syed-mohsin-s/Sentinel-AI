"""
Sample Call Transcripts for Testing
Includes India-Specific Scams (Digital Arrest, UPI Coercion, Fake KYC, Parcel Phishing)
in English and Hinglish, plus hard negative legitimate controls.
"""

SCAM_SCRIPTS = {
    # 1. India Target Threat: Digital Arrest (CBI / Customs / Narcotic Police)
    "digital_arrest_cbi": """
    Hello, this is Inspector Rajesh Kumar calling from the CBI Enforcement Directorate in New Delhi. 
    We have intercepted a FedEx parcel bearing your Aadhaar card details sent from Mumbai to Taiwan. 
    The parcel contains 5 illegal passports, 200 grams of MDMA narcotics, and fraudulent bank credit cards. 
    A Supreme Court digital arrest warrant has been issued in your name under Section 302 and Narcotic Drugs Act. 
    Do not hang up this call or inform anyone. You are currently under digital arrest. 
    You must immediately connect to our official Skype video call for verification of your bank accounts and transfer your funds to the RBI secret verification account for safety.
    """,

    # 2. India Target Threat: Hinglish Digital Arrest
    "digital_arrest_hinglish": """
    Aapka bol rahe hain Cyber Crime Branch Delhi Police se. 
    Aapke naam par ek FedEx parcel intercept hua hai jisme 150 gram MDMA drug aur forged passport mila hai. 
    Aap par digital arrest ka court order pass ho gaya hai. Aapko abhi video call par aana padega. 
    Kisi ko mat batana varna ghar pe police force bhej denge. 
    Apne saare bank khate verify karwane ke liye hamare police clearance account me 50000 rupees transfer kariye.
    """,

    # 3. India Target Threat: UPI PIN Coercion (OLX / PhonePe / Google Pay)
    "upi_coercion_gpay": """
    Hello sir, I am calling regarding the sofa set you posted on OLX. I am an army officer posted in Cantonment. 
    I want to buy it immediately. I have sent you a payment request on PhonePe / Google Pay for 15,000 rupees. 
    To receive the money into your bank account, please open your PhonePe app right now. 
    You will see a green collect request. Tap on accept and enter your 6-digit UPI PIN. 
    Remember, entering your UPI PIN is mandatory to credit the money into your bank balance. Enter the PIN now.
    """,

    # 4. India Target Threat: Fake KYC & SIM Deactivation / APK Download
    "fake_kyc_sim_block": """
    Urgent notification from TRAI and HDFC Bank Verification Department. 
    Your SIM card and bank account will be permanently blocked within 2 hours due to pending KYC update. 
    To stop immediate termination, click on the link sent to your SMS and download the official QuickSupport / AnyDesk APK app. 
    Once installed, share the 9-digit remote access code and read out the 6-digit OTP code sent by your bank. 
    Failure to verify immediately will result in complete asset freeze and fine of 25,000 rupees.
    """,

    # 5. India Target Threat: Parcel / Customs Phishing
    "parcel_customs_phishing": """
    Attention: This is Customs Clearance Officer at Mumbai International Airport. 
    An incoming international shipment registered to your mobile number has been detained by border police. 
    The package contains contraband electronics and illegal currency. 
    To clear yourself from criminal prosecution, you must pay a clearance penalty of 18,500 rupees via UPI within 30 minutes. 
    Do not disconnect or contact local authorities until clearance certificate is generated.
    """,

    # 6. Legacy US Scam: IRS Impersonation
    "irs_scam": """
    This is Officer John Thomas from the Internal Revenue Service. 
    We have filed a lawsuit against your name for tax fraud and evasion. 
    If you do not respond immediately, federal marshals will arrive at your home within 45 minutes to execute an arrest warrant. 
    To resolve this lawsuit out of court, you must purchase $2,000 in Target gift cards or send a wire transfer immediately. 
    Stay on the line and do not hang up.
    """,

    # 7. Legacy US Scam: Tech Support Remote Access
    "tech_support_scam": """
    Hello, this is Alex from Microsoft Technical Support. 
    Our automated security monitoring system detected severe virus infection and malware emanating from your Windows computer. 
    Your financial information and passwords are being transmitted to illegal hackers right now. 
    You must open your web browser, navigate to our support portal, and download the remote connection utility. 
    Allow me full access to your desktop so I can clean the infection. There is a $299 service fee payable via iTunes gift cards.
    """,
}


LEGITIMATE_SCRIPTS = {
    # 1. Genuine Bank Fraud Alert (Control / Hard Negative)
    "genuine_bank_fraud_alert": """
    Hello, this is an automated transaction verification alert from HDFC Bank Security. 
    We noticed a debit transaction attempt of 4,500 rupees at 10:14 PM on your card ending in 4092. 
    If you authorized this transaction, please press 1. 
    If you did not authorize this charge, please press 2 to block your card immediately. 
    HDFC Bank will never ask you for your PIN, CVV, or OTP on a call. Thank you for banking with us.
    """,

    # 2. Genuine Bank KYC Reminder
    "genuine_bank_kyc_reminder": """
    Good morning, this is State Bank of India main branch calling. 
    This is a routine notification that your periodic account KYC documentation is due for renewal this month. 
    Please visit your nearest SBI branch with your original PAN card and Aadhaar card at your convenience. 
    Alternatively, you can log in securely to the official SBI YONO application. 
    Please do not share your bank password or OTP with anyone over the phone. Have a great day.
    """,

    # 3. Doctor Appointment Confirmation
    "doctor_appointment": """
    Hi, this is City Health Clinic calling to confirm your appointment with Dr. Sharma tomorrow, 
    Tuesday at 3:30 PM. Please arrive 10 minutes early to complete check-in paperwork. 
    If you need to reschedule or cancel, please reply to this call or visit our website. 
    Thank you and see you tomorrow.
    """,

    # 4. Delivery Notification
    "delivery_notification": """
    Hello, your BlueDart delivery driver is en route with your package. 
    Expected delivery time is between 2:00 PM and 4:00 PM today. 
    No signature or OTP is required for this package. It will be left at your front door.
    """,

    # 5. Job Interview Invitation
    "job_interview": """
    Hello, I am calling from the Human Resources team at TechCorp. 
    We reviewed your application for the Senior Software Engineer position and were impressed with your background. 
    We would like to schedule a 30-minute introductory interview via Google Meet this Thursday. 
    Please let us know what times work best for you.
    """,
}


BORDERLINE_SCRIPTS = {
    # 1. Aggressive Sales / Telemarketing (Hard Negative - Should NOT be flagged CRITICAL/HIGH!)
    "aggressive_sales": """
    Hello sir! Calling from Star Health Insurance with an exclusive end-of-month discount offer! 
    Our comprehensive health cover is available at a massive 40% discount, but this offer expires strictly today at 6 PM! 
    You must act now and lock in this special rate before prices increase tomorrow. 
    Don't miss out on this limited-time opportunity! Can I take 2 minutes of your time to explain the policy benefits?
    """,

    # 2. Customer Feedback Survey
    "survey_call": """
    Hi, I'm calling on behalf of Telecom Insights to conduct a brief 2-minute feedback survey 
    regarding your mobile service satisfaction. Your opinion is important to us. 
    We do not require any personal identification or financial information. 
    Would you be open to answering 3 quick questions about your network coverage?
    """,
}


def get_all_scripts() -> dict:
    """Return dictionary of all sample scripts grouped by category."""
    return {
        "scam": SCAM_SCRIPTS,
        "legitimate": LEGITIMATE_SCRIPTS,
        "borderline": BORDERLINE_SCRIPTS,
    }


def get_script_by_name(name: str) -> str:
    """Find a script by name across all categories."""
    for category in [SCAM_SCRIPTS, LEGITIMATE_SCRIPTS, BORDERLINE_SCRIPTS]:
        if name in category:
            return category[name]
    return ""

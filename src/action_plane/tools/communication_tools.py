"""Communication tools - Dangerous risk level examples.

These tools demonstrate dangerous tool implementations that require
user confirmation before execution due to their potential impact.
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from ...core.domain.tools import RiskLevel
from ..tool_registry import governor_tool


class EmailSchema(BaseModel):
    """Schema for email tool arguments."""
    recipients: List[str] = Field(..., description="List of email recipients")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    cc: List[str] | None = Field(None, description="Optional CC recipients")
    priority: str = Field(default="normal", description="Email priority (low/normal/high)")


class SMSSchema(BaseModel):
    """Schema for SMS tool arguments."""
    phone_number: str = Field(..., description="Phone number to send SMS to")
    message: str = Field(..., max_length=160, description="SMS message content (max 160 chars)")
    sender_name: str | None = Field(None, description="Optional sender name")


@governor_tool(
    name="send_email",
    description="Send email to one or more recipients",
    risk_level=RiskLevel.DANGEROUS,
    args_schema=EmailSchema,
    category="communication",
    tags=["email", "communication", "dangerous"],
    timeout_seconds=30.0,
    max_retries=1,
    requires_auth=True
)
def send_email(
    recipients: List[str], 
    subject: str, 
    body: str,
    cc: List[str] | None = None,
    priority: str = "normal"
) -> dict:
    """Send email to specified recipients using SMTP.
    
    This is a dangerous tool that sends actual emails and requires
    user confirmation before execution.
    
    Args:
        recipients: List of email addresses to send to
        subject: Email subject line
        body: Email body content
        cc: Optional CC recipients
        priority: Email priority level
        
    Returns:
        Dictionary with email sending results
    """
    import uuid
    
    # Get SMTP configuration from environment
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL", smtp_username)
    
    if not all([smtp_server, smtp_username, smtp_password]):
        return {
            "status": "error",
            "error": "SMTP configuration missing",
            "message": "Set SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD environment variables"
        }
    
    # Validate priority
    valid_priorities = ["low", "normal", "high"]
    if priority not in valid_priorities:
        priority = "normal"
    
    message_id = str(uuid.uuid4())
    sent_to = []
    failed = []
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ", ".join(recipients)
        if cc:
            msg['Cc'] = ", ".join(cc)
        msg['Subject'] = subject
        msg['Message-ID'] = f"<{message_id}@governor-system>"
        
        # Set priority header
        if priority == "high":
            msg['X-Priority'] = "1"
        elif priority == "low":
            msg['X-Priority'] = "5"
        else:
            msg['X-Priority'] = "3"
        
        # Attach body
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            
            # Send to each recipient individually to track failures
            all_recipients = recipients + (cc or [])
            for recipient in all_recipients:
                try:
                    server.send_message(msg, from_addr=from_email, to_addrs=[recipient])
                    sent_to.append(recipient)
                except Exception as e:
                    failed.append({
                        "email": recipient,
                        "error": str(e)
                    })
        
        result = {
            "status": "completed",
            "message_id": message_id,
            "sent_at": datetime.utcnow().isoformat(),
            "sent_to": sent_to,
            "failed": failed,
            "total_recipients": len(recipients),
            "successful_sends": len(sent_to),
            "failed_sends": len(failed),
            "subject": subject,
            "priority": priority,
            "smtp_server": smtp_server
        }
        
        if cc:
            result["cc_recipients"] = cc
            
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": "Email sending failed",
            "message": str(e),
            "message_id": message_id
        }


@governor_tool(
    name="send_sms",
    description="Send SMS message to a phone number via Twilio",
    risk_level=RiskLevel.DANGEROUS,
    args_schema=SMSSchema,
    category="communication",
    tags=["sms", "text", "communication", "dangerous"],
    timeout_seconds=20.0,
    max_retries=2,
    requires_auth=True
)
def send_sms(phone_number: str, message: str, sender_name: str | None = None) -> dict:
    """Send SMS message to a phone number using Twilio.
    
    Args:
        phone_number: Phone number to send SMS to
        message: SMS message content (max 160 characters)
        sender_name: Optional sender name
        
    Returns:
        Dictionary with SMS sending results
    """
    import uuid
    
    # Get Twilio configuration from environment
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not all([twilio_account_sid, twilio_auth_token, twilio_phone_number]):
        return {
            "status": "error",
            "error": "Twilio configuration missing",
            "message": "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER environment variables"
        }
    
    # Basic phone number validation
    cleaned_number = phone_number.replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
    
    if not cleaned_number.isdigit() or len(cleaned_number) < 10:
        return {
            "status": "failed",
            "error": "Invalid phone number format",
            "phone_number": phone_number
        }
    
    if len(message) > 160:
        return {
            "status": "failed", 
            "error": "Message exceeds 160 character limit",
            "message_length": len(message)
        }
    
    # Format phone number with country code if missing
    if not cleaned_number.startswith("+"):
        if len(cleaned_number) == 10:
            cleaned_number = "+1" + cleaned_number
        elif len(cleaned_number) == 11 and cleaned_number.startswith("1"):
            cleaned_number = "+" + cleaned_number
        else:
            cleaned_number = "+" + cleaned_number
    
    try:
        # Use Twilio REST API
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Messages.json"
        
        data = {
            "From": twilio_phone_number,
            "To": cleaned_number,
            "Body": message
        }
        
        response = requests.post(
            url,
            data=data,
            auth=(twilio_account_sid, twilio_auth_token),
            timeout=20
        )
        
        response.raise_for_status()
        result_data = response.json()
        
        return {
            "status": "sent",
            "sms_id": result_data.get("sid", str(uuid.uuid4())),
            "sent_at": datetime.utcnow().isoformat(),
            "phone_number": cleaned_number,
            "message_length": len(message),
            "character_count": len(message),
            "delivery_status": result_data.get("status", "queued"),
            "twilio_sid": result_data.get("sid"),
            "price": result_data.get("price"),
            "price_unit": result_data.get("price_unit")
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "failed",
            "error": "SMS sending failed",
            "message": str(e),
            "phone_number": cleaned_number
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": "Unexpected error",
            "message": str(e),
            "phone_number": cleaned_number
        }
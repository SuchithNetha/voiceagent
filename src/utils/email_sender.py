import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

logger = logging.getLogger("EmailSender")

def send_admin_email(subject: str, body: str, html_body: Optional[str] = None, recipient_list: Optional[List[str]] = None):
    """
    Send an email to site administrators.
    Requires SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS in .env.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    
    # Base recipients from .env
    env_emails = os.getenv("ADMIN_NOTIFICATION_EMAILS", "").split(",")
    # Combine with dynamically passed recipients (from DB)
    all_recipients = list(set([e.strip() for e in env_emails if e.strip()] + (recipient_list or [])))

    if not all([smtp_host, smtp_user, smtp_pass, all_recipients]):
        logger.warning("📩 Email notifications are not configured or no recipients. Skipping email alert.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = ", ".join(all_recipients)
        msg["Subject"] = f"[Arya Alert] {subject}"

        msg.attach(MIMEText(body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            
        logger.info(f"📩 Email alert sent to {len(all_recipients)} admins: {subject}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send email alert: {e}")
        return False

def send_performance_report(metrics: dict):
    """Specific helper for weekly/performance reports."""
    subject = "Weekly Performance Report"
    body = f"Arya Voice Agent Performance Summary:\n\n"
    for k, v in metrics.items():
        body += f"- {k}: {v}\n"
    
    return send_admin_email(subject, body)

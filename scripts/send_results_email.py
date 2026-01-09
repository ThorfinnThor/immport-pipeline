from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List


def must_get_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def existing_files(paths: List[str]) -> List[Path]:
    out = []
    for p in paths:
        pp = Path(p)
        if pp.exists() and pp.is_file():
            out.append(pp)
    return out


def main() -> None:
    smtp_host = must_get_env("SMTP_HOST")
    smtp_port = int(must_get_env("SMTP_PORT"))
    smtp_user = must_get_env("SMTP_USERNAME")
    smtp_pass = must_get_env("SMTP_PASSWORD")

    mail_to = must_get_env("MAIL_TO")
    mail_from = must_get_env("MAIL_FROM")

    # Default attachments (edit here if you want)
    default_attachments = [
        "output/immport_cytometry_candidates_full_ranked.csv",
        "output/immport_failed_trials_ranked.csv",
        "output/external_cytometry_candidates.csv",
    ]

    # Allow override via ATTACHMENTS env var (comma-separated)
    att_env = os.environ.get("ATTACHMENTS", "").strip()
    if att_env:
        attachment_paths = [x.strip() for x in att_env.split(",") if x.strip()]
    else:
        attachment_paths = default_attachments

    files = existing_files(attachment_paths)

    msg = EmailMessage()
    msg["Subject"] = "Monthly cytometry dataset report (ImmPort + external repositories)"
    msg["From"] = mail_from
    msg["To"] = mail_to

    body_lines = [
        "Hi,",
        "",
        "Attached are the latest CSV outputs from the automated pipeline.",
        "",
        "Attachments included:",
    ]
    if files:
        body_lines += [f"- {f.as_posix()} ({f.stat().st_size} bytes)" for f in files]
    else:
        body_lines += ["- (none found)"]

    body_lines += [
        "",
        "Regards,",
        "GitHub Actions pipeline",
    ]
    msg.set_content("\n".join(body_lines))

    for f in files:
        data = f.read_bytes()
        msg.add_attachment(
            data,
            maintype="text",
            subtype="csv",
            filename=f.name,
        )

    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    print("Email sent successfully.")


if __name__ == "__main__":
    main()

import os
import re
import uuid

from Backend.database import get_db_connection
from Backend.embeddings import create_embedding
from Backend.pdf_parser import extract_text_from_pdf


UPLOAD_FOLDER = "uploads/resumes"
MAX_RESUME_SIZE = 10 * 1024 * 1024  # 10 MB


def extract_candidate_info(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    name = lines[0] if lines else "Unknown"

    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    email = email_match.group(0) if email_match else None

    return name, email


def process_resume(file, user_id, batch_id):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # -------------------------
    # Validate filename
    # -------------------------
    original_filename = os.path.basename(
        file.filename or ""
    )

    if not original_filename:
        raise ValueError("Filename is missing")

    if not original_filename.lower().endswith(".pdf"):
        raise ValueError("File is not a PDF")

    # -------------------------
    # Read uploaded file
    # -------------------------
    file_content = file.file.read()

    if not file_content:
        raise ValueError("Uploaded file is empty")

    if len(file_content) > MAX_RESUME_SIZE:
        raise ValueError(
            "Resume file is too large. Maximum size is 10 MB"
        )

    # -------------------------
    # Validate PDF signature
    # -------------------------
    if not file_content.startswith(b"%PDF"):
        raise ValueError(
            "Uploaded file is not a valid PDF"
        )

    # -------------------------
    # Generate unique filename
    # -------------------------
    stored_filename = f"{uuid.uuid4()}.pdf"

    file_path = os.path.join(
        UPLOAD_FOLDER,
        stored_filename
    )

    # -------------------------
    # Save resume
    # -------------------------
    with open(file_path, "wb") as output_file:
        output_file.write(file_content)

    # -------------------------
    # Extract resume text
    # -------------------------
    resume_text = extract_text_from_pdf(file_path)

    if not resume_text:
        raise ValueError(
            "Could not extract text from PDF"
        )

    # -------------------------
    # Extract candidate info
    # -------------------------
    name, email = extract_candidate_info(
        resume_text
    )

    # -------------------------
    # Create embedding
    # -------------------------
    embedding = create_embedding(
        resume_text
    )

    # -------------------------
    # Save candidate to database
    # -------------------------
    with get_db_connection() as connection:
        candidate = connection.execute(
            """
            INSERT INTO candidates
            (
                user_id,
                batch_id,
                name,
                email,
                resume_filename,
                resume_text,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                batch_id,
                name,
                email,
                original_filename,
                resume_text,
                embedding
            )
        ).fetchone()

        connection.commit()

    return {
        "candidate_id": candidate["id"],
        "filename": original_filename
    }
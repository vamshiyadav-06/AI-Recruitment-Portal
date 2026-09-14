import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from Backend.database import (
    get_db_connection,
    close_db_pool,
)

from Backend.resume import process_resume
from Backend.job import create_job
from Backend.matching import get_top_candidates

from Backend.auth import (
    create_users_table,
    register_user,
    login_user,
    get_current_user,
)

from Backend.interview import generate_interview_questions

from Backend.candidate import (
    get_user_candidates,
    get_candidate,
    delete_candidate,
)


# -------------------------
# Application Lifespan
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_users_table()

    yield

    # Shutdown
    close_db_pool()


app = FastAPI(
    title="AI Recruitment Portal API",
    lifespan=lifespan,
)


# -------------------------
# Authentication
# -------------------------

security = HTTPBearer()


def get_logged_in_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        return get_current_user(
            credentials.credentials
        )

    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
        )


# -------------------------
# Request Models
# -------------------------

class JobRequest(BaseModel):
    title: str
    description: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# -------------------------
# Home & Status
# -------------------------

@app.get("/")
def home():
    return {
        "message": "AI Recruitment Portal Backend is running",
        "groq_configured": bool(
            os.getenv("GROQ_API_KEY")
        ),
    }


@app.get("/api/groq/status")
def groq_status():
    api_key_set = bool(
        os.getenv("GROQ_API_KEY")
    )

    masked_key = ""

    if api_key_set:
        raw = os.getenv("GROQ_API_KEY", "")

        masked_key = (
            raw[:6] + "..." + raw[-4:]
            if len(raw) > 10
            else "***"
        )

    return {
        "configured": api_key_set,
        "masked_key": masked_key,
        "model": "openai/gpt-oss-20b",
    }


# -------------------------
# Authentication Endpoints
# -------------------------

@app.post("/auth/register")
def register(request: RegisterRequest):
    try:
        user = register_user(
            request.name,
            request.email,
            request.password,
        )

        return {
            "message": "User registered successfully",
            **user,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@app.post("/auth/login")
def login(request: LoginRequest):
    try:
        user = login_user(
            request.email,
            request.password,
        )

        return {
            "message": "Login successful",
            **user,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
        )


# -------------------------
# Upload Resumes
# -------------------------

@app.post("/resumes/upload")
async def upload_resumes(
    files: list[UploadFile] = File(...),
    current_user=Depends(get_logged_in_user),
):
    user_id = current_user["id"]

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded",
        )

    # -------------------------
    # Create batch
    # -------------------------

    with get_db_connection() as connection:
        batch = connection.execute(
            """
            INSERT INTO resume_batches
            (
                user_id,
                total_files,
                status
            )
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                len(files),
                "processing",
            )
        ).fetchone()

        connection.commit()

    batch_id = batch["id"]

    successful = []
    failed = []

    # -------------------------
    # Process one resume
    # -------------------------

    def process_single_resume(file):
        try:
            result = process_resume(
                file,
                user_id,
                batch_id,
            )

            return {
                "success": True,
                "result": result,
                "filename": file.filename,
            }

        except Exception as error:
            return {
                "success": False,
                "filename": file.filename,
                "error": str(error),
            }

    # -------------------------
    # Process resumes concurrently
    # -------------------------

    with ThreadPoolExecutor(
        max_workers=5
    ) as executor:

        futures = [
            executor.submit(
                process_single_resume,
                file
            )
            for file in files
        ]

        for future in as_completed(futures):
            result = future.result()

            if result["success"]:
                successful.append(
                    result["result"]
                )

            else:
                failed.append({
                    "filename": result["filename"],
                    "error": result["error"],
                })

            # -------------------------
            # Update batch progress
            # -------------------------

            with get_db_connection() as connection:
                connection.execute(
                    """
                    UPDATE resume_batches
                    SET
                        processed_files = processed_files + 1,
                        successful_files = successful_files
                            + %s,
                        failed_files = failed_files
                            + %s
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (
                        1 if result["success"] else 0,
                        0 if result["success"] else 1,
                        batch_id,
                        user_id,
                    )
                )

                connection.commit()

    # -------------------------
    # Mark batch completed
    # -------------------------

    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE resume_batches
            SET
                status = %s,
                completed_at = NOW()
            WHERE id = %s
              AND user_id = %s
            """,
            (
                "completed",
                batch_id,
                user_id,
            )
        )

        connection.commit()

    return {
        "batch_id": batch_id,
        "total_uploaded": len(files),
        "successful_resumes": successful,
        "failed_resumes": failed,
        "status": "completed",
    }


# -------------------------
# Create Job
# -------------------------

@app.post("/jobs")
def create_new_job(
    job: JobRequest,
    current_user=Depends(get_logged_in_user),
):
    try:
        return create_job(
            job.title,
            job.description,
            current_user["id"],
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# -------------------------
# Match Candidates
# -------------------------

@app.get("/matching/{job_id}")
def match_candidates(
    job_id: int,
    current_user=Depends(get_logged_in_user),
):
    try:
        matches = get_top_candidates(
            job_id,
            current_user["id"],
        )

        return {
            "job_id": job_id,
            "matches": matches,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# -------------------------
# Interview Questions
# -------------------------

@app.post("/interview/{job_id}/{candidate_id}")
def generate_candidate_interview(
    job_id: int,
    candidate_id: int,
    current_user=Depends(get_logged_in_user),
):
    try:
        return generate_interview_questions(
            job_id,
            candidate_id,
            current_user["id"],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# -------------------------
# Candidates
# -------------------------

@app.get("/candidates")
def list_candidates(
    current_user=Depends(get_logged_in_user),
):
    return {
        "candidates": get_user_candidates(
            current_user["id"],
        ),
    }


@app.get("/candidates/{candidate_id}")
def get_candidate_by_id(
    candidate_id: int,
    current_user=Depends(get_logged_in_user),
):
    return get_candidate(
        candidate_id,
        current_user["id"],
    )


@app.delete("/candidates/{candidate_id}")
def remove_candidate(
    candidate_id: int,
    current_user=Depends(get_logged_in_user),
):
    return delete_candidate(
        candidate_id,
        current_user["id"],
    )
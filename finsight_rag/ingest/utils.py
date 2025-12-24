import hashlib
import os

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_company_from_filename(filename: str) -> str:
    """
    Extracts company name from filenames like:
    'Embraer-Annual-Report.pdf' -> 'Embraer'
    """
    name = os.path.splitext(filename)[0]   # remove .pdf
    company = name.split("-", 1)[0]        # split only on first "-"
    return company.strip()

def extract_year_from_filename(filename: str) -> str:
    """
    Extracts year from filenames like:
    'Embraer-Annual-Report-2022.pdf' -> '2022'
    """
    name = os.path.splitext(filename)[0]   # remove .pdf
    parts = name.split("-")
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4:
            return part
    return "unknown"
from finsight_rag.ingest.utils import extract_company_from_filename, extract_year_from_filename
    
def test_extract_company_from_filename():
    filename = "Embraer-AnnualReport-2024.pdf"
    company = extract_company_from_filename(filename)
    assert company == "Embraer"
    
    filename = "Petrobas-Management-Report-2024.pdf"
    company = extract_company_from_filename(filename)
    assert company == "Petrobas"

def test_extract_year_from_filename():
    filename = "Embraer-AnnualReport-2024.pdf"
    year = extract_year_from_filename(filename)
    assert year == "2024"
    
    filename = "Petrobas-Management-Report-2024.pdf"
    year = extract_year_from_filename(filename)
    assert year == "2024"
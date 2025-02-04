import pypdf

def extract_text_from_pdf(pdf_path, output_path="source.txt"):
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])

    # Write the cleaned text to source.txt
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(text)

    print(f"Extracted text saved to {output_path}")

# Example usage
pdf_file = input("[.]Enter PDF Path: ")  # Replace with your actual PDF file path
extract_text_from_pdf(pdf_file)

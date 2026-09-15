"""
pdf_export.py

Converts the generated markdown report to PDF using pandoc + wkhtmltopdf.
Run from within the report's own output directory so the relative image
paths already embedded in the markdown (e.g. ![...](chart.png)) resolve
correctly during conversion.

Requires: pandoc and wkhtmltopdf installed on the system.
"""

import os
import subprocess

_STYLE = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 14px; color: #1f2937;
       line-height: 1.6; margin: 40px; }
h1, h2, h3 { color: #1E3A5F; }
img { max-width: 100%; height: auto; margin: 12px 0; display: block; }
"""


def convert_markdown_to_pdf(markdown_path: str, pdf_path: str) -> str:
    output_dir = os.path.dirname(markdown_path) or "."
    md_filename = os.path.basename(markdown_path)
    pdf_filename = os.path.basename(pdf_path)

    body_html_name = "_report_body.html"
    subprocess.run(
        ["pandoc", md_filename, "-f", "markdown", "-t", "html", "-o", body_html_name, "--wrap=none"],
        cwd=output_dir,
        check=True,
    )

    body_html_path = os.path.join(output_dir, body_html_name)
    with open(body_html_path, "r") as f:
        body_html = f.read()

    full_html = f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><style>{_STYLE}</style></head><body>\n{body_html}\n</body></html>"

    full_html_name = "_report_full.html"
    full_html_path = os.path.join(output_dir, full_html_name)
    with open(full_html_path, "w") as f:
        f.write(full_html)

    subprocess.run(
        ["wkhtmltopdf", "--enable-local-file-access", full_html_name, pdf_filename],
        cwd=output_dir,
        check=True,
    )

    for tmp_name in (body_html_name, full_html_name):
        tmp_path = os.path.join(output_dir, tmp_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return pdf_path
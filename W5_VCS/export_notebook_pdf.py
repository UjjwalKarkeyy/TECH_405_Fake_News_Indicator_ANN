from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config
from playwright.sync_api import sync_playwright


EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
)


def find_browser() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No supported Chromium browser was found. Install Microsoft Edge or Chrome."
    )


def notebook_to_html(notebook_path: Path, exclude_input: bool) -> str:
    config = Config()
    config.HTMLExporter.embed_images = True
    config.TemplateExporter.exclude_input = exclude_input

    notebook = nbformat.read(notebook_path, as_version=4)
    exporter = HTMLExporter(config=config)
    html, _ = exporter.from_notebook_node(notebook)
    return html


def export_pdf(notebook_path: Path, output_path: Path, exclude_input: bool) -> None:
    html = notebook_to_html(notebook_path, exclude_input)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "notebook.html"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(find_browser()),
                headless=True,
            )
            page = browser.new_page()
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(output_path),
                format="A4",
                print_background=True,
                margin={
                    "top": "0.5in",
                    "right": "0.5in",
                    "bottom": "0.5in",
                    "left": "0.5in",
                },
            )
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an executed Jupyter notebook to PDF without requiring LaTeX."
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default="RNN_Sentiment_Analysis.ipynb",
        type=Path,
        help="Path to the notebook to export.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path. Defaults to the notebook name with .pdf.",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Hide code cells and export only markdown plus outputs.",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_path = (args.output or notebook_path.with_suffix(".pdf")).resolve()
    export_pdf(notebook_path, output_path, args.no_input)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import (
    PictureItem,
    TableItem
)
from langchain_core.documents import Document

PDF_PATH = Path("input/wildfire_paper.pdf")
OUTPUT_DIR = Path( "output")

def create_directories() -> dict[str, Path]:

    directories = {
        "markdown": OUTPUT_DIR / "markdown",
        "json": OUTPUT_DIR / "json",
        "text": OUTPUT_DIR / "text",
        "pages": OUTPUT_DIR / "pages",
        "figures": OUTPUT_DIR / "figures",
        "tables": OUTPUT_DIR / "tables",
        "chunks": OUTPUT_DIR / "chunks"
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def create_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options =  EasyOcrOptions(
            lang=[ "en" ],
            force_full_page_ocr=False
        )

    pipeline_options.do_table_structure = True

    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.generate_page_images = True
    
    pipeline_options.generate_picture_images = True

    pipeline_options.generate_table_images = True

    pipeline_options.images_scale = 2.0

    return DocumentConverter(

        allowed_formats=[
            InputFormat.PDF
        ],

        format_options={

            InputFormat.PDF:
                PdfFormatOption(

                    pipeline_options=
                        pipeline_options
                )
        },
    )


def export_standard_formats(document: Any,directories: dict[str, Path],) -> None:
    stem = PDF_PATH.stem
    markdown_path = directories["markdown"]/ f"{stem}.md"
    markdown_path.write_text(
        document.export_to_markdown(),
        encoding="utf-8",
    )

    text_path = directories["text"]/ f"{stem}.txt"
    text_path.write_text(
        document.export_to_text(),
        encoding="utf-8",
    )

    json_path = directories["json"]/ f"{stem}.json"
    document.save_as_json(
        json_path
    )


def export_page_images(document: Any,pages_directory: Path) -> None:
    for page_number, page in document.pages.items():
        if page.image is None:
            continue
        output_path = pages_directory/f"page-{page_number:04d}.png"
        page.image.pil_image.save(
            output_path
        )

def export_figures_and_tables(document: Any,figures_directory: Path,tables_directory: Path) -> None:
    figure_number = 0
    table_number = 0
    for element, _level in document.iterate_items():
        if isinstance(element,PictureItem):
            figure_number += 1
            image = element.get_image(document)
            if image is not None:

                figure_path = (

                    figures_directory

                    / (
                        f"figure-"
                        f"{figure_number:04d}.png"
                    )
                )

                image.save(
                    figure_path
                )

        elif isinstance(
            element,
            TableItem,
        ):

            table_number += 1

            table_image = (
                element.get_image(
                    document
                )
            )

            if table_image is not None:

                image_path = (

                    tables_directory

                    / (
                        f"table-"
                        f"{table_number:04d}.png"
                    )
                )

                table_image.save(
                    image_path
                )

            markdown_path = (

                tables_directory

                / (
                    f"table-"
                    f"{table_number:04d}.md"
                )
            )

            markdown_path.write_text(

                element.export_to_markdown(
                    document
                ),

                encoding="utf-8",
            )

            html_path = (

                tables_directory

                / (
                    f"table-"
                    f"{table_number:04d}.html"
                )
            )

            html_path.write_text(

                element.export_to_html(
                    document
                ),

                encoding="utf-8",
            )

            dataframe = (
                element.export_to_dataframe(
                    document
                )
            )

            csv_path = (

                tables_directory

                / (
                    f"table-"
                    f"{table_number:04d}.csv"
                )
            )

            dataframe.to_csv(
                csv_path,
                index=False,
            )


def get_chunk_pages(chunk: Any) -> list[int]:
    page_numbers: set[int] = set()
    doc_items = getattr(chunk.meta, "doc_items", [])
    for item in doc_items:
        provenance_items = getattr(item, "prov", [])
        for provenance in provenance_items:
            page_number = getattr(provenance, "page_no",None)
            if page_number is not None:
                page_numbers.add(page_number)
    return sorted(page_numbers)

def create_rag_chunks(document: Any,chunks_directory: Path) -> list[Document]:
    chunker = HybridChunker()
    chunks = list(chunker.chunk(dl_doc=document))
    jsonl_path = chunks_directory/f"{PDF_PATH.stem}-chunks.jsonl"
    langchain_documents: list[ Document] = []
    with jsonl_path.open("w", encoding="utf-8") as file:
        for index, chunk in enumerate(chunks):
            enriched_text = chunker.contextualize(chunk=chunk)
            headings = list(getattr(chunk.meta, "headings", []) or [])
            captions = list(getattr( chunk.meta,"captions",[],) or [])
            chunk_id = (f"{PDF_PATH.stem}"f"-chunk-{index:05d}")
            metadata = { "chunk_id": chunk_id, "source":PDF_PATH.name,
                         "chunk_index":index, "page_numbers":get_chunk_pages(chunk),
                         "headings": headings,"captions":captions
                       }

            record = {"id":chunk_id, "text":enriched_text, 
                       "metadata":metadata,
                      }

            file.write(json.dumps(record,ensure_ascii=False)+ "\n")

            langchain_documents.append(Document(
                    page_content=enriched_text,
                    metadata = metadata)
            )

    return langchain_documents


def inspect_document(document: Any) -> None:
    counts = Counter(

        type(element).__name__

        for element, _level
        in document.iterate_items()
    )

    print(
        "\nDetected document elements:"
    )

    for element_type, count in sorted(
        counts.items()
    ):

        print(
            f"  {element_type}: {count}"
        )


def run_chunking_pipeline() -> list[Document]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF was not found: "
            f"{PDF_PATH.resolve()}"
        )

    directories = create_directories()
    converter = create_converter()
    print(f"\nParsing: {PDF_PATH}")
    conversion_result = converter.convert(PDF_PATH)
    document = conversion_result.document
    inspect_document(document)
    export_standard_formats(
        document = document,
        directories = directories,
        )
    export_page_images(
        document = document,
        pages_directory=directories["pages"]
        )
    export_figures_and_tables(
        document = document,
        figures_directory = directories["figures"],
        tables_directory = directories["tables"]
           )
    langchain_documents = create_rag_chunks(
            document = document,
            chunks_directory = directories["chunks"]
        )

    print(f"\nRAG chunks created: "f"{len(langchain_documents)}")
    return langchain_documents
run_chunking_pipeline()
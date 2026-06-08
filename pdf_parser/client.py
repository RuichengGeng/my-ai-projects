"""
MinerU cloud API client for PDF parsing.

Strategy (auto-selected based on page count):
  - ≤ 200 pages → v4 Batch File Extract, single item (one upload, no splitting)
  - > 200 pages → split into 200-page chunks with pypdf, all chunks submitted
                  as one batch request (processed in parallel by the API)

Folder parsing
--------------
  parse_folder() scans a directory for PDFs, skips already-parsed ones,
  and submits all short PDFs in a single MinerU batch call.

Usage:
    from pdf_parser import MinerUClient
    client = MinerUClient(api_token="your-token")

    # Single file
    client.parse_pdf("report.pdf", output_path="out/")

    # Whole folder
    client.parse_folder("APAC Oil week/", output_path="APAC Oil week/")
"""

import io
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

from .config import (
    BATCH_PAGE_LIMIT,
    MINERU_API_TOKEN,
    MINERU_BASE_URL,
    POLL_INTERVAL_SECONDS,
    POLL_TIMEOUT_SECONDS,
)


class MinerUError(Exception):
    """Raised when the MinerU API returns an error."""


class MinerUTimeoutError(MinerUError):
    """Raised when polling for results times out."""


def _require_token(token: Optional[str]) -> str:
    resolved = token or MINERU_API_TOKEN
    if not resolved:
        raise MinerUError(
            "No MinerU API token provided. Set MINERU_API_TOKEN in your environment "
            "or pass api_token to MinerUClient().\n"
            "Get your token at: https://mineru.net/apiManage/token"
        )
    return resolved


class MinerUClient:
    def __init__(
        self,
        api_token: Optional[str] = None,
        timeout: int = POLL_TIMEOUT_SECONDS,
        poll_interval: int = POLL_INTERVAL_SECONDS,
    ):
        self.api_token = _require_token(api_token)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_token}"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_pdf(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        """Parse a single local PDF and return markdown.

        Args:
            file_path:   Path to the PDF file.
            output_path: Parent directory for output. A subfolder named after the
                         PDF stem is created there with the .md and images/.
                         If None, no output is written.

        Returns:
            Parsed markdown string.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        output_dir = self._make_output_dir(file_path, output_path)
        markdown = self._parse_one(file_path, output_dir)

        if output_dir is not None:
            (output_dir / (file_path.stem + ".md")).write_text(markdown, encoding="utf-8")
            print(f"  Written → {output_dir}/")

        return markdown

    def parse_folder(
        self,
        folder_path: str | Path,
        output_path: str | Path | None = None,
        skip_existing: bool = True,
    ) -> dict[str, str]:
        """Parse all PDFs in a folder, submitting short ones in a single batch.

        Args:
            folder_path:    Directory containing PDF files.
            output_path:    Parent directory for output (default: same as folder_path).
                            Each PDF gets its own subfolder: {output_path}/{pdf_stem}/
            skip_existing:  Skip PDFs whose output subfolder already exists.

        Returns:
            Dict mapping PDF stem → markdown string.
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        out_parent = Path(output_path) if output_path else folder_path
        pdfs = sorted(folder_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {folder_path}")
            return {}

        print(f"Folder: {folder_path.name}/ — {len(pdfs)} PDF(s) found")

        # Separate into: skip, short (≤200 pages), large (>200 pages)
        to_skip: list[Path] = []
        short_pdfs: list[Path] = []
        large_pdfs: list[Path] = []

        for pdf in pdfs:
            output_dir = out_parent / pdf.stem
            if skip_existing and self._is_already_parsed(output_dir):
                to_skip.append(pdf)
                continue
            pages = self._count_pages(pdf)
            if pages <= BATCH_PAGE_LIMIT:
                short_pdfs.append(pdf)
            else:
                large_pdfs.append(pdf)

        for pdf in to_skip:
            print(f"  Skipping (already parsed): {pdf.name}")

        results: dict[str, str] = {}

        # Submit all short PDFs in one batch
        if short_pdfs:
            print(f"\n  Submitting {len(short_pdfs)} PDF(s) as one batch:")
            for pdf in short_pdfs:
                print(f"    {pdf.name}")

            output_dirs = [out_parent / pdf.stem for pdf in short_pdfs]
            for d in output_dirs:
                d.mkdir(parents=True, exist_ok=True)

            markdowns = self._batch_extract_files(short_pdfs, output_dirs)
            for pdf, output_dir, markdown in zip(short_pdfs, output_dirs, markdowns):
                (output_dir / (pdf.stem + ".md")).write_text(markdown, encoding="utf-8")
                print(f"  Written → {output_dir}/")
                results[pdf.stem] = markdown

        # Process large PDFs individually (chunked)
        for pdf in large_pdfs:
            pages = self._count_pages(pdf)
            n_chunks = (pages + BATCH_PAGE_LIMIT - 1) // BATCH_PAGE_LIMIT
            print(f"\n  {pdf.name}: {pages} pages → chunked ({n_chunks} chunks)")
            output_dir = out_parent / pdf.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            markdown = self._parse_in_chunks(pdf, pages, output_dir)
            (output_dir / (pdf.stem + ".md")).write_text(markdown, encoding="utf-8")
            print(f"  Written → {output_dir}/")
            results[pdf.stem] = markdown

        print(f"\nDone — {len(results)} PDF(s) parsed.")
        return results

    def parse_url(self, pdf_url: str) -> str:
        """Parse a PDF from a public URL via v4 single-task API."""
        resp = self._session.post(
            f"{MINERU_BASE_URL}/api/v4/extract/task",
            json={"url": pdf_url},
        )
        payload = resp.json()
        if payload.get("code") != 0:
            raise MinerUError(f"v4 URL submit failed: {payload.get('msg')}")
        return self._poll_v4_single(payload["data"]["task_id"])

    # ------------------------------------------------------------------
    # Internal: single-file dispatch
    # ------------------------------------------------------------------

    def _parse_one(self, file_path: Path, output_dir: Path | None) -> str:
        """Parse one PDF — direct batch if short, chunked if large."""
        total = self._count_pages(file_path)
        if total <= BATCH_PAGE_LIMIT:
            print(f"  {file_path.name}: {total} pages → direct batch extract")
            return self._batch_extract_files([file_path], [output_dir])[0]
        else:
            n_chunks = (total + BATCH_PAGE_LIMIT - 1) // BATCH_PAGE_LIMIT
            print(f"  {file_path.name}: {total} pages → chunked batch ({n_chunks} chunks)")
            return self._parse_in_chunks(file_path, total, output_dir)

    # ------------------------------------------------------------------
    # Chunking (> BATCH_PAGE_LIMIT pages)
    # ------------------------------------------------------------------

    def _parse_in_chunks(self, file_path: Path, total: int, output_dir: Path | None = None) -> str:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(file_path))
        tmp_files: list[Path] = []

        try:
            for start in range(0, total, BATCH_PAGE_LIMIT):
                end = min(start + BATCH_PAGE_LIMIT, total)
                writer = PdfWriter()
                for i in range(start, end):
                    writer.add_page(reader.pages[i])
                tmp = Path(tempfile.mktemp(suffix=f"_p{start+1}-{end}.pdf"))
                with open(tmp, "wb") as f:
                    writer.write(f)
                tmp_files.append(tmp)
                print(f"    chunk: pages {start + 1}–{end}")

            # All chunks share the same output_dir (they form one document)
            output_dirs = [output_dir] * len(tmp_files)
            results = self._batch_extract_files(tmp_files, output_dirs)
            return "\n\n".join(results)

        finally:
            for f in tmp_files:
                f.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # v4 Batch File Extract
    # ------------------------------------------------------------------

    def _batch_extract_files(
        self,
        file_paths: list[Path],
        output_dirs: list[Path | None],
    ) -> list[str]:
        """Upload files, poll for completion, return markdown in original order.

        output_dirs is parallel to file_paths — each file extracts its images
        into its own directory.
        """
        batch_id, upload_urls = self._request_batch_urls(file_paths)
        for path, url in zip(file_paths, upload_urls):
            self._upload_to_oss(url, path)
        zip_urls = self._poll_batch_zip_urls(batch_id, len(file_paths))
        return [self._extract_zip(zip_url, odir) for zip_url, odir in zip(zip_urls, output_dirs)]

    def _request_batch_urls(self, file_paths: list[Path]) -> tuple[str, list[str]]:
        files_meta = [
            {"name": p.name, "data_id": str(i)}
            for i, p in enumerate(file_paths)
        ]
        resp = self._session.post(
            f"{MINERU_BASE_URL}/api/v4/file-urls/batch",
            json={"files": files_meta, "enable_formula": True, "enable_table": True},
        )
        payload = resp.json()
        if payload.get("code") != 0:
            raise MinerUError(f"Batch URL request failed: {payload.get('msg')}")
        data = payload["data"]
        return data["batch_id"], data["file_urls"]

    @staticmethod
    def _upload_to_oss(url: str, file_path: Path) -> None:
        with open(file_path, "rb") as f:
            resp = requests.put(url, data=f.read())
        if resp.status_code != 200:
            raise MinerUError(
                f"Failed to upload {file_path.name} (HTTP {resp.status_code}): {resp.text[:300]}"
            )

    def _poll_batch_zip_urls(self, batch_id: str, n_files: int) -> list[str]:
        """Poll until all batch items are done; return zip URLs in data_id order."""
        start = time.time()
        url = f"{MINERU_BASE_URL}/api/v4/extract-results/batch/{batch_id}"

        while True:
            if time.time() - start > self.timeout:
                raise MinerUTimeoutError(f"Batch {batch_id} timed out after {self.timeout}s.")

            payload = self._session.get(url).json()
            if payload.get("code") != 0:
                raise MinerUError(f"Batch poll failed: {payload.get('msg')}")

            results = payload["data"]["extract_result"]
            done = [r for r in results if r["state"] == "done"]
            failed = [r for r in results if r["state"] == "failed"]

            if failed:
                msgs = [f"{r.get('file_name')}: {r.get('err_msg')}" for r in failed]
                raise MinerUError(f"Batch items failed: {'; '.join(msgs)}")

            if len(done) == n_files:
                done.sort(key=lambda r: int(r.get("data_id", 0)))
                return [r["full_zip_url"] for r in done]

            running = sum(1 for r in results if r["state"] == "running")
            pending = n_files - len(done) - len(failed) - running
            print(
                f"  [{batch_id[:8]}…] {len(done)}/{n_files} done"
                + (f", {running} running" if running else "")
                + (f", {pending} pending" if pending else "")
            )
            time.sleep(self.poll_interval)

    def _extract_zip(self, zip_url: str, output_dir: Path | None = None) -> str:
        """Download and unpack a result ZIP.

        Images are written to output_dir so relative links in the .md resolve
        correctly when opened from that directory.
        """
        resp = self._session.get(zip_url)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            md_name = next(n for n in zf.namelist() if n.endswith(".md"))
            markdown = zf.read(md_name).decode("utf-8")
            if output_dir is not None:
                for name in zf.namelist():
                    if name.endswith("/") or name.endswith(".md"):
                        continue
                    dest = output_dir / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
            return markdown

    # ------------------------------------------------------------------
    # v4 single-task poll (for parse_url)
    # ------------------------------------------------------------------

    def _poll_v4_single(self, task_id: str) -> str:
        start = time.time()
        url = f"{MINERU_BASE_URL}/api/v4/extract/task/{task_id}"

        while True:
            if time.time() - start > self.timeout:
                raise MinerUTimeoutError(f"Task {task_id} timed out.")

            d = self._session.get(url).json()
            state = d["data"]["state"]

            if state == "done":
                return self._extract_zip(d["data"]["full_zip_url"])
            if state == "failed":
                raise MinerUError(f"Task {task_id} failed: {d['data'].get('err_msg')}")

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _make_output_dir(file_path: Path, output_path) -> Path | None:
        if not output_path:
            return None
        d = Path(output_path) / file_path.stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _is_already_parsed(output_dir: Path) -> bool:
        return output_dir.is_dir() and any(output_dir.glob("*.md"))

    @staticmethod
    def _count_pages(file_path: Path) -> int:
        from pypdf import PdfReader
        return len(PdfReader(str(file_path)).pages)

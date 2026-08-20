"""Auto-generated function wrapper for tool: extract_pdf_summary"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'extract_pdf_summary', 'type': 'script', 'command': 'python -c "import sys; from pathlib import Path; print(Path(sys.argv[1]).name)" {pdf_path}', 'description': 'Example PDF-aware tool entry; the server extracts the first three pages from the provided PDF output path.', 'output_control': {'intercept_large_output': False, 'max_preview_lines': 50}, 'inputs': {'pdf_path': {'type': 'string', 'description': 'Path to a PDF file inside /data.'}}, 'expected_outputs': [{'name': 'pdf_path', 'render_as': 'pdf'}]}


def extract_pdf_summary(pdf_path: str) -> str:
    """Example PDF-aware tool entry; the server extracts the first three pages from the provided PDF output path."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'pdf_path': pdf_path}))

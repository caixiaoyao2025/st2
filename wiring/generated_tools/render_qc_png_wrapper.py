"""Auto-generated function wrapper for tool: render_qc_png"""
from agent_connector.tool_runner import run_tool_spec, format_result

_TOOL_SPEC = {'name': 'render_qc_png', 'type': 'script', 'command': 'python {script_path} {metrics_tsv_path} {png_output_path}', 'description': 'Run a user-provided Python plotting script that converts a metrics TSV into a PNG image.', 'output_control': {'intercept_large_output': True, 'max_preview_lines': 40}, 'inputs': {'script_path': {'type': 'string', 'description': 'Path to a Python plotting script inside /data.'}, 'metrics_tsv_path': {'type': 'string', 'description': 'Path to a metrics TSV file inside /data.'}, 'png_output_path': {'type': 'string', 'description': 'Target path for the generated PNG image inside /data.'}}, 'expected_outputs': [{'name': 'metrics_tsv_path', 'render_as': 'dataframe', 'max_rows': 20}, {'name': 'png_output_path', 'render_as': 'image'}]}


def render_qc_png(script_path: str, metrics_tsv_path: str, png_output_path: str) -> str:
    """Run a user-provided Python plotting script that converts a metrics TSV into a PNG image."""
    return format_result(run_tool_spec(_TOOL_SPEC, {'script_path': script_path, 'metrics_tsv_path': metrics_tsv_path, 'png_output_path': png_output_path}))

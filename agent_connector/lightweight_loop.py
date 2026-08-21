"""Agent-agnostic lightweight tool-calling loop.

Generic ReAct/Reflexion-style loop: retrieve relevant tools from the MCP
registry graph, prompt an OpenAI-compatible LLM to emit a ``<tool_call>`` (JSON)
or ``<execute>`` (code) block, execute it via the registry's ``tool_runner``,
feed the result back, and repeat until the LLM returns a final answer.

This is intentionally independent of any specific foreign agent. It is used:

* as the "Native tool calling" driver for agents (e.g. BioChatter) that expose
  no generic tool-calling loop of their own, and
* as the Unified Tool Runner for code-execution agents (e.g. A1).
"""

import re
import json
import subprocess
import sys

from agent_connector.graph_retrieval import retrieve_tools
from agent_connector.tool_runner import run_tool_spec, format_result


class LightweightToolLoop:
    def __init__(self, tool_retriever, openai_client=None, model=None,
                 max_iter=5, temperature=0.3):
        self.tool_retriever = tool_retriever
        self.openai_client = openai_client
        self.model = model
        self.max_iter = max_iter
        self.temperature = temperature
        self.tools = []

    def run(self, query):
        if self.openai_client is None or self.model is None:
            raise RuntimeError("LightweightToolLoop requires openai_client and model")
        spec_lookup = (self.tool_retriever or {}).get('spec_lookup', {})
        if not spec_lookup:
            return f"[error] No tools available for query: {query}"
        enhanced = self._retrieve_and_build_prompt(query)
        retrieved = re.findall(r'\[Tool: (\w+)\]', enhanced)
        messages = [{"role": "user", "content": enhanced}]
        done = []
        raw_str = ''
        for iteration in range(self.max_iter):
            resp = self.openai_client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature)
            raw_str = resp.choices[0].message.content or ''
            calls = self._parse_tool_calls(raw_str)
            if not calls:
                return raw_str
            tool_results = []
            for c in calls:
                name = c.get('name', c.get('_type', '?'))
                result_str = self._execute_tool_call(c, spec_lookup)
                tool_results.append(f'<tool_result>\n{result_str}\n</tool_result>')
                akey = json.dumps(c.get('arguments', {}) or {}, sort_keys=True)
                ok = ('error' not in result_str.lower()) and (
                    'command_error' not in result_str.lower())
                done.append((name, akey, ok))
            summary = '\n'.join(
                f'- {nm} (args={akey}): {"SUCCESS" if ok else "FAILED"}'
                for (nm, akey, ok) in done)
            loop_note = (
                '\n\nAlready executed this session \u2014 do NOT repeat any of them; '
                'chain to the NEXT step using the output they produced:\n' + summary
            ) if done else ''
            feedback = (
                'Tool execution results:\n' + '\n'.join(tool_results) + loop_note +
                '\n\nContinue with the next step or provide the final answer.')
            messages.append({"role": "assistant", "content": raw_str})
            messages.append({"role": "user", "content": feedback})
        return raw_str if raw_str else '(max iterations reached)'

    def _build_tool_prompt(self, specs):
        if not specs:
            return ''
        lines = ['You have access to the following tools:', '']
        for s in specs:
            lines.append(f'[Tool: {s["name"]}]')
            lines.append(f'Description: {s.get("description", "")}')
            inputs = s.get('inputs') or {}
            if inputs:
                lines.append('Arguments:')
                for pname, pmeta in inputs.items():
                    if pname == 'subcommand':
                        continue
                    ptype = (pmeta or {}).get('type', 'string')
                    req = 'required' if (pmeta or {}).get('required') else 'optional'
                    pdesc = (pmeta or {}).get('description', '')
                    lines.append(f'  - {pname} ({ptype}, {req}): {pdesc}')
        lines.append('')
        lines.append('IMPORTANT: You MUST use these tools. Do NOT write your own Python code.')
        lines.append('To call a tool, use <tool_call> JSON format:')
        lines.append('')
        lines.append('<tool_call>')
        lines.append('{"name": "<tool_name_from_list_above>", "arguments": {"<param>": "<value>"}}')
        lines.append('</tool_call>')
        lines.append('')
        lines.append('Rules:')
        lines.append('- Use the exact tool name from the list above.')
        lines.append('- Arguments must match the schema (required/optional, types).')
        lines.append('- Call ONE tool per <tool_call> block.')
        lines.append('- After receiving <tool_result>, continue solving the task.')
        return '\n'.join(lines)

    def _parse_tool_calls(self, text):
        calls = []
        for m in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.S):
            try:
                calls.append(json.loads(m.group(1).strip()))
            except json.JSONDecodeError:
                pass
        if not calls:
            for m in re.finditer(r'<execute>(.*?)</execute>', text, re.S):
                code = m.group(1).strip()
                if code:
                    calls.append({'_type': 'code', 'code': code})
        return calls

    def _execute_tool_call(self, call, spec_lookup):
        if call.get('_type') == 'code':
            code = call['code']
            try:
                cp = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                    text=True, timeout=120, encoding='utf-8',
                                    errors='replace')
                output = cp.stdout or ''
                if cp.stderr:
                    output += '\n[stderr]\n' + cp.stderr
                return output.strip() or '[no output]'
            except Exception as e:
                return f'[error] {type(e).__name__}: {e}'
        name = call.get('name', '')
        args = call.get('arguments', {})
        spec = spec_lookup.get(name)
        if not spec:
            return f'[error] Tool "{name}" not found in registry'
        return format_result(run_tool_spec(spec, args))

    def _retrieve_and_build_prompt(self, query):
        retriever = self.tool_retriever or {}
        graph = retriever.get('graph')
        if graph is None:
            specs = list((retriever.get('spec_lookup') or {}).values())
            return self._build_tool_prompt(specs) + f'\n\nTask: {query}'
        results = retrieve_tools(graph, query, top_k=20)
        injected_results = []
        if retriever.get('injected_graph'):
            injected_results = retrieve_tools(retriever['injected_graph'], query, top_k=20)
        scored = {}
        for name, score, conf in (injected_results or []) + (results or []):
            if name not in scored or score > scored[name]:
                scored[name] = score
        lookup = retriever.get('spec_lookup', {})
        def rank_key(name):
            method = ((lookup.get(name) or {}).get('install') or {}).get('method', '')
            installable = method in ('pip', 'pip_pkg', 'pip_url', 'npm')
            return (0 if installable else 1, -scored.get(name, 0.0))
        top_tools = sorted(scored, key=rank_key)
        if not top_tools:
            specs = list(lookup.values())
            return self._build_tool_prompt(specs) + f'\n\nTask: {query}'
        specs = [lookup[n] for n in top_tools if n in lookup]
        tool_block = self._build_tool_prompt(specs)
        return f'{tool_block}\n\nTask: {query}'

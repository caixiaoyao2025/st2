import json

with open('interactive_agent.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Step 1: update KNOWN_AGENTS to dict with init
    if 'KNOWN_AGENTS' in src and 'PROBE_ORDER' in src:
        cell['source'] = [
            "import os, subprocess\n",
            "from IPython.display import display, Markdown\n",
            "import ipywidgets as widgets\n",
            "\n",
            "# ---- Known agent mappings: method + init code + register method ----\n",
            "KNOWN_AGENTS = {\n",
            "    'biomni': {'method': 'go', 'register': 'add_tool',\n",
            "               'init': 'from biomni.agent.A1 import A1; agent = A1(path=\"/content/_biomni_data\", expected_data_lake_files=[])'},\n",
            "    'cellagent': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "    'geneagent': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "    'crispr': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "    'biochatter': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "    'langchain': {'method': 'invoke', 'register': 'add_tool', 'init': ''},\n",
            "    'smolagents': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "    'dspy': {'method': 'forward', 'register': 'add_tool', 'init': ''},\n",
            "    'crewai': {'method': 'kickoff', 'register': 'add_tool', 'init': ''},\n",
            "    'metagpt': {'method': 'run', 'register': 'add_tool', 'init': ''},\n",
            "}\n",
            "PROBE_ORDER = ['go', 'run', 'execute', 'predict', 'forward', 'invoke']\n",
            "\n",
        ]
        break

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Step 3: use dict format
    if 'Known agent' in src and 'KNOWN_AGENTS.items()' in src:
        cell['source'] = [
            "from IPython.display import display, Markdown\n",
            "\n",
            "agent_class = (schema.get('agent_class') or '').lower()\n",
            "\n",
            "# Layer 1: known agent -> direct mapping\n",
            "exec_method = None\n",
            "for pat, info in KNOWN_AGENTS.items():\n",
            "    if pat in agent_class or pat in agent_dir.lower():\n",
            "        exec_method = info['method']\n",
            "        display(Markdown(f'**Known agent** `{pat}` -> `{exec_method}`'))\n",
            "        break\n",
            "\n",
            "# Layer 2: scan source for .go()/.run() hints\n",
            "if exec_method is None:\n",
            "    hits = {}\n",
            "    for root, dirs, files in os.walk(agent_dir):\n",
            "        dirs[:] = [d for d in dirs if d not in ('.git','__pycache__','node_modules','.venv')]\n",
            "        for f in files:\n",
            "            if f.endswith('.py'):\n",
            "                try:\n",
            "                    text = open(os.path.join(root,f), encoding='utf-8', errors='replace').read()\n",
            "                except: continue\n",
            "                for m in PROBE_ORDER:\n",
            "                    hits[m] = hits.get(m, 0) + text.count(f'.{m}(')\n",
            "    if any(v > 0 for v in hits.values()):\n",
            "        exec_method = max(hits, key=hits.get)\n",
            "        display(Markdown(f'**Source hint** -> `{exec_method}` ({hits[exec_method]} occurrences)'))\n",
            "\n",
            "# Layer 3: default\n",
            "if exec_method is None:\n",
            "    exec_method = 'run'\n",
            "    display(Markdown('**No signal found.** Defaulting to `run`.'))\n",
            "\n",
            "get_ipython().user_ns['exec_method_override'] = exec_method\n",
        ]
        break

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Step 5: known agent uses hardcoded init, then register_tools
    if 'create_agent' in src and 'generate_wiring' in src:
        cell['source'] = [
            "import os, sys, json, importlib\n",
            "from IPython.display import display, Markdown\n",
            "\n",
            "from agent_connector.generator import generate_wiring, load_wrappers\n",
            "\n",
            "schema['execution_method'] = exec_method_override\n",
            "\n",
            "# Generate wiring\n",
            "wiring_dir = os.path.join(ST2_DIR, 'wiring')\n",
            "wiring = generate_wiring(tools, schema, out_dir=wiring_dir)\n",
            "display(Markdown('Wiring mode: `' + wiring['mode'] + '`'))\n",
            "\n",
            "# Load wrappers\n",
            "sys.path.insert(0, wiring_dir)\n",
            "sys.path.insert(0, agent_dir)\n",
            "reg_style = schema.get('registration_style') or 'object'\n",
            "wrappers = load_wrappers(package_name='generated_tools', registration_style=reg_style)\n",
            "for w in wrappers:\n",
            "    if not hasattr(w, 'name'): w.name = getattr(w, '__name__', None)\n",
            "display(Markdown(f'**{len(wrappers)} wrappers loaded**'))\n",
            "\n",
            "# Create agent\n",
            "agent = None\n",
            "\n",
            "# Layer 1: known agent -> hardcoded init\n",
            "known_info = None\n",
            "for pat, info in KNOWN_AGENTS.items():\n",
            "    if pat in agent_dir.lower() or pat in (schema.get('agent_class') or '').lower():\n",
            "        known_info = info\n",
            "        break\n",
            "\n",
            "if known_info and known_info.get('init'):\n",
            "    try:\n",
            "        local_ns = {}\n",
            "        exec(known_info['init'], local_ns)\n",
            "        agent = local_ns['agent']\n",
            "        display(Markdown(f'**Known agent** initialized: `{type(agent).__name__}`'))\n",
            "    except Exception as e:\n",
            "        display(Markdown(f'Known agent init failed: `{e}`'))\n",
            "\n",
            "# Layer 2: adapter.create_agent()\n",
            "if agent is None:\n",
            "    from agent_connector.generator import load_adapter\n",
            "    adapter_cls = load_adapter(schema.get('agent_class') or 'Agent',\n",
            "                               adapter_path=wiring['artifacts']['adapter'])\n",
            "    try:\n",
            "        adapter = adapter_cls()\n",
            "        agent = adapter.create_agent()\n",
            "        display(Markdown(f'**Agent created:** `{type(agent).__name__}`'))\n",
            "    except Exception as e:\n",
            "        display(Markdown(f'`create_agent()` failed: `{e}`'))\n",
            "\n",
            "# Layer 3: DynamicAgent fallback\n",
            "if agent is None:\n",
            "    class DynamicAgent:\n",
            "        def __init__(self): self.tools = []\n",
            "    DynamicAgent.add_tool = lambda self, t: self.tools.append(t)\n",
            "    agent = DynamicAgent()\n",
            "    display(Markdown(f'**Fallback:** DynamicAgent'))\n",
            "\n",
            "# Register tools\n",
            "reg_method = (known_info or {}).get('register') or schema.get('registration_method') or 'add_tool'\n",
            "if hasattr(agent, reg_method):\n",
            "    for w in wrappers:\n",
            "        getattr(agent, reg_method)(w)\n",
            "    display(Markdown(f'Injected {len(wrappers)} tools via `{reg_method}()`'))\n",
            "else:\n",
            "    display(Markdown(f'Warning: `{reg_method}` not found on agent'))\n",
            "\n",
            "# Verify exec method\n",
            "is_known = known_info is not None\n",
            "if is_known:\n",
            "    display(Markdown(f'**Known agent** -> method `{exec_method_override}` (trusted)'))\n",
            "else:\n",
            "    probe = None\n",
            "    for m in PROBE_ORDER:\n",
            "        if m == '__call__':\n",
            "            if callable(agent): probe = m; break\n",
            "        elif hasattr(agent, m) and callable(getattr(agent, m)):\n",
            "            probe = m; break\n",
            "    if probe:\n",
            "        display(Markdown(f'**Probe OK:** `agent.{probe}()` exists'))\n",
            "        get_ipython().user_ns['exec_method_override'] = probe\n",
            "    else:\n",
            "        tried = ', '.join(f'`{m}()`' for m in PROBE_ORDER)\n",
            "        display(Markdown(f'**Probe FAILED:** tried {tried}'))\n",
            "        display(Markdown('Go to **Step 5b** to provide your agent init code manually.'))\n",
            "\n",
            "get_ipython().user_ns['agent'] = agent\n",
            "get_ipython().user_ns['wrappers'] = wrappers\n",
        ]
        break

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Step 5b: use dict format
    if 'already_ok' in src and 'KNOWN_AGENTS' in src:
        cell['source'] = [
            "from IPython.display import display, Markdown\n",
            "import ipywidgets as widgets\n",
            "\n",
            "already_ok = known_info is not None\n",
            "if not already_ok:\n",
            "    for m in PROBE_ORDER:\n",
            "        if m == '__call__':\n",
            "            if callable(agent): already_ok = True; break\n",
            "        elif hasattr(agent, m) and callable(getattr(agent, m)):\n",
            "            already_ok = True; break\n",
            "\n",
            "if already_ok:\n",
            "    display(Markdown('Step 5 succeeded. **Skipping.**'))\n",
            "else:\n",
            "    display(Markdown('### Agent execution method not recognized\\n\\n'\n",
            "        '**Tried:** ' + ', '.join(f'`{m}`' for m in PROBE_ORDER) + '\\n\\n'\n",
            "        '**Please paste your agent init code below:**'))\n",
            "    code_area = widgets.Textarea(\n",
            "        value='# from my_agent import MyAgent\\n# agent = MyAgent(model=\"gpt-4\")\\n# agent.add_tools(wrappers)\\n',\n",
            "        placeholder='agent = MyAgent(...)',\n",
            "        layout=widgets.Layout(width='90%', height='150px'))\n",
            "    method_input = widgets.Dropdown(\n",
            "        options=['run', 'execute', 'go', 'predict', 'forward', 'invoke', '__call__'],\n",
            "        value='run', description='Exec method:', layout=widgets.Layout(width='60%'))\n",
            "    apply_btn = widgets.Button(description='Apply', button_style='primary')\n",
            "    out = widgets.Output()\n",
            "\n",
            "    def on_apply(_):\n",
            "        out.clear_output()\n",
            "        with out:\n",
            "            try:\n",
            "                local_ns = {'wrappers': wrappers, 'agent_dir': agent_dir}\n",
            "                exec(code_area.value, local_ns)\n",
            "                new_agent = local_ns.get('agent')\n",
            "                if new_agent is None:\n",
            "                    display(Markdown('Error: no `agent` variable found.')); return\n",
            "                method = method_input.value\n",
            "                fn = getattr(new_agent, method, None) if method != '__call__' else new_agent\n",
            "                if fn is None or not callable(fn):\n",
            "                    display(Markdown(f'Error: `{method}` not callable.')); return\n",
            "                get_ipython().user_ns['agent'] = new_agent\n",
            "                get_ipython().user_ns['exec_method_override'] = method\n",
            "                agent = new_agent\n",
            "                display(Markdown(f'**Done!** Agent=`{type(new_agent).__name__}` method=`{method}`'))\n",
            "            except Exception as e:\n",
            "                display(Markdown(f'Error: `{e}`'))\n",
            "\n",
            "    apply_btn.on_click(on_apply)\n",
            "    display(widgets.VBox([code_area, method_input, apply_btn, out]))\n",
        ]
        break

with open('interactive_agent.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done:', len(nb['cells']), 'cells')

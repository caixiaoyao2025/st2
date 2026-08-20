"""Auto-generated adapter for agent class: react"""
import importlib as _importlib


class reactAdapter:
    """Unified adapter: create_agent / register_tools / run.

    Auto-generated from the agent's source code.  The execution entrypoint
    (go) was detected by scanner.py.
    """

    def __init__(self, agent=None, agent_class_path="biomni.agent.react", agent_class_name="react"):
        self._agent_class_path = agent_class_path
        self._agent_class_name = agent_class_name
        self._reg_method = "add_tool"
        self._exec_method = "go"
        self._init_defaults = {'path': None, 'llm': None, 'use_tool_retriever': None, 'timeout_seconds': None}
        self.agent = agent

    # -- lifecycle -----------------------------------------------------------

    def create_agent(self, **overrides):
        """Import the agent class and instantiate it with detected defaults."""
        mod = _importlib.import_module(self._agent_class_path)
        cls = getattr(mod, self._agent_class_name)
        kwargs = dict(self._init_defaults)
        import os, inspect
        _model = os.environ.get("OPENAI_MODEL", "")
        _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("WESTLAKE_API_KEY")
        _base_url = os.environ.get("OPENAI_BASE_URL", "")
        for k in list(kwargs):
            kl = k.lower()
            if kl in ("llm", "model") and not kwargs[k]:
                kwargs[k] = _model or "gpt-4o"
            elif kl == "api_key" and not kwargs[k]:
                kwargs[k] = _api_key
            elif kl == "base_url" and not kwargs[k]:
                kwargs[k] = _base_url
            elif kl == "source" and not kwargs[k]:
                kwargs[k] = "Custom"
        try:
            sig = inspect.signature(cls.__init__)
            valid = set(sig.parameters.keys()) - {"self"}
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        except (ValueError, TypeError):
            pass
        kwargs.update(overrides)
        self.agent = cls(**kwargs) if kwargs else cls()
        return self.agent

    def register_tools(self, agent, tools):
        """Inject tools: try agent's native method, fallback to direct append."""
        target = agent or self.agent
        # Try agent's native registration method
        reg_fn = getattr(target, self._reg_method, None)
        if reg_fn:
            for t in tools:
                try:
                    reg_fn(t)
                except Exception:
                    # Fallback: direct append to tools list
                    if not hasattr(target, 'tools') or target.tools is None:
                        target.tools = []
                    target.tools.append(t)
        else:
            if not hasattr(target, 'tools') or target.tools is None:
                target.tools = []
            for t in tools:
                target.tools.append(t)
        return len(tools)

    def install_tools(self, tools):
        """Legacy: inject tools into self.agent (backward compat)."""
        return self.register_tools(self.agent, tools)

    def run(self, agent=None, prompt=""):
        """Execute the agent with a prompt via its detected entrypoint."""
        target = agent or self.agent
        return getattr(target, self._exec_method)(prompt)

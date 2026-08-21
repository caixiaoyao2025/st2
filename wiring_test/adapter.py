"""Auto-generated adapter for agent class: Agent"""
import importlib as _importlib


class AgentAdapter:
    """Unified adapter: create_agent / register_tools / run.

    Auto-generated from the agent's source code.  The execution entrypoint
    (run) was detected by scanner.py.
    """

    def __init__(self, agent=None, agent_class_path="", agent_class_name="Agent"):
        self._agent_class_path = agent_class_path
        self._agent_class_name = agent_class_name
        self._reg_method = "add_tool"
        self._exec_method = "run"
        self._init_defaults = {}
        self.agent = agent

    # -- lifecycle -----------------------------------------------------------

    def create_agent(self, **overrides):
        """Import the agent class and instantiate it with detected defaults."""
        import os, inspect
        _model = os.environ.get("OPENAI_MODEL", "")
        _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("WESTLAKE_API_KEY")
        _base_url = os.environ.get("OPENAI_BASE_URL", "")

        # --- Set BIOMNI env vars BEFORE importing so default_config picks them up ---
        if _model and not os.environ.get("BIOMNI_LLM"):
            os.environ["BIOMNI_LLM"] = _model
        if _api_key and not os.environ.get("BIOMNI_CUSTOM_API_KEY"):
            os.environ["BIOMNI_CUSTOM_API_KEY"] = _api_key
        if _base_url and not os.environ.get("BIOMNI_CUSTOM_BASE_URL"):
            os.environ["BIOMNI_CUSTOM_BASE_URL"] = _base_url
        if not os.environ.get("BIOMNI_SOURCE"):
            os.environ["BIOMNI_SOURCE"] = "Custom"

        mod = _importlib.import_module(self._agent_class_path)
        cls = getattr(mod, self._agent_class_name)

        # --- Monkey-patch default_config in case it was created before env vars ---
        try:
            from biomni.config import default_config as _dc
            if _model:
                _dc.llm = _model
            if _base_url:
                _dc.base_url = _base_url
            if _api_key:
                _dc.api_key = _api_key
            _dc.source = "Custom"
        except Exception:
            pass

        kwargs = dict(self._init_defaults)
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

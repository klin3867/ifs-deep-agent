"""
prompt_loader.py - Simple template resolver for prompts/

Loads markdown templates and resolves ${VARIABLE} placeholders.
Follows learn-claude-code philosophy: "The model is 80%. Code is 20%."

Usage:
    loader = PromptLoader("./src/prompts", {"VAR": "value"})
    prompt = loader.load("system-prompt-main.md")
"""

import re
import yaml
from pathlib import Path
from typing import Optional


class PromptLoader:
    """Load and resolve prompt templates from prompts/ directory."""

    def __init__(self, prompts_dir: str, variables: Optional[dict] = None):
        """
        Initialize loader with prompts directory and base variables.

        Args:
            prompts_dir: Path to prompts/ directory
            variables: Base variables for ${VAR} replacement
        """
        self.prompts_dir = Path(prompts_dir)
        self.variables = variables or {}
        self._cache: dict[str, str] = {}

    def load(self, name: str, extra_vars: Optional[dict] = None) -> str:
        """
        Load prompt file, strip front-matter, resolve variables.

        Args:
            name: Filename within prompts_dir (e.g., "system-prompt-main.md")
            extra_vars: Additional variables to merge for this load

        Returns:
            Resolved prompt content as string
        """
        # Check cache first
        cache_key = f"{name}:{hash(frozenset((extra_vars or {}).items()))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.prompts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        content = path.read_text(encoding="utf-8")

        # Strip front-matter (<!-- ... -->)
        content = re.sub(r"<!--[\s\S]*?-->", "", content).strip()

        # Merge variables
        merged_vars = {**self.variables, **(extra_vars or {})}

        # Resolve ${VAR} and ${VAR.property} patterns
        content = self._resolve_variables(content, merged_vars)

        # Strip remaining complex template expressions (conditionals like ${X ? 'a' : 'b'})
        content = re.sub(r"\$\{[^}]+\}", "", content)

        # Clean up multiple blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        self._cache[cache_key] = content
        return content

    def _resolve_variables(self, content: str, variables: dict) -> str:
        """Replace ${VAR} and ${VAR.property} patterns."""

        def replace(match: re.Match) -> str:
            key = match.group(1)

            # Handle nested property access: VAR.property
            if "." in key:
                parts = key.split(".")
                val = variables.get(parts[0], {})
                for p in parts[1:]:
                    if isinstance(val, dict):
                        val = val.get(p, "")
                    else:
                        val = ""
                        break
                return str(val)

            # Simple variable
            return str(variables.get(key, ""))

        # Match ${UPPER_CASE_VAR} or ${VAR.property}
        pattern = r"\$\{([A-Z_][A-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\}"
        return re.sub(pattern, replace, content)

    def load_multiple(self, names: list[str], extra_vars: Optional[dict] = None) -> str:
        """Load and concatenate multiple prompt files."""
        parts = []
        for name in names:
            try:
                parts.append(self.load(name, extra_vars))
            except FileNotFoundError:
                continue  # Skip missing files
        return "\n\n".join(parts)

    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()

    @classmethod
    def from_config(cls, config_path: str) -> "PromptLoader":
        """
        Create loader from config file.

        Config should have:
            prompts_dir: ./src/prompts
            variables:
                TASK_TOOL_NAME: Task
                ...
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        prompts_dir = config.get("prompts_dir", "./src/prompts")
        variables = config.get("variables", {})

        # Resolve prompts_dir relative to config file
        if not Path(prompts_dir).is_absolute():
            prompts_dir = config_path.parent / prompts_dir

        return cls(str(prompts_dir), variables)


# Convenience function
def load_variables(path: str) -> dict:
    """Load variables from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prompt_loader.py <prompt_file>")
        sys.exit(1)

    loader = PromptLoader("./src/prompts")
    content = loader.load(sys.argv[1])
    print(content[:500] + "..." if len(content) > 500 else content)

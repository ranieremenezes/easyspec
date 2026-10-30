"""
Welcome banner shown once when easyspec is imported: the CBPF logo in Jupyter notebooks, or a short text in terminals.

It can be switched off by setting the environment variable EASYSPEC_NO_BANNER=1 before importing easyspec, e.g.
    import os; os.environ["EASYSPEC_NO_BANNER"] = "1"
or in the shell:
    export EASYSPEC_NO_BANNER=1
"""

import base64
import os
import sys
from pathlib import Path

LOGO = Path(__file__).parent / "images" / "cbpf_logo.png"
INSTITUTION = "Centro Brasileiro de Pesquisas Físicas (CBPF)"
WEBSITE = "https://github.com/ranieremenezes/easyspec"


def _in_notebook():
    """True when running inside a Jupyter kernel (notebook, JupyterLab, VS Code notebooks)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _html(version):
    """HTML banner with the logo embedded (base64), on a white card so that it is readable with dark themes as well."""
    logo = base64.b64encode(LOGO.read_bytes()).decode("ascii")
    return (f'<div style="display:inline-flex;align-items:center;gap:16px;background:#ffffff;border-radius:8px;padding:8px 16px;margin:4px 0;'
            f'border:1px solid #d0d4e0;font-family:sans-serif;color:#2e3a59">'
            f'<img src="data:image/png;base64,{logo}" alt="CBPF" style="height:56px">'
            f'<div style="font-size:13px;line-height:1.5"><b>easyspec</b> v{version}<br>Developed at the {INSTITUTION}<br>'
            f'<a href="{WEBSITE}" target="_blank" style="color:#2f9aa8">{WEBSITE}</a></div></div>')


def _text(version):
    """Text banner for terminals (with colours only when the output is an interactive terminal)."""
    colour = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    # Bold and teal (the CBPF colour) only: the dark blue of the logo would be unreadable on dark terminal themes
    bold, teal, reset = ("\033[1m", "\033[38;2;47;154;168m", "\033[0m") if colour else ("",) * 3
    return f"{bold}easyspec{reset} v{version} | Developed at the {bold}{INSTITUTION}{reset} | {teal}{WEBSITE}{reset}"


def show_banner(version, force=False):

    """
    Show the banner (logo in notebooks, text in terminals). Nothing is shown if EASYSPEC_NO_BANNER is set, or in non-interactive runs
    (scripts whose output is redirected, batch jobs), unless force=True. Any failure is silent: the banner never breaks the import.
    """

    try:
        if not force and os.environ.get("EASYSPEC_NO_BANNER", "").strip().lower() not in ("", "0", "false", "no"):
            return
        if _in_notebook() and LOGO.exists():
            from IPython.display import HTML, display
            display(HTML(_html(version)))
        elif force or sys.stdout.isatty() or hasattr(sys, "ps1"):
            print(_text(version))
    except Exception:
        pass

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{filename} konnte nicht geladen werden.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    try:
        gui_module = _load_module("fetch_pick_and_place_gui.py", "fetch_pick_and_place_gui")
        window = gui_module.FetchPickAndPlaceUI()
        try:
            window.deiconify()
            window.update_idletasks()
            window.lift()
            window.focus_force()
            window.attributes("-topmost", True)
            window.after(200, lambda: window.attributes("-topmost", False))
        except Exception:
            pass
        window.mainloop()
    except Exception as exc:
        message = (
            "Fetch Pick And Place konnte nicht gestartet werden. "
            "Ursache oft: headless/Remote-Umgebung oder kein funktionierender Render-Kontext.\n"
            f"Details: {exc}"
        )
        print(message)
        raise


if __name__ == "__main__":
    main()

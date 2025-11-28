# Meta3DScene : Text → Editable 3D Scene

A simple, user‑facing tool that turns a short natural‑language prompt into an editable Blender scene and preview images.

Key points
- Input: a short text prompt (e.g., "a small cabin in a pine forest").
- Output: a project folder (named from your prompt) containing:
  - assets/ — downloaded 3D models used in the scene
  - measurement/ — thumbnails and measured asset dimensions
  - scene/ — generated Blender Python script (editable) and rendered previews
- Designed for local use with Blender; external API keys are required for some asset sources.

Quick start (Windows PowerShell)
1. Create a virtual environment and activate it:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure secrets locally (do NOT commit):
   ```powershell
   copy .env.example .env
   notepad .env   # add your API keys and BLENDER_EXECUTABLE_PATH
   ```
4. Run:
   - CLI: `python FINAL_FILE.py` and follow the prompt
   - Notebook UI: open `dashboard.ipynb` in JupyterLab and run cells

Example prompt
- Input: "a cabin surrounded by tall pine trees"
- Result: directory `a_cabin_surrounded_by_tall_pine_trees/` with `scene/generate_scene.py`, renders, and assets

Notes for users
- The generated Blender script is editable — open it in Blender to refine layout, materials, or lighting.
- Blender must be installed and reachable from the path set in your `.env`.
- Keep your real API keys in `.env` (this file is ignored by git). Share only `.env.example`.

Troubleshooting
- Blender not found: confirm `BLENDER_EXECUTABLE_PATH` in `.env`.
- Long runs: Blender operations and model downloads can take time; monitor console output.
- If an asset fails to download or measure, try a different prompt or check internet/API key.

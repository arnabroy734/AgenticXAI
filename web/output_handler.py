"""
output_handler.py

The concrete OutputHandler for this demo (src/pipeline/output_handler.py
ships none itself - see its docstring). Every artifact is copied
straight into this run's own results/<session_id>/ directory, which is
already its final location - a mounted volume, so it survives this
container restarting. A manifest (artifacts.json) records which real
filename each generic artifact key maps to, so a later request (e.g.
after a restart, with nothing in memory) can still resolve
"give me the report_pdf for session X" to a real path.
"""

import json
import os
import shutil

from pipeline.output_handler import OutputHandler


class LocalResultsOutputHandler(OutputHandler):
    def __init__(self, session_dir: str):
        self.session_dir = session_dir

    def persist(self, working_dir: str, artifact_filenames: dict) -> dict:
        final_paths = {}
        for key, filename in artifact_filenames.items():
            shutil.copy(os.path.join(working_dir, filename), os.path.join(self.session_dir, filename))
            final_paths[key] = filename

        with open(os.path.join(self.session_dir, "artifacts.json"), "w") as f:
            json.dump(final_paths, f, indent=2)

        return final_paths

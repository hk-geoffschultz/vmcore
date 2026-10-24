"""Input-agnostic video plumbing, shared across pipelines.

Everything in here answers a question about *media* — what are this file's
dimensions, how do I get frames out of it, what does Premiere need this XML
to look like, have I rendered this exact segment before. Nothing in here
knows what a good clip is, what a montage is for, or who is in frame.

That line is the whole point of the package, and it is what lets the same
code serve two unrelated pipelines: a birthday-montage tool and an
event/behind-the-scenes edit pipeline. `tests/test_no_app_imports.py`
enforces it — no module here may import anything outside the standard
library and numpy — because the rule is invisible to every other test.
Everything keeps working right up until the package stops being liftable.
"""
# The release this source tree is. pyproject.toml's `version` must say the
# same thing: tests/test_version.py fails when they differ. It is a literal
# here, and not read back from the installed metadata, so that it describes
# the code actually imported. A copy on PYTHONPATH, a vendored copy and an
# editable install made before a version bump all report their own version,
# where `importlib.metadata.version("vmcore")` would report whichever
# distribution pip happens to have recorded.
__version__ = "0.4.0"

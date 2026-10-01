# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Read the Docs -------------------------------------------------------------
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.


# -- Project information -----------------------------------------------------

project = "pytorch-ood"
copyright = "2023, K. Kirchheim"
author = "Konstantin Kirchheim"


def _read_version():
    # parsed rather than imported, so the header badge does not depend on importing torch
    import re

    with open(os.path.join("..", "src", "pytorch_ood", "__init__.py")) as f:
        return re.search(r'^__version__ = "([^"]+)"', f.read(), re.M).group(1)


release = _read_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_design",
]

# Copy only the code: strip interactive prompts and shell dollars. Line numbers
# (sphinx-gallery's line_numbers) and prompt spans are excluded by default.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": [
        "../examples/benchmarks",
        "../examples/detectors",
        "../examples/loss",
        "../examples/segmentation",
        "../examples/text",
        "../examples/osr",
        "../examples/metrics",
        "../examples/hpo",
    ],
    # path to where to save gallery generated output,
    "gallery_dirs": [
        "auto_examples/benchmarks",
        "auto_examples/detectors",
        "auto_examples/loss",
        "auto_examples/segmentation",
        "auto_examples/text",
        "auto_examples/osr",
        "auto_examples/metrics",
        "auto_examples/hpo",
    ],
    "nested_sections": False,
    "line_numbers": True,
    "min_reported_time": 20,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "pytorch-ood"

# Furo switches between these with the color mode. The GitHub pair is used
# because its dark blue (#79c0ff) sits right next to the accent color.
pygments_style = "github-light"
pygments_dark_style = "github-dark"

# Look modeled on the TorchMetrics docs: near-black canvas, raised content card,
# blue accent. Colors live in _static/custom.css; Furo only gets the brand
# colors here so its own widgets (search, toggles, links) pick them up.
html_theme_options = {
    "light_logo": "pytorch-ood-logo.svg",
    "dark_logo": "pytorch-ood-logo-white.svg",
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#1f64d6",
        "color-brand-content": "#1f64d6",
        # set here, not in custom.css: Furo derives this one from the Pygments style
        # in an inline <style> that would override the stylesheet
        "color-code-background": "#f6f8fb",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#78b4ff",
        "color-brand-content": "#78b4ff",
        "color-code-background": "#0d0d0d",
    },
    "source_repository": "https://github.com/kkirchheim/pytorch-ood/",
    "source_branch": "dev",
    "source_directory": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;700&display=swap",
    "custom.css",
]
# fixes the leading space sphinx-copybutton leaves on line-numbered code, see the file
html_js_files = ["copybutton-linenos.js"]

# include init arguments
autoclass_content = "both"
autodoc_typehints_format = "short"

# Graphviz configuration for inheritance diagrams
graphviz_output_format = "png"


def _generate_model_table():
    """
    Generate an overview table of all models in the registry, included by models.rst.
    """
    from pytorch_ood.model import get_model_info, list_models

    lines = [
        ".. list-table:: Available Pre-Trained Models",
        "   :header-rows: 1",
        "   :widths: 25 10 10 20 45",
        "   :class: model-table",
        "",
        "   * - Identifier",
        "     - Dataset",
        "     - Method",
        "     - Metrics",
        "     - Description",
    ]

    for key in list_models():
        entry = get_model_info(key)
        metrics = ", ".join(f"{k}: {v:.4f}" for k, v in entry.metrics.items()) or "—"
        description = entry.description
        if entry.source:
            description += f" (`Source <{entry.source}>`__)"
        lines += [
            f"   * - ``{entry.key}``",
            f"     - {entry.dataset}",
            f"     - {entry.loss}",
            f"     - {metrics}",
            f"     - {description}",
        ]

    os.makedirs("generated", exist_ok=True)
    with open(os.path.join("generated", "pretrained_models.rst"), "w") as f:
        f.write("\n".join(lines) + "\n")


_generate_model_table()


def _component_counts():
    """
    Counts for the landing page, as ``|n_detectors|`` etc. substitutions. Rounded
    down to a multiple of five and shown as "35+", so aliases or helper classes
    caught by the heuristics below never make the page overstate anything.
    """
    import inspect

    import torch
    from torch.utils.data import Dataset

    from pytorch_ood import detector, loss, model
    from pytorch_ood.api import Detector
    from pytorch_ood.dataset import audio, img, txt

    def classes(module, base, prefix):
        # set of objects, so re-exported aliases count once
        return {
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, base)
            and not inspect.isabstract(obj)
            and obj.__module__.startswith(prefix)
        }

    counts = {
        "n_detectors": len(classes(detector, Detector, "pytorch_ood.detector")),
        "n_losses": len(classes(loss, torch.nn.Module, "pytorch_ood.loss")),
        "n_datasets": len(set().union(*(classes(m, Dataset, "pytorch_ood.dataset") for m in (img, txt, audio)))),
        "n_models": len(model.list_models()),
    }
    return "\n".join(f".. |{k}| replace:: {v // 5 * 5}+" for k, v in counts.items())


rst_epilog = _component_counts()


def _skip_hpo_members(app, what, name, obj, skip, options):
    """
    Keep the hyperparameter-optimization interface from cluttering every detector
    page. The ``get_hyperparameters``/``set_hyperparameters`` methods are generic
    boilerplate inherited from :class:`~pytorch_ood.api.Detector`, and the inherited
    empty ``hyperparameter_space`` adds nothing. Detectors that define a real search
    space (e.g. ASH, KNN, ReAct) keep showing it.
    """
    if name in ("get_hyperparameters", "set_hyperparameters"):
        return True
    if name == "hyperparameter_space" and not obj:
        return True
    return skip


def _sidebar_with_sections(app, pagename, templatename, context, doctree):
    """
    Furo builds its sidebar with ``toctree(titles_only=True, maxdepth=-1)``, which
    lists pages only. Single-page references like the detector overview then show no
    children, while the example galleries (one page per example) do. Showing section
    headings two levels deep makes both expandable; the depth limit keeps headings
    inside individual examples out of the sidebar.
    """
    toctree = context.get("toctree")
    if toctree is None:
        return

    def toctree_with_sections(**kwargs):
        kwargs.update(titles_only=False, maxdepth=3)
        return toctree(**kwargs)

    context["toctree"] = toctree_with_sections


# Pages whose .rst is only an ``automodule`` directive: the text a reader wants to
# edit lives in the package docstring.
_DOCSTRING_PAGES = {
    "detector": "src/pytorch_ood/detector/__init__.py",
    "augmentations": "src/pytorch_ood/augment/__init__.py",
    "models": "src/pytorch_ood/model/__init__.py",
}


def _page_source_path(pagename):
    """
    Repository-relative path of the file a page is actually written in, or ``None``
    to keep Furo's default (the page's own .rst file).
    """
    if pagename in _DOCSTRING_PAGES:
        return _DOCSTRING_PAGES[pagename]

    # sphinx-gallery renders examples/<dir>/<name>.py into auto_examples/<dir>/<name>.rst,
    # and each gallery's index.rst from its README.rst.
    for src, dst in zip(sphinx_gallery_conf["examples_dirs"], sphinx_gallery_conf["gallery_dirs"]):
        if pagename.startswith(dst + "/"):
            rel = pagename[len(dst) + 1 :]
            # examples_dirs are relative to docs/; make them relative to the repository
            src_dir = os.path.normpath(os.path.join("docs", src)).replace(os.sep, "/")
            if rel == "index" or rel.endswith("/index"):
                return f"{src_dir}/{rel[: -len('index')]}README.rst"
            return f"{src_dir}/{rel}.py"
    return None


def _source_links(app, pagename, templatename, context, doctree):
    """
    Point Furo's "view/edit this page" buttons at the file a page is written in
    (see _templates/components/). Generated pages whose source cannot be found get
    no buttons rather than a link to a file that does not exist in the repository.
    """
    path = _page_source_path(pagename)
    if path is None:
        return
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    context["page_source_path"] = path if os.path.isfile(os.path.join(repo_root, path)) else ""


def setup(app):
    app.connect("autodoc-skip-member", _skip_hpo_members)
    app.connect("html-page-context", _source_links)
    # must run before Furo's own html-page-context handler (default priority 500)
    app.connect("html-page-context", _sidebar_with_sections, priority=400)

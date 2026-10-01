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
import datetime
import functools
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Read the Docs -------------------------------------------------------------
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
# URL of the version being built on Read the Docs; empty in local builds.
_CANONICAL_URL = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
# Read the Docs no longer sets this itself; without it pages carry no canonical
# link, and search engines index old versions next to the current one.
html_baseurl = _CANONICAL_URL

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Brand colors (light / dark theme). custom.css defines the same pair as
# --color-accent; the inheritance diagrams use the light one on both themes.
_BRAND = "#1f64d6"
_BRAND_DARK = "#78b4ff"

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
# from the first commit to the year of the build
copyright = f"2021–{datetime.date.today().year}, K. Kirchheim"
author = "Konstantin Kirchheim"

# conf.py imports the package anyway (component counts, model registry)
from pytorch_ood import __version__ as release  # noqa: E402

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
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
    "notfound.extension",
]

# Links types from other projects (torch.Tensor, DataLoader, ...) to their docs.
# With -W, an unreachable inventory fails the build, hence the generous timeout.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}
intersphinx_timeout = 30

# Link previews (Slack, Mastodon, GitHub, ...). On Read the Docs the canonical
# URL of the version being built, otherwise the stable docs.
ogp_site_url = _CANONICAL_URL or "https://pytorch-ood.readthedocs.io/en/stable/"
ogp_image = "_static/og-image.png"
ogp_image_alt = "pytorch-ood: Out-of-Distribution Detection for PyTorch"

# 404 page served by Read the Docs. Its links are absolute; the prefix defaults to
# the path of the version being built there, and to the site root locally.
notfound_urls_prefix = urlparse(_CANONICAL_URL).path or "/"
notfound_context = {
    "title": "Page not found",
    "body": f"""
<h1>Page not found</h1>
<p>This page does not exist, or it has moved: the API reference now has one page
per component. Try the search above, or start from one of these pages:</p>
<ul>
  <li><a href="{notfound_urls_prefix}index.html">Home</a></li>
  <li><a href="{notfound_urls_prefix}getting_started.html">Getting Started</a></li>
  <li><a href="{notfound_urls_prefix}detector.html">Detectors</a></li>
  <li><a href="{notfound_urls_prefix}auto_examples/detectors/index.html">Examples</a></li>
</ul>
""",
}

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
    # examples pick their section's thumbnail (docs/tools/gallery_thumbnails.py) with
    # a "# sphinx_gallery_thumbnail_path" comment, which is hidden in the rendered code
    "remove_config_comments": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# generated/ holds fragments pulled in with ``.. include::``, not pages of their own.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "generated"]

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
    "dark_logo": "pytorch-ood-logo.svg",
    "light_css_variables": {
        "color-brand-primary": _BRAND,
        "color-brand-content": _BRAND,
        # set here, not in custom.css: Furo derives this one from the Pygments style
        # in an inline <style> that would override the stylesheet
        "color-code-background": "#f6f8fb",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": _BRAND_DARK,
        "color-brand-content": _BRAND_DARK,
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
html_css_files = ["custom.css"]
html_js_files = [
    # fixes the leading space sphinx-copybutton leaves on line-numbered code, see the file
    "copybutton-linenos.js",
    # search shortcuts and the Read the Docs search window for _templates/page.html
    ("site-header.js", {"defer": "defer"}),
]

# include init arguments
autoclass_content = "both"
autodoc_typehints_format = "short"

# Inheritance diagrams: SVG (sharp, clickable nodes) in colors that read on both
# the light and the dark theme, since an SVG embedded via <object> cannot use the
# page's CSS variables. Base classes are recolored after the build, see
# _style_inheritance_diagrams.
graphviz_output_format = "svg"
# Sphinx inserts these values into the DOT source as is, hence the extra quotes.
inheritance_graph_attrs = {
    "rankdir": '"LR"',
    "bgcolor": '"transparent"',
    "nodesep": '"0.08"',
    "ranksep": '"0.45"',
    "fontsize": '"10"',
}
inheritance_node_attrs = {
    "shape": '"box"',
    "style": '"rounded,filled"',
    "fillcolor": f'"{_BRAND}"',
    "color": f'"{_BRAND}"',
    "fontcolor": '"#ffffff"',
    "fontname": '"Helvetica"',
    "fontsize": '"10"',
    "height": '"0.28"',
    "margin": '"0.12,0.03"',
    "penwidth": '"1"',
}
inheritance_edge_attrs = {
    "color": '"#8b949e"',
    "penwidth": '"0.8"',
    "arrowsize": '"0.55"',
}


# Sections of the model registry page, in display order. Datasets missing here are
# appended under their raw registry name, so new entries are never dropped.
_REGISTRY_SECTIONS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet200": "ImageNet-200",
    "imagenet1k": "ImageNet-1k",
    "imagenet32": "ImageNet 32x32",
    "imagenet32-nocifar": "ImageNet 32x32",
}


def _generate_model_registry():
    """
    Write the model tables of docs/models/registry.rst: one section per training
    dataset, one row per model with its seeds merged into a single row.
    """
    import re
    from collections import defaultdict

    from pytorch_ood.model import get_model_info, list_models

    def base_key(entry):
        suffix = f"/{entry.seed}"
        return (
            entry.key[: -len(suffix)] if entry.seed and entry.key.endswith(suffix) else entry.key
        )

    def shared_description(entry):
        # fine-tuned models name the checkpoint of the same seed they start from
        if not entry.seed:
            return entry.description
        return re.sub(rf"(\S+)/{entry.seed}\b", r"\1 (same seed)", entry.description)

    # section -> (base key, description, source, host) -> entries of the seeds
    rows = defaultdict(lambda: defaultdict(list))
    for key in list_models():
        entry = get_model_info(key)
        section = _REGISTRY_SECTIONS.get(entry.dataset, entry.dataset or "Other")
        host = urlparse(entry.url).netloc
        rows[section][(base_key(entry), shared_description(entry), entry.source, host)].append(
            entry
        )

    order = list(dict.fromkeys(_REGISTRY_SECTIONS.values()))
    sections = sorted(rows, key=lambda s: (order.index(s) if s in order else len(order), s))

    lines = []
    for section in sections:
        lines += [section, "~" * len(section), ""]
        lines += [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 70 14 16",
            "   :class: model-table",
            "",
            "   * - Model",
            "     - Seeds",
            "     - Accuracy",
        ]
        for (key, description, source, host), entries in rows[section].items():
            seeds = " ".join(f"``{e.seed}``" for e in entries if e.seed) or "—"
            # rounded first, so seeds that agree to one decimal show a single value
            accuracies = sorted(
                {
                    f"{100 * e.metrics['best_accuracy']:.1f}"
                    for e in entries
                    if "best_accuracy" in e.metrics
                },
                key=float,
            )
            if not accuracies:
                accuracy = "—"
            elif len(accuracies) == 1:
                accuracy = f"{accuracies[0]} %"
            else:
                accuracy = f"{accuracies[0]}–{accuracies[-1]} %"
            if source:
                description += f" `Source <{source}>`__"
            if host != "huggingface.co":
                # most weights live on Hugging Face; flag the ones downloaded elsewhere
                description += f" :bdg-secondary-line:`{host}`"
            lines += [
                f"   * - ``{key}``",
                "",
                "       .. container:: model-description",
                "",
                f"          {description}",
                f"     - {seeds}",
                f"     - {accuracy}",
            ]
        lines.append("")

    # rewritten only on change, so an unchanged registry does not mark the page outdated
    content = "\n".join(lines) + "\n"
    path = os.path.join("generated", "model_registry.rst")
    if not os.path.exists(path) or open(path).read() != content:
        os.makedirs("generated", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)


_generate_model_registry()


def _component_counts():
    """
    Counts for the landing page, as ``|n_detectors|`` etc. substitutions. Rounded
    down to a multiple of five and shown as "35+", so aliases or helper classes
    caught by the heuristics below never make the page overstate anything.
    """
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
        "n_datasets": len(
            set().union(*(classes(m, Dataset, "pytorch_ood.dataset") for m in (img, txt, audio)))
        ),
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
    headings makes both expandable. Four levels are needed for Datasets > Image >
    Classification > dataset; the depth limit keeps headings inside individual
    examples out of the sidebar, and API object entries (a page's class, rendered
    as code) are dropped since they only repeat the page title.
    """
    toctree = context.get("toctree")
    if toctree is None:
        return

    def toctree_with_sections(**kwargs):
        from bs4 import BeautifulSoup  # a dependency of Furo

        kwargs.update(titles_only=False, maxdepth=4)
        soup = BeautifulSoup(toctree(**kwargs), "html.parser")
        for link in soup.select("li > a > code"):
            link.parent.parent.decompose()
        for empty in soup.select("li > ul"):
            if not empty.find("li"):
                empty.decompose()
        return str(soup)

    context["toctree"] = toctree_with_sections


@dataclass(frozen=True)
class _SplitSection:
    """
    Content that moved out of the page ``overview`` into the pages whose names start
    with ``prefix``, usually one page per component of ``packages``. Drives the
    coverage warning and the redirects for links to the former anchors.
    """

    overview: str
    prefix: str
    packages: Tuple[str, ...]
    # which exported objects need their own page
    needs_page: Callable[[Any], bool]
    # former section anchors of the overview page: anchor -> (docname, anchor or None)
    legacy_anchors: Dict[str, Tuple[str, Optional[str]]] = field(default_factory=dict)
    # exported names that are deliberately left out of the docs
    ignore: Tuple[str, ...] = ()


def _is_detector(obj):
    from pytorch_ood.api import Detector

    return inspect.isclass(obj) and issubclass(obj, Detector)


def _is_class_or_function(obj):
    return inspect.isclass(obj) or inspect.isfunction(obj)


def _is_dataset(obj):
    from torch.utils.data import Dataset

    return inspect.isclass(obj) and issubclass(obj, Dataset) and not inspect.isabstract(obj)


_SPLIT_SECTIONS = [
    _SplitSection(
        overview="detector",
        prefix="detectors/",
        packages=("pytorch_ood.detector",),
        needs_page=_is_detector,
        legacy_anchors={
            "api": ("detectors/api", None),
            "overview": ("detectors/api", "class-hierarchy"),
        },
    ),
    _SplitSection(
        overview="models",
        prefix="models/",
        packages=("pytorch_ood.model",),
        needs_page=_is_class_or_function,
        legacy_anchors={"pre-trained-models": ("models/registry", None)},
    ),
    _SplitSection(
        overview="data",
        prefix="datasets/",
        packages=(
            "pytorch_ood.dataset.img",
            "pytorch_ood.dataset.txt",
            "pytorch_ood.dataset.audio",
            "pytorch_ood.dataset.ossim",
        ),
        needs_page=_is_dataset,
    ),
    # not a package: "Getting Started" was the second half of info.rst
    _SplitSection(
        overview="info",
        prefix="getting_started",
        packages=(),
        needs_page=lambda obj: False,
    ),
    _SplitSection(
        overview="losses",
        prefix="losses/",
        packages=("pytorch_ood.loss",),
        needs_page=_is_class_or_function,
    ),
    _SplitSection(
        overview="augmentations",
        prefix="augmentations/",
        packages=("pytorch_ood.augment.img",),
        needs_page=_is_class_or_function,
    ),
    _SplitSection(
        overview="benchmark",
        prefix="benchmarks/",
        packages=("pytorch_ood.benchmark",),
        needs_page=_is_class_or_function,
        # headings of the former single page that the new page titles do not reproduce;
        # repeated headings (the second "CIFAR-10", ...) had generated ids and are not mapped
        legacy_anchors={
            "api": ("benchmarks/api", None),
            "image": ("benchmark", "odin"),
            "cifar-10": ("benchmarks/cifar10_odin", None),
            "cifar-100": ("benchmarks/cifar100_odin", None),
            "imagenet": ("benchmarks/imagenet_openood", None),
        },
    ),
    _SplitSection(
        overview="utils",
        prefix="utils/",
        packages=("pytorch_ood.utils",),
        needs_page=_is_class_or_function,
        legacy_anchors={"transformations": ("utils/transforms", None)},
        # internal helpers that leak into the package through ``from .utils import *``
        ignore=(
            "apply_reduction",
            "evaluate_energy_logistic_loss",
            "pairwise_distances",
            "to_np",
            "torch_get_distances",
        ),
    ),
]


def _page_source_path(pagename):
    """
    Repository-relative path of the file a page is actually written in, or ``None``
    to keep Furo's default (the page's own .rst file).
    """
    if any(pagename.startswith(s.prefix) for s in _SPLIT_SECTIONS):
        return _component_page_source(pagename)

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


@functools.lru_cache(maxsize=None)
def _component_page_source(pagename):
    """
    Source file of a component page that consists only of a title and autodoc
    directives which all document the same file. Pages with prose of their own keep
    the default (their .rst file).
    """
    import importlib
    import re

    # relative to this file: sphinx-build may run from the repository root (CI)
    with open(os.path.join(_REPO_ROOT, "docs", pagename + ".rst")) as f:
        lines = f.read().splitlines()
    body = lines[2:]  # title and underline
    # unindented lines that are not directives or comments are prose
    if any(line and not line[0].isspace() and not line.startswith("..") for line in body):
        return None

    files = set()
    for line in body:
        m = re.match(r"\.\.\s+auto(?:module|class|function)::\s+(\S+)", line)
        if m is None:
            continue
        name = m.group(1)
        try:
            obj = importlib.import_module(name)
        except ImportError:
            module, attr = name.rsplit(".", 1)
            obj = getattr(importlib.import_module(module), attr)
        # resolves packages to their __init__.py and re-exported objects to their module
        files.add(inspect.getsourcefile(obj))
    if len(files) != 1:
        return None
    return os.path.relpath(files.pop(), _REPO_ROOT).replace(os.sep, "/")


def _source_links(app, pagename, templatename, context, doctree):
    """
    Point Furo's "view/edit this page" buttons at the file a page is written in
    (see _templates/components/). Generated pages whose source cannot be found get
    no buttons rather than a link to a file that does not exist in the repository.
    """
    path = _page_source_path(pagename)
    if path is None:
        return
    context["page_source_path"] = path if os.path.isfile(os.path.join(_REPO_ROOT, path)) else ""


def _moved_anchors(env, section):
    """
    Map every documented object, module and section anchor that lives on one of the
    section's component pages to that page, for links into the former single page.
    """
    from docutils import nodes

    def on_component_page(docname):
        return docname.startswith(section.prefix)

    py = env.get_domain("py")
    targets = {}
    for entry in py.objects.values():
        # aliases (canonical names) have no anchor of their own; modules are added below
        if on_component_page(entry.docname) and not entry.aliased and entry.objtype != "module":
            targets[entry.node_id] = (entry.docname, entry.node_id)
    for entry in py.modules.values():
        if on_component_page(entry.docname):
            targets[entry.node_id] = (entry.docname, entry.node_id)
    for docname, title in env.titles.items():
        if on_component_page(docname):
            # detector.html#energy-based-ebo -> detectors/energy.html
            targets[nodes.make_id(title.astext())] = (docname, None)
            # sections within a page keep their ids, e.g. #representation-interface
            for node in env.get_doctree(docname).findall(nodes.section):
                for node_id in node["ids"]:
                    targets.setdefault(node_id, (docname, node_id))
    for anchor, target in section.legacy_anchors.items():
        targets.setdefault(anchor, target)
    # never redirect an anchor that still exists on the overview itself
    for node in env.get_doctree(section.overview).findall(nodes.Element):
        for node_id in node.get("ids", []):
            targets.pop(node_id, None)
    return targets


def _check_component_pages(app, env):
    """
    Warn about exported components without a page, e.g. because the page or its
    toctree entry on the overview was forgotten.
    """
    import importlib

    from sphinx.util import logging

    py = env.get_domain("py")
    for section in _SPLIT_SECTIONS:
        documented = {
            name
            for name, entry in py.objects.items()
            if entry.docname.startswith(section.prefix) and not entry.aliased
        }
        # aliases (e.g. ImageNet1K_OpenOOD = ImageNet_OpenOOD) count as documented
        # when the object is documented under another name
        documented_objects = set()
        for name in documented:
            module, _, attr = name.rpartition(".")
            try:
                documented_objects.add(id(getattr(importlib.import_module(module), attr)))
            except (ImportError, AttributeError):
                pass  # methods and attributes resolve to their class, not a module
        for package_name in section.packages:
            package = importlib.import_module(package_name)
            names = getattr(package, "__all__", None) or [
                n for n in vars(package) if not n.startswith("_")
            ]
            for name in names:
                obj = getattr(package, name)
                if (
                    name not in section.ignore
                    and section.needs_page(obj)
                    and getattr(obj, "__module__", "").startswith(package_name)
                    and id(obj) not in documented_objects
                ):
                    logging.getLogger(__name__).warning(
                        f"{package_name}.{name} is not documented under docs/{section.prefix} "
                        f"(add a page and list it in docs/{section.overview}.rst)"
                    )


def _redirect_moved_anchors(app, pagename, templatename, context, doctree):
    """
    Components used to share one page per section, so external links point to anchors
    such as detector.html#pytorch_ood.detector.EnergyBased. Send those to the new pages.
    The map (up to ~40 KB) is a separate file, fetched only when the URL has an anchor.
    """
    import json

    section = next((s for s in _SPLIT_SECTIONS if s.overview == pagename), None)
    if section is None:
        return
    targets = {
        anchor: app.builder.get_target_uri(docname) + (f"#{target}" if target else "")
        for anchor, (docname, target) in _moved_anchors(app.env, section).items()
    }
    name = f"_static/moved-anchors/{pagename}.json"
    os.makedirs(os.path.join(app.outdir, "_static", "moved-anchors"), exist_ok=True)
    with open(os.path.join(app.outdir, name), "w") as f:
        json.dump(targets, f, sort_keys=True)
    app.add_js_file(
        None,
        body="(function () {"
        f"var url = {json.dumps(context['pathto'](name, 1))}, targets = null;"
        "function go() {"
        " var a = decodeURIComponent(location.hash.slice(1));"
        " if (!a) return;"
        " (targets ? Promise.resolve(targets) : fetch(url).then(function (r) { return r.json(); }))"
        " .then(function (t) { targets = t; if (t[a]) location.replace(t[a]); });"
        "}"
        "go(); window.addEventListener('hashchange', go);"
        "})();",
    )


# fill / outline of base classes in inheritance diagrams; other classes use the
# node colors from inheritance_node_attrs
_DIAGRAM_BASE_CLASS_COLORS = ("#0b2f6b", _BRAND_DARK)


def _style_inheritance_diagrams(app, exception):
    """
    Give the base classes of the library's API (pytorch_ood.api.*) a darker fill
    in the generated inheritance diagrams, so the hierarchy reads at a glance.
    Graphviz' inheritance_node_attrs apply to all nodes alike.
    """
    import glob
    import re

    from sphinx.util import logging

    if exception is not None or app.builder.format != "html":
        return
    fill, stroke = _DIAGRAM_BASE_CLASS_COLORS
    node = re.compile(r'(<(?:\w+:)?g id="node\d+" class="node">.*?</(?:\w+:)?a>)', re.S)
    # Graphviz writes the node colors from inheritance_node_attrs like this
    default = f'fill="{_BRAND}" stroke="{_BRAND}"'

    def restyle(match):
        group = match.group(1)
        if "#pytorch_ood.api." not in group:
            return group
        return group.replace(default, f'fill="{fill}" stroke="{stroke}"')

    for path in glob.glob(os.path.join(app.outdir, "_images", "inheritance-*.svg")):
        with open(path) as f:
            svg = f.read()
        styled = node.sub(restyle, svg)
        if styled == svg:
            # unchanged although the diagram has base classes and was not styled by an
            # earlier build: e.g. Graphviz changed its SVG output
            if "#pytorch_ood.api." in svg and f'fill="{fill}"' not in svg:
                logging.getLogger(__name__).warning(
                    f"no base class recolored in {os.path.basename(path)}"
                )
            continue
        with open(path, "w") as f:
            f.write(styled)


def _preview_description(app, doctree):
    """Describe component pages by their first sentence of prose in link previews.

    sphinxext-opengraph takes a page's leading text as its description and skips
    API entries (Sphinx models them as admonitions). Component pages start with
    capability badges or directly with an API entry, so it would describe them
    by the badges' alt text, or not at all.
    """
    from docutils import nodes
    from sphinx import addnodes

    section = next(iter(doctree.findall(nodes.section)), None)
    if section is None:
        return

    def is_note(node):  # API entries (desc) are Admonition subclasses, too
        return isinstance(node, nodes.Admonition) and not isinstance(node, addnodes.desc)

    # automodule puts invisible target and index nodes before the module's content;
    # some pages open with a note (e.g. on optional dependencies)
    first = next(
        (
            n
            for n in section.children
            if not isinstance(n, (nodes.title, nodes.Invisible)) and not is_note(n)
        ),
        None,
    )
    leads_with_badges = isinstance(first, (nodes.image, nodes.reference)) or (
        isinstance(first, nodes.paragraph) and first.next_node(nodes.image) is not None
    )
    if not (leads_with_badges or isinstance(first, addnodes.desc)):
        return

    def is_prose(paragraph):
        if paragraph.next_node(nodes.image) is not None:
            return False
        if paragraph.astext().startswith("Bases: "):  # added by :show-inheritance:
            return False
        return not any(
            isinstance(a, nodes.field_list) or is_note(a) for a in _ancestors(paragraph)
        )

    paragraph = next((p for p in section.findall(nodes.paragraph) if is_prose(p)), None)
    if paragraph is not None:
        text = " ".join(paragraph.astext().split())
        app.env.metadata[app.env.docname].setdefault("og:description", text[:200])


def _meta_description(app, pagename, templatename, context, doctree):
    # sphinxext-opengraph adds <meta name="description"> from the leading text
    # unless one exists; give it the same text as og:description.
    description = (context.get("meta") or {}).get("og:description")
    if description:
        from html import escape

        tag = f'<meta name="description" content="{escape(description)}" />\n'
        context["metatags"] = tag + context.get("metatags", "")


def _ancestors(node):
    while node.parent is not None:
        node = node.parent
        yield node


def setup(app):
    app.connect("doctree-read", _preview_description)
    # before sphinxext-opengraph's own handler (default priority 500)
    app.connect("html-page-context", _meta_description, priority=400)
    app.connect("build-finished", _style_inheritance_diagrams)
    app.connect("env-check-consistency", _check_component_pages)
    app.connect("html-page-context", _redirect_moved_anchors)
    app.connect("autodoc-skip-member", _skip_hpo_members)
    app.connect("html-page-context", _source_links)
    # must run before Furo's own html-page-context handler (default priority 500)
    app.connect("html-page-context", _sidebar_with_sections, priority=400)

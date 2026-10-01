// Site header behavior (see _templates/page.html), loaded deferred on every page.
// On Read the Docs, every search entry point opens RTD's search window
// (server-side search, results as you type). Elsewhere, e.g. in a local build,
// it falls back to Sphinx's search page. RTD's add-ons announce their config
// with an event, and keep it on window for scripts that run after the event.
var rtdSearch = false;
function useRtdConfig(eventData) {
  try {
    var addons = eventData.data().addons;
    rtdSearch = !!(addons && addons.search && addons.search.enabled);
  } catch (err) {
    rtdSearch = false;
  }
}
if (window.ReadTheDocsEventData) useRtdConfig(window.ReadTheDocsEventData);
document.addEventListener("readthedocs-addons-data-ready", function (e) {
  useRtdConfig(e.detail);
});
function showRtdSearch() {
  document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
}

var headerInput = document.querySelector(".site-header-search input");
headerInput.addEventListener("focus", function () {
  if (!rtdSearch) return;
  headerInput.blur();
  showRtdSearch();
});

// Ctrl/Cmd+K or "/" focuses the header search, as on most modern doc sites.
document.addEventListener("keydown", function (e) {
  var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
  if (!(((e.ctrlKey || e.metaKey) && e.key === "k") || (e.key === "/" && !typing))) return;
  if (rtdSearch) {
    e.preventDefault();
    showRtdSearch();
  } else if (headerInput.offsetParent !== null) {
    e.preventDefault();
    headerInput.focus();
  }
});
// On narrow screens the header search button opens the navigation drawer,
// which holds Furo's search field.
document.querySelectorAll(".site-header-search-button").forEach(function (button) {
  button.addEventListener("click", function () {
    if (rtdSearch) return showRtdSearch();
    document.getElementById("__navigation").checked = true;
    var input = document.querySelector(".sidebar-search");
    if (input) setTimeout(function () { input.focus(); }, 250);
  });
});

/* Result viewers. Contract for real viewers (added by later tasks):
 *   LabViewers.register(name, fn)  where fn(resultObj, container, ctx) renders into container.
 *   ctx = {task, evalId, fileUrl(name) -> "/api/evals/<task>/<eval>/file/<name>"}.
 *   LabViewers.render(viewerName, resultObj, container, ctx) dispatches by spec.viewer;
 *   unknown/unregistered names fall back to pretty-printed JSON. */
(function () {
  const registry = {};

  function jsonFallback(result, container) {
    const pre = document.createElement("pre");
    pre.className = "json";
    pre.textContent = JSON.stringify(result, null, 2);
    container.replaceChildren(pre);
  }

  function render(viewer, result, container, ctx) {
    const fn = registry[viewer] || jsonFallback;
    try {
      fn(result, container, ctx || {});
    } catch (e) {
      jsonFallback(result, container);
    }
  }

  window.LabViewers = { register: (name, fn) => { registry[name] = fn; }, render, jsonFallback };
})();

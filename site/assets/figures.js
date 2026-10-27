/* Figure behaviour: clicking a figure opens it at full window size. The viewer moves between the figures of the page and
   zooms in and out (buttons, plus and minus keys, mouse wheel, double click), and a zoomed figure can be dragged. */
(function () {
  var idx = 0, scale = 1, tx = 0, ty = 0, drag = null;
  var MIN = 0.5, MAX = 8;

  function viewerEl() { return document.getElementById("fig-viewer"); }
  function imgEl() { return viewerEl().querySelector("img"); }

  function apply() {
    var img = imgEl();
    img.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    viewerEl().querySelector(".zoomlabel").textContent = Math.round(scale * 100) + "%";
  }
  function resetZoom() { scale = 1; tx = 0; ty = 0; apply(); }
  function zoomTo(ns, px, py) {
    ns = Math.min(MAX, Math.max(MIN, ns));
    var r = viewerEl().getBoundingClientRect();
    var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    var ox = (px === undefined ? cx : px) - cx, oy = (py === undefined ? cy : py) - cy;
    tx = ox - (ns / scale) * (ox - tx);
    ty = oy - (ns / scale) * (oy - ty);
    scale = ns;
    if (scale <= 1) { tx = 0; ty = 0; }
    apply();
  }

  function show(i) {
    var links = window.__figLinks || [];
    if (!links.length) return;
    idx = (i + links.length) % links.length;
    var v = viewerEl();
    v.querySelector("img").src = links[idx].href;
    v.querySelector(".count").textContent = "Figure " + links[idx].getAttribute("data-fig") + "  (" + (idx + 1) + " of " + links.length + ")";
    v.classList.add("on");
    resetZoom();
  }
  function closeViewer() { viewerEl().classList.remove("on"); resetZoom(); }

  function build() {
    var viewer = document.createElement("div");
    viewer.id = "fig-viewer";
    viewer.innerHTML =
      "<div class='bar'><span class='count'></span><span>" +
      "<button class='zout' title='Zoom out (minus key)'>Zoom out</button>" +
      "<span class='zoomlabel'>100%</span>" +
      "<button class='zin' title='Zoom in (plus key)'>Zoom in</button>" +
      "<button class='zreset' title='Fit to window (0 key)'>Reset</button>" +
      "<button class='prev'>Previous</button><button class='next'>Next</button><button class='close'>Close</button>" +
      "</span></div><img alt='' draggable='false'>";
    document.body.appendChild(viewer);
    viewer.querySelector(".prev").onclick = function () { show(idx - 1); };
    viewer.querySelector(".next").onclick = function () { show(idx + 1); };
    viewer.querySelector(".close").onclick = closeViewer;
    viewer.querySelector(".zin").onclick = function () { zoomTo(scale * 1.4); };
    viewer.querySelector(".zout").onclick = function () { zoomTo(scale / 1.4); };
    viewer.querySelector(".zreset").onclick = resetZoom;
    viewer.addEventListener("click", function (e) { if (e.target === viewer) closeViewer(); });
    viewer.addEventListener("wheel", function (e) {
      e.preventDefault();
      zoomTo(scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15), e.clientX, e.clientY);
    }, { passive: false });
    var img = viewer.querySelector("img");
    img.addEventListener("dblclick", function (e) { if (scale > 1) resetZoom(); else zoomTo(2.5, e.clientX, e.clientY); });
    img.addEventListener("pointerdown", function (e) {
      if (scale <= 1) return;
      drag = { x: e.clientX - tx, y: e.clientY - ty }; img.setPointerCapture(e.pointerId); img.classList.add("grabbing");
    });
    img.addEventListener("pointermove", function (e) { if (drag) { tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply(); } });
    function end(e) { drag = null; img.classList.remove("grabbing"); }
    img.addEventListener("pointerup", end); img.addEventListener("pointercancel", end);
    document.addEventListener("keydown", function (e) {
      if (!viewer.classList.contains("on")) return;
      if (e.key === "Escape") closeViewer();
      else if (e.key === "ArrowLeft") show(idx - 1);
      else if (e.key === "ArrowRight") show(idx + 1);
      else if (e.key === "+" || e.key === "=") zoomTo(scale * 1.4);
      else if (e.key === "-" || e.key === "_") zoomTo(scale / 1.4);
      else if (e.key === "0") resetZoom();
    });
    return viewer;
  }

  function setup() {
    var links = Array.prototype.slice.call(document.querySelectorAll("a.figlink"));
    if (!viewerEl()) build();
    window.__figLinks = links;
    links.forEach(function (a, i) { a.onclick = function (e) { e.preventDefault(); show(i); }; });
  }
  if (window.document$) { window.document$.subscribe(setup); } else { document.addEventListener("DOMContentLoaded", setup); }
})();

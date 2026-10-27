/* Figure behaviour: clicking a figure opens it at full window size, with a viewer to move between the figures of the page. */
(function () {
  function setup() {
    var links = Array.prototype.slice.call(document.querySelectorAll("a.figlink"));
    var viewer = document.getElementById("fig-viewer");
    if (!viewer) {
      viewer = document.createElement("div"); viewer.id = "fig-viewer";
      viewer.innerHTML = "<div class='bar'><span class='count'></span><span><button class='prev'>Previous</button><button class='next'>Next</button><button class='close'>Close</button></span></div><img alt=''>";
      document.body.appendChild(viewer);
      viewer.querySelector(".prev").onclick = function () { show(idx - 1); };
      viewer.querySelector(".next").onclick = function () { show(idx + 1); };
      viewer.querySelector(".close").onclick = closeViewer;
      viewer.addEventListener("click", function (e) { if (e.target === viewer) closeViewer(); });
      document.addEventListener("keydown", function (e) {
        if (!viewer.classList.contains("on")) return;
        if (e.key === "Escape") closeViewer();
        else if (e.key === "ArrowLeft") show(idx - 1);
        else if (e.key === "ArrowRight") show(idx + 1);
      });
    }
    window.__figLinks = links;
    links.forEach(function (a, i) {
      a.onclick = function (e) { e.preventDefault(); show(i); };
    });
  }
  var idx = 0;
  function show(i) {
    var links = window.__figLinks || [];
    if (!links.length) return;
    idx = (i + links.length) % links.length;
    var viewer = document.getElementById("fig-viewer");
    viewer.querySelector("img").src = links[idx].href;
    viewer.querySelector(".count").textContent = "Figure " + links[idx].getAttribute("data-fig") + "  (" + (idx + 1) + " of " + links.length + ")";
    viewer.classList.add("on");
  }
  function closeViewer() { document.getElementById("fig-viewer").classList.remove("on"); }
  if (window.document$) { window.document$.subscribe(setup); } else { document.addEventListener("DOMContentLoaded", setup); }
})();

/* =========================================================
   Latent-Dynamics-Conditioned Flow Policy — page behavior
   - density / transport / schematic figures (inline SVG)
   - toy point-mass simulation (faithful to the stated dynamics)
   - scroll reveal, progress line, dot rail, keyboard navigation
   ========================================================= */
(function () {
"use strict";

var NS = "http://www.w3.org/2000/svg";
var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

var C = {
  env1: "#2f5fa5", env2: "#b9691c",
  ink: "#191b20", body: "#3a3d44", muted: "#6f727b", faint: "#9b9ea6",
  hair: "#e5e4de", hair2: "#d3d2ca",
  good: "#23784a", bad: "#b13a2e", gray: "#4a4d55"
};

/* ---------------- svg helpers ---------------- */
function S(tag, attrs, parent) {
  var el = document.createElementNS(NS, tag);
  if (attrs) for (var k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}
function txt(parent, x, y, str, attrs) {
  var t = S("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
  t.textContent = str;
  return t;
}
/* text with a trailing subscript, e.g. P with subscript z1 */
function subText(parent, x, y, main, sub, attrs) {
  var t = S("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
  var m = document.createElementNS(NS, "tspan"); m.textContent = main; t.appendChild(m);
  var s = document.createElementNS(NS, "tspan");
  s.setAttribute("dy", "3"); s.setAttribute("font-size", "9.5");
  s.textContent = sub; t.appendChild(s);
  return t;
}
function mkSvg(mount, w, h, label) {
  var el = typeof mount === "string" ? document.getElementById(mount) : mount;
  if (!el) return null;
  var svg = S("svg", {
    viewBox: "0 0 " + w + " " + h, class: "plot",
    role: "img", "aria-label": label || "", preserveAspectRatio: "xMidYMid meet"
  });
  el.appendChild(svg);
  return svg;
}

/* ---------------- gaussian curves ---------------- */
function mixY(x, comps) {
  var y = 0;
  for (var i = 0; i < comps.length; i++) {
    var c = comps[i], d = (x - c.mu) / c.sig;
    y += (c.w || 1) * Math.exp(-0.5 * d * d);
  }
  return y;
}
function maxMix(comps) {
  var m = 0;
  for (var x = 0; x <= 1.0001; x += 0.004) m = Math.max(m, mixY(x, comps));
  return m;
}
/* path for a mixture over x in [x0px .. x1px], baseline ybase, height scale */
function curveD(comps, x0, x1, ybase, scale, close) {
  var n = 130, d = "";
  for (var i = 0; i <= n; i++) {
    var u = i / n, px = x0 + u * (x1 - x0);
    var py = ybase - scale * mixY(u, comps);
    d += (i === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2);
  }
  if (close) d += "L" + x1 + " " + ybase + "L" + x0 + " " + ybase + "Z";
  return d;
}
function xpix(u, x0, x1) { return x0 + u * (x1 - x0); }

function axisLine(svg, x0, x1, y, label) {
  S("line", { x1: x0 - 4, y1: y, x2: x1 + 10, y2: y, class: "axis" }, svg);
  S("path", { d: "M" + (x1 + 10) + " " + (y - 3.4) + "L" + (x1 + 17) + " " + y +
       "L" + (x1 + 10) + " " + (y + 3.4) + "Z", class: "axis-arr" }, svg);
  if (label) txt(svg, x1 + 14, y + 14, label);
}

/* one density panel: curves = [{comps, stroke, fill, dash, width}] */
function densityPanel(mount, opts) {
  var w = opts.w || 340, h = opts.h || 150;
  var x0 = 18, x1 = w - 26, ybase = h - 22;
  var svg = mkSvg(mount, w, h, opts.label);
  if (!svg) return null;
  var peak = h - 46 - (opts.topPad || 0);
  var m = 0;
  opts.curves.forEach(function (c) { m = Math.max(m, maxMix(c.comps)); });
  var scale = peak / m;
  axisLine(svg, x0, x1, ybase, opts.axis || "a");
  opts.curves.forEach(function (c) {
    if (c.fill) S("path", { d: curveD(c.comps, x0, x1, ybase, scale, true),
      fill: c.fill, "fill-opacity": c.fillOp || 0.13, stroke: "none", class: "area" }, svg);
    var p = S("path", { d: curveD(c.comps, x0, x1, ybase, scale, false),
      fill: "none", stroke: c.stroke, "stroke-width": c.width || 2.2,
      class: "curve" }, svg);
    p.setAttribute("pathLength", "1");
    if (c.dash) p.setAttribute("stroke-dasharray", c.dash);
  });
  (opts.peakMarks || []).forEach(function (mk) {
    var px = xpix(mk.u, x0, x1);
    var py = ybase - scale * mixY(mk.u, mk.comps) - 8;
    txt(svg, px, py, mk.text, { "text-anchor": "middle", class: "late " + (mk.cls || ""),
      fill: mk.color || C.muted, "font-size": mk.size || 12.5 });
  });
  return { svg: svg, x0: x0, x1: x1, ybase: ybase, scale: scale };
}

/* ---------------- figure parameters shared across the page ---------------- */
var MU1 = 0.30, MU2 = 0.70, SIG = 0.075;
var g1 = [{ mu: MU1, sig: SIG, w: 1 }];
var g2 = [{ mu: MU2, sig: SIG, w: 1 }];
var gmix = [{ mu: MU1, sig: SIG, w: 0.5 }, { mu: MU2, sig: SIG, w: 0.5 }];
var gmix1 = [{ mu: MU1, sig: SIG, w: 0.5 }];
var gmix2 = [{ mu: MU2, sig: SIG, w: 0.5 }];

/* ---------------- section 02: merge figure ---------------- */
densityPanel("plot-env1", {
  w: 340, h: 150, label: "Unimodal expert action distribution under environment z1 (left peak).",
  curves: [{ comps: g1, stroke: C.env1, fill: C.env1 }],
  peakMarks: [{ u: MU1, comps: g1, text: "Mode 1", color: C.env1, cls: "mark" }]
});
densityPanel("plot-env2", {
  w: 340, h: 150, label: "Unimodal expert action distribution under environment z2 (right peak).",
  curves: [{ comps: g2, stroke: C.env2, fill: C.env2 }],
  peakMarks: [{ u: MU2, comps: g2, text: "Mode 2", color: C.env2, cls: "mark" }]
});
densityPanel("plot-marg", {
  w: 460, h: 168, label: "Bimodal marginal action distribution mixing both environments.",
  curves: [
    { comps: gmix1, stroke: "none", fill: C.env1, fillOp: 0.15, width: 0 },
    { comps: gmix2, stroke: "none", fill: C.env2, fillOp: 0.15, width: 0 },
    { comps: gmix, stroke: C.gray, width: 2.4 }
  ]
});

/* ---------------- section 03: training panel ---------------- */
densityPanel("plot-train", {
  w: 380, h: 158, label: "Training-time policy: bimodal distribution covering both expert modes.",
  curves: [
    { comps: gmix1, stroke: "none", fill: C.env1, fillOp: 0.15 },
    { comps: gmix2, stroke: "none", fill: C.env2, fillOp: 0.15 },
    { comps: gmix, stroke: C.gray, width: 2.4 }
  ],
  peakMarks: [
    { u: MU1, comps: gmix, text: "Mode 1", color: C.env1, cls: "mark" },
    { u: MU2, comps: gmix, text: "Mode 2", color: C.env2, cls: "mark" }
  ]
});

/* ---------------- section 03: deployment transport figure ---------------- */
(function deployFig() {
  var w = 460, h = 305;
  var svg = mkSvg("plot-deploy", w, h,
    "Flow transport at deployment: base noise at the bottom is transported upward to both modes; " +
    "arrivals on mode 2 are incompatible with the true environment z-star equals z1.");
  if (!svg) return;
  var x0 = 34, x1 = w - 30;
  var yTop = 150;                       /* baseline of the action distribution */
  var yBot = h - 26;                    /* baseline of the base density */
  var peak = 92, scaleT = peak / maxMix(gmix);
  var base = [{ mu: 0.5, sig: 0.11, w: 1 }];
  var scaleB = 48 / maxMix(base);

  /* lambda axis on the left */
  S("line", { x1: 16, y1: yBot, x2: 16, y2: yTop - 40, class: "axis" }, svg);
  S("path", { d: "M12.6 " + (yTop - 40) + " L16 " + (yTop - 48) + " L19.4 " + (yTop - 40) + "Z",
    class: "axis-arr" }, svg);
  txt(svg, 16, yBot + 14, "λ = 0", { "text-anchor": "middle", "font-size": 11 });
  txt(svg, 16, yTop - 54, "λ = 1", { "text-anchor": "middle", "font-size": 11 });

  /* action distribution (top) */
  axisLine(svg, x0, x1, yTop, "a");
  S("path", { d: curveD(gmix1, x0, x1, yTop, scaleT, true), fill: C.env1,
    "fill-opacity": 0.15, class: "area" }, svg);
  S("path", { d: curveD(gmix2, x0, x1, yTop, scaleT, true), fill: C.env2,
    "fill-opacity": 0.15, class: "area" }, svg);
  var pc = S("path", { d: curveD(gmix, x0, x1, yTop, scaleT, false), fill: "none",
    stroke: C.gray, "stroke-width": 2.4, class: "curve" }, svg);
  pc.setAttribute("pathLength", "1");

  /* base density (bottom) */
  var pb = S("path", { d: curveD(base, x0, x1, yBot, scaleB, false), fill: "none",
    stroke: C.faint, "stroke-width": 2, class: "curve" }, svg);
  pb.setAttribute("pathLength", "1");
  txt(svg, xpix(0.5, x0, x1), yBot + 16, "base noise", {
    "text-anchor": "middle", "font-size": 11.5 });

  /* transport paths: left half of the base flows to mode 1, right half to mode 2 */
  var flows = [
    { s: 0.40, d: MU1 - 0.05, m: 1 }, { s: 0.455, d: MU1, m: 1 }, { s: 0.49, d: MU1 + 0.05, m: 1 },
    { s: 0.51, d: MU2 - 0.05, m: 2 }, { s: 0.545, d: MU2, m: 2 }, { s: 0.60, d: MU2 + 0.05, m: 2 }
  ];
  flows.forEach(function (f) {
    var sx = xpix(f.s, x0, x1), dx = xpix(f.d, x0, x1);
    var sy = yBot - 2, dy = yTop + 3;
    var mid = (sy + dy) / 2;
    var p = S("path", {
      d: "M" + sx + " " + sy + " C" + sx + " " + mid + " " + dx + " " + mid + " " + dx + " " + dy,
      fill: "none", stroke: f.m === 1 ? C.env1 : C.env2,
      "stroke-width": 1.5, opacity: 0.6, class: "curve"
    }, svg);
    p.setAttribute("pathLength", "1");
    S("circle", { cx: dx, cy: dy - 3, r: 2.8, fill: f.m === 1 ? C.env1 : C.env2,
      class: "late" }, svg);
  });

  /* verdicts over the two peaks */
  txt(svg, xpix(MU1, x0, x1), yTop - scaleT * mixY(MU1, gmix) - 10, "✓",
    { "text-anchor": "middle", class: "late mark mark--good", "font-size": 16 });
  txt(svg, xpix(MU2, x0, x1), yTop - scaleT * mixY(MU2, gmix) - 10, "✗",
    { "text-anchor": "middle", class: "late mark mark--bad", "font-size": 16 });
})();

/* ---------------- section 06: conditional mini flow ---------------- */
(function miniFlow() {
  var w = 320, h = 205;
  var svg = mkSvg("minifig-flow", w, h,
    "Conditioned on z1, all transport paths from the base go to mode 1; mode 2 is not produced.");
  if (!svg) return;
  var x0 = 22, x1 = w - 26;
  var yTop = 96, yBot = h - 24;
  var scaleT = 62 / maxMix(g1);
  var base = [{ mu: 0.5, sig: 0.11, w: 1 }];
  var scaleB = 40 / maxMix(base);

  axisLine(svg, x0, x1, yTop, "a");
  S("path", { d: curveD(g1, x0, x1, yTop, scaleT, true), fill: C.env1,
    "fill-opacity": 0.15, class: "area" }, svg);
  var p1 = S("path", { d: curveD(g1, x0, x1, yTop, scaleT, false), fill: "none",
    stroke: C.env1, "stroke-width": 2.2, class: "curve" }, svg);
  p1.setAttribute("pathLength", "1");
  /* the suppressed other mode, ghosted */
  S("path", { d: curveD(g2, x0, x1, yTop, scaleT * 0.9, false), fill: "none",
    stroke: C.faint, "stroke-width": 1.4, "stroke-dasharray": "4 4", opacity: 0.6 }, svg);
  txt(svg, xpix(MU2, x0, x1), yTop - scaleT * 0.9 - 8, "suppressed", {
    "text-anchor": "middle", "font-size": 10.5, "font-style": "normal",
    "font-family": "inherit" });

  var pb = S("path", { d: curveD(base, x0, x1, yBot, scaleB, false), fill: "none",
    stroke: C.faint, "stroke-width": 2, class: "curve" }, svg);
  pb.setAttribute("pathLength", "1");

  [0.42, 0.475, 0.51, 0.55].forEach(function (s, i) {
    var dU = MU1 + (i - 1.5) * 0.035;
    var sx = xpix(s, x0, x1), dx = xpix(dU, x0, x1);
    var sy = yBot - 2, dy = yTop + 3, mid = (sy + dy) / 2;
    var p = S("path", {
      d: "M" + sx + " " + sy + " C" + sx + " " + mid + " " + dx + " " + mid + " " + dx + " " + dy,
      fill: "none", stroke: C.env1, "stroke-width": 1.5, opacity: 0.6, class: "curve"
    }, svg);
    p.setAttribute("pathLength", "1");
    S("circle", { cx: dx, cy: dy - 3, r: 2.6, fill: C.env1, class: "late" }, svg);
  });
  txt(svg, x0 + 4, 16, "given z = z₁", { "font-size": 12, fill: C.env1 });
})();

/* ---------------- section 07: behavior vs environment latent ---------------- */
(function behaviorFig() {
  var w = 320, h = 185;
  var svg = mkSvg("fig-behavior", w, h,
    "Two fans of expert actions from the same state, clustered into behavior labels go-left and go-right.");
  if (!svg) return;
  var sx = w / 2, sy = h - 30;
  S("circle", { cx: sx, cy: sy, r: 5, fill: C.ink }, svg);
  txt(svg, sx, sy + 18, "s", { "text-anchor": "middle" });

  function fan(dir) { /* dir -1 left, +1 right */
    var angles = [58, 44, 30];
    angles.forEach(function (adeg) {
      var a = adeg * Math.PI / 180;
      var dx = dir * Math.sin(a) * 92, dy = -Math.cos(a) * 92;
      var ex = sx + dx, ey = sy + dy;
      S("line", { x1: sx + dir * 8, y1: sy - 6, x2: ex, y2: ey, stroke: C.gray,
        "stroke-width": 1.7, class: "curve" }, svg).setAttribute("pathLength", "1");
      var ux = dx / 92, uy = dy / 92;
      S("path", { d: "M" + (ex + 4.5 * ux) + " " + (ey + 4.5 * uy) +
        "L" + (ex - 4 * uy - 1.8 * ux) + " " + (ey + 4 * ux - 1.8 * uy) +
        "L" + (ex + 4 * uy - 1.8 * ux) + " " + (ey - 4 * ux - 1.8 * uy) + "Z",
        fill: C.gray, class: "late" }, svg);
    });
    S("ellipse", { cx: sx + dir * 62, cy: sy - 64, rx: 42, ry: 30,
      fill: "none", stroke: C.bad, "stroke-width": 1.3, "stroke-dasharray": "5 4",
      transform: "rotate(" + dir * 32 + " " + (sx + dir * 62) + " " + (sy - 64) + ")",
      class: "late" }, svg);
    txt(svg, sx + dir * 96, sy - 104, dir < 0 ? "“z = 1?”" : "“z = 2?”",
      { "text-anchor": "middle", fill: C.bad, "font-size": 12, class: "late" });
  }
  fan(-1); fan(1);
  txt(svg, w / 2, 16, "clusters of action patterns", {
    "text-anchor": "middle", "font-size": 11.5 });
})();

(function transitionFig() {
  var w = 320, h = 185;
  var svg = mkSvg("fig-transition", w, h,
    "One state and one action; the next state differs by environment, so z must explain the world response.");
  if (!svg) return;
  var sx = 44, sy = h / 2 + 12;
  S("circle", { cx: sx, cy: sy, r: 5, fill: C.ink }, svg);
  txt(svg, sx, sy + 20, "s", { "text-anchor": "middle" });

  var ax = 150, ay = sy;
  S("line", { x1: sx + 9, y1: sy, x2: ax - 8, y2: ay, stroke: C.ink,
    "stroke-width": 2, class: "curve" }, svg).setAttribute("pathLength", "1");
  S("path", { d: "M" + (ax - 8) + " " + (ay - 4) + "L" + ax + " " + ay +
    "L" + (ax - 8) + " " + (ay + 4) + "Z", fill: C.ink }, svg);
  txt(svg, (sx + ax) / 2, sy - 10, "a", { "text-anchor": "middle" });

  function outcome(color, dy, sub, labdy) {
    var ex = 258, ey = ay + dy;
    S("line", { x1: ax + 4, y1: ay, x2: ex - 10, y2: ey, stroke: color,
      "stroke-width": 1.7, "stroke-dasharray": "5 4", class: "curve" }, svg)
      .setAttribute("pathLength", "1");
    var vx = ex - 10 - (ax + 4), vy = ey - ay, L = Math.hypot(vx, vy);
    var ux = vx / L, uy = vy / L;
    S("path", { d: "M" + (ex - 10 + 6 * ux) + " " + (ey + 6 * uy) +
      "L" + (ex - 10 - 4 * uy - 2 * ux) + " " + (ey + 4 * ux - 2 * uy) +
      "L" + (ex - 10 + 4 * uy - 2 * ux) + " " + (ey - 4 * ux - 2 * uy) + "Z",
      fill: color, class: "late" }, svg);
    S("circle", { cx: ex, cy: ey, r: 5, fill: color, class: "late" }, svg);
    txt(svg, ex + 10, ey + 4, "s′ under z" + sub, { fill: color, "font-size": 12 });
    subText(svg, (ax + ex) / 2 - 6, ay + dy / 2 + labdy, "P", "z" + sub,
      { fill: color, "font-size": 12 });
  }
  outcome(C.env1, -52, "₁", -8);
  outcome(C.env2, 52, "₂", 16);
  txt(svg, w / 2, 16, "same (s, a) — different response", {
    "text-anchor": "middle", "font-size": 11.5 });
})();

/* ---------------- section 09: identifiability panels ---------------- */
(function identFigs() {
  var same = [{ mu: 0.5, sig: 0.09, w: 1 }];
  var p0 = densityPanel("plot-ident0", {
    w: 300, h: 140, axis: "s′",
    label: "Two coinciding next-state densities: the environments are indistinguishable.",
    curves: [
      { comps: same, stroke: C.env1, width: 3 },
      { comps: same, stroke: C.env2, width: 2, dash: "4 7" }
    ]
  });
  if (p0) {
    var t0 = S("text", { x: xpix(0.5, p0.x0, p0.x1) - 24, y: 22, "font-size": 12 }, p0.svg);
    var prevSub = false;
    [["P", C.env1, false], ["z", C.env1, true], [" = ", C.muted, false],
     ["P", C.env2, false], ["z′", C.env2, true]].forEach(function (part) {
      var sp = document.createElementNS(NS, "tspan");
      sp.textContent = part[0];
      sp.setAttribute("fill", part[1]);
      sp.setAttribute("dy", part[2] === prevSub ? "0" : (part[2] ? "3" : "-3"));
      if (part[2]) sp.setAttribute("font-size", "9.5");
      prevSub = part[2];
      t0.appendChild(sp);
    });
  }
  var pP = densityPanel("plot-identpos", {
    w: 300, h: 140, axis: "s′",
    label: "Two separated next-state densities: the environments can be told apart.",
    curves: [
      { comps: [{ mu: 0.34, sig: 0.09, w: 1 }], stroke: C.env1, fill: C.env1, width: 2.2 },
      { comps: [{ mu: 0.66, sig: 0.09, w: 1 }], stroke: C.env2, fill: C.env2, width: 2.2, dash: "5 5" }
    ]
  });
  if (pP) {
    subText(pP.svg, xpix(0.34, pP.x0, pP.x1) - 10, 22, "P", "z", { fill: C.env1, "font-size": 12 });
    subText(pP.svg, xpix(0.66, pP.x0, pP.x1) - 10, 22, "P", "z′", { fill: C.env2, "font-size": 12 });
  }
})();

/* =========================================================
   Toy simulation — dynamics exactly as stated in section 08:
   x' = x + u + z (+ small noise), y' = y + v, z in {-c,+c}.
   Expert mode under environment z: u = -z (wind compensation)
   plus a proportional steering term toward the goal line.
   ========================================================= */
var toy = (function () {
  var mountStd = document.getElementById("toy-std");
  var mountOurs = document.getElementById("toy-ours");
  if (!mountStd || !mountOurs) return null;

  var W = 340, H = 268;
  var XS = 27, YS = 22;                    /* px per world unit */
  var CX = W / 2, Y0 = H - 18;             /* world (0,0) in px */
  function PX(wx) { return CX + wx * XS; }
  function PY(wy) { return Y0 - wy * YS; }

  var c = 0.4, K = 0.35, NOISE = 0.055, SIGN = 0.22;
  var STEPS = 11, VSTEP = (10 - 0.4) / STEPS;
  var GOALX = 0, GOALY = 10, GOALR = 0.62;
  var START = { x: 0, y: 0.4 };

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gaussRnd(rnd) {
    var u = 1 - rnd(), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function steer(x) {
    var s = -K * x;
    return Math.max(-0.5, Math.min(0.5, s));
  }

  /* one episode; policy: "std" samples a mode each step, "ours" infers z */
  function episode(kind, zTrue, rnd) {
    var pts = [{ x: START.x, y: START.y }];
    var modes = [], post = [];              /* post[t] = q(z=-c | h_t) after t transitions */
    var x = START.x, y = START.y;
    var logw = 0;                           /* log q(-c) - log q(+c) */
    post.push(0.5);
    for (var t = 0; t < STEPS; t++) {
      var mode;
      if (kind === "std") {
        mode = rnd() < 0.5 ? -c : +c;       /* compensation sampled from the marginal */
      } else {
        if (t === 0) mode = rnd() < 0.5 ? -c : +c;   /* prior: no information yet */
        else mode = (logw > 0) ? +c : -c;   /* compensate the inferred wind: u = -zhat */
      }
      var u = mode + steer(x);
      var eps = gaussRnd(rnd) * NOISE;
      var dx = u + zTrue + eps;
      x += dx; y += VSTEP;
      pts.push({ x: x, y: y });
      modes.push(mode);
      /* Bayes update on z from the observed transition: dx - u = z + noise */
      var r = dx - u;
      var lm = -0.5 * Math.pow((r + c) / SIGN, 2);   /* z = -c */
      var lp = -0.5 * Math.pow((r - c) / SIGN, 2);   /* z = +c */
      logw += lm - lp;
      post.push(1 / (1 + Math.exp(-logw)));
    }
    var hit = Math.abs(x - GOALX) < GOALR;
    return { pts: pts, modes: modes, post: post, hit: hit };
  }

  /* -------- field rendering -------- */
  function buildField(mount, label) {
    var svg = mkSvg(mount, W, H, label);
    svg.classList.add("field");
    svg.classList.remove("plot");
    var wind = S("g", null, svg);
    var deco = S("g", null, svg);
    var trails = S("g", null, svg);
    var agents = S("g", null, svg);
    var marks = S("g", null, svg);
    /* center guide */
    S("line", { x1: PX(0), y1: PY(0.2), x2: PX(0), y2: PY(GOALY), stroke: C.hair,
      "stroke-width": 1, "stroke-dasharray": "2 5" }, deco);
    /* goal */
    S("circle", { cx: PX(GOALX), cy: PY(GOALY), r: GOALR * XS, class: "goal-zone" }, deco);
    S("circle", { cx: PX(GOALX), cy: PY(GOALY), r: 6, class: "goal-outer" }, deco);
    S("circle", { cx: PX(GOALX), cy: PY(GOALY), r: 2.1, class: "goal-inner" }, deco);
    txt(deco, PX(GOALX) + 24, PY(GOALY) + 4, "goal", { class: "flab" });
    txt(deco, PX(START.x) + 24, PY(START.y) + 4, "start", { class: "flab" });
    txt(deco, 24, 22, "wind", { class: "flab" });
    return { svg: svg, wind: wind, trails: trails, agents: agents, marks: marks };
  }

  function drawWind(field, zTrue) {
    field.wind.innerHTML = "";
    field.svg.classList.toggle("wind-pos", zTrue > 0);
    var xa = 46, xb = W - 46;
    [78, 128, 178, 228].forEach(function (y) {
      S("line", { x1: xa, y1: y, x2: xb, y2: y, class: "windline" }, field.wind);
      var tip = zTrue > 0 ? xb : xa;
      var dirn = zTrue > 0 ? 1 : -1;
      S("path", { d: "M" + (tip + 7 * dirn) + " " + y + "L" + tip + " " + (y - 3.6) +
        "L" + tip + " " + (y + 3.6) + "Z", class: "windhead" }, field.wind);
    });
  }

  var fldStd = buildField(mountStd,
    "Animated rollout of the standard flow policy: it keeps sampling both compensation modes, so the point mass drifts with the wind and misses the goal.");
  var fldOurs = buildField(mountOurs,
    "Animated rollout of the conditioned policy: after the first observed transition the wind is identified and only the compatible mode is used, so the point mass reaches the goal.");

  var bars = {
    s1: document.getElementById("bar-std-1"), s2: document.getElementById("bar-std-2"),
    o1: document.getElementById("bar-ours-1"), o2: document.getElementById("bar-ours-2"),
    vs1: document.getElementById("val-std-1"), vs2: document.getElementById("val-std-2"),
    vo1: document.getElementById("val-ours-1"), vo2: document.getElementById("val-ours-2")
  };

  var state = { z: -c, seed: 11, eps: null, raf: 0, timer: 0, visible: false, running: false };

  function makeEpisodes() {
    var rnd = mulberry32(state.seed);
    state.eps = {
      std: [episode("std", state.z, rnd), episode("std", state.z, rnd), episode("std", state.z, rnd)],
      ours: episode("ours", state.z, rnd)
    };
  }

  function segColor(mode) { return mode < 0 ? "seg2" : "seg1"; } /* u=+c compensates z=-c (env 1, blue) */

  function clearRun() {
    cancelAnimationFrame(state.raf); clearTimeout(state.timer);
    fldStd.trails.innerHTML = ""; fldStd.agents.innerHTML = ""; fldStd.marks.innerHTML = "";
    fldOurs.trails.innerHTML = ""; fldOurs.agents.innerHTML = ""; fldOurs.marks.innerHTML = "";
    setBars(0.5, false);
  }

  function setBars(pNeg, inferred) {
    var a = Math.round(pNeg * 100), b = 100 - a;
    bars.o1.style.width = a + "%"; bars.o2.style.width = b + "%";
    bars.vo1.textContent = (pNeg).toFixed(2); bars.vo2.textContent = (1 - pNeg).toFixed(2);
    bars.s1.style.width = "50%"; bars.s2.style.width = "50%";
    bars.vs1.textContent = "—"; bars.vs2.textContent = "—";
    if (!inferred) { bars.vo1.textContent = "0.50"; bars.vo2.textContent = "0.50"; }
  }

  function pathLines(field, ep, dim) {
    var lines = [];
    for (var i = 0; i < STEPS; i++) {
      var l = S("line", {
        x1: PX(ep.pts[i].x), y1: PY(ep.pts[i].y),
        x2: PX(ep.pts[i].x), y2: PY(ep.pts[i].y),
        class: "trail " + segColor(ep.modes[i]) + (dim ? " trail--dim" : "")
      }, field.trails);
      lines.push(l);
    }
    return lines;
  }

  function endMark(field, ep, dim) {
    var last = ep.pts[STEPS];
    txt(field.marks, PX(last.x), PY(last.y) - 10, ep.hit ? "✓" : "✗", {
      "text-anchor": "middle", class: "endmark",
      fill: ep.hit ? C.good : C.bad, opacity: dim ? 0.45 : 1
    });
  }

  function renderStatic() {
    clearRun();
    state.eps.std.forEach(function (ep, i) {
      var lines = pathLines(fldStd, ep, i > 0);
      lines.forEach(function (l, t) {
        l.setAttribute("x2", PX(ep.pts[t + 1].x));
        l.setAttribute("y2", PY(ep.pts[t + 1].y));
      });
      endMark(fldStd, ep, i > 0);
    });
    var eo = state.eps.ours;
    var lo = pathLines(fldOurs, eo, false);
    lo.forEach(function (l, t) {
      l.setAttribute("x2", PX(eo.pts[t + 1].x));
      l.setAttribute("y2", PY(eo.pts[t + 1].y));
    });
    endMark(fldOurs, eo, false);
    setBars(eo.post[STEPS], true);
  }

  function animate() {
    clearRun();
    state.running = true;
    var STEP_MS = 360, SETTLE = 500;
    var linesStd = state.eps.std.map(function (ep, i) { return pathLines(fldStd, ep, i > 0); });
    var lineOurs = pathLines(fldOurs, state.eps.ours, false);
    var dotsStd = state.eps.std.map(function (ep, i) {
      return S("circle", { cx: PX(ep.pts[0].x), cy: PY(ep.pts[0].y), r: i === 0 ? 5 : 4,
        fill: C.gray, class: "agent", opacity: i === 0 ? 1 : 0.5 }, fldStd.agents);
    });
    var dotOurs = S("circle", { cx: PX(START.x), cy: PY(START.y), r: 5, fill: C.env1,
      class: "agent" }, fldOurs.agents);
    var start = performance.now();
    var postShown = 0;

    function frame(now) {
      var el = now - start;
      var step = Math.min(STEPS, Math.floor(el / STEP_MS));
      var frac = Math.min(1, (el - step * STEP_MS) / (STEP_MS * 0.82));
      function drawEp(ep, lines, dot) {
        for (var t = 0; t < STEPS; t++) {
          var l = lines[t];
          if (t < step) {
            l.setAttribute("x2", PX(ep.pts[t + 1].x)); l.setAttribute("y2", PY(ep.pts[t + 1].y));
          } else if (t === step && step < STEPS) {
            var ax = ep.pts[t].x + (ep.pts[t + 1].x - ep.pts[t].x) * frac;
            var ay = ep.pts[t].y + (ep.pts[t + 1].y - ep.pts[t].y) * frac;
            l.setAttribute("x2", PX(ax)); l.setAttribute("y2", PY(ay));
            if (dot) { dot.setAttribute("cx", PX(ax)); dot.setAttribute("cy", PY(ay)); }
          } else {
            l.setAttribute("x2", PX(ep.pts[t].x)); l.setAttribute("y2", PY(ep.pts[t].y));
          }
        }
        if (step >= STEPS && dot) {
          dot.setAttribute("cx", PX(ep.pts[STEPS].x)); dot.setAttribute("cy", PY(ep.pts[STEPS].y));
        }
      }
      state.eps.std.forEach(function (ep, i) { drawEp(ep, linesStd[i], dotsStd[i]); });
      drawEp(state.eps.ours, lineOurs, dotOurs);
      /* posterior bars update as transitions are observed */
      if (step > postShown) {
        postShown = step;
        setBars(state.eps.ours.post[Math.min(step, STEPS)], true);
      }
      if (el < STEPS * STEP_MS + SETTLE) {
        state.raf = requestAnimationFrame(frame);
      } else {
        state.eps.std.forEach(function (ep, i) { endMark(fldStd, ep, i > 0); });
        endMark(fldOurs, state.eps.ours, false);
        state.running = false;
        state.timer = setTimeout(function () {
          if (state.visible) { state.seed += 7; run(); }
        }, 2600);
      }
    }
    state.raf = requestAnimationFrame(frame);
  }

  function run() {
    makeEpisodes();
    drawWind(fldStd, state.z); drawWind(fldOurs, state.z);
    if (REDUCED) renderStatic(); else animate();
  }

  /* controls */
  var bNeg = document.getElementById("windNeg");
  var bPos = document.getElementById("windPos");
  function setWind(zSign) {
    state.z = zSign * c;
    bNeg.setAttribute("aria-pressed", String(zSign < 0));
    bPos.setAttribute("aria-pressed", String(zSign > 0));
    state.seed += 3;
    run();
  }
  bNeg.addEventListener("click", function () { setWind(-1); });
  bPos.addEventListener("click", function () { setWind(1); });
  document.getElementById("replay").addEventListener("click", function () {
    state.seed += 7; run();
  });

  /* run only while the section is on screen */
  var sec = document.getElementById("toy");
  new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      state.visible = e.isIntersecting;
      if (e.isIntersecting && !state.running) { state.seed += 1; run(); }
      if (!e.isIntersecting) { cancelAnimationFrame(state.raf); clearTimeout(state.timer); state.running = false; }
    });
  }, { threshold: 0.25 }).observe(sec);

  drawWind(fldStd, state.z); drawWind(fldOurs, state.z);
  makeEpisodes(); renderStatic();          /* meaningful first paint before any scroll */
  return state;
})();

/* =========================================================
   Scroll machinery: reveal, progress line, dot rail, keys
   ========================================================= */
var sections = Array.prototype.slice.call(document.querySelectorAll("section.sec"));

/* reveal */
var io = new IntersectionObserver(function (entries) {
  entries.forEach(function (e) {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  });
}, { threshold: 0.22 });
sections.forEach(function (s) { io.observe(s); });

/* dot rail */
var dotnav = document.getElementById("dotnav");
var dots = sections.map(function (s) {
  var a = document.createElement("a");
  a.href = "#" + s.id;
  a.title = s.getAttribute("data-nav") || s.id;
  a.setAttribute("aria-label", a.title);
  dotnav.appendChild(a);
  return a;
});

/* progress + active dot */
var pbar = document.getElementById("pbar");
var ticking = false;
function currentIndex() {
  var probe = window.scrollY + window.innerHeight * 0.42;
  var idx = 0;
  for (var i = 0; i < sections.length; i++) {
    if (sections[i].offsetTop <= probe) idx = i;
  }
  return idx;
}
function onScroll() {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(function () {
    ticking = false;
    var max = document.documentElement.scrollHeight - window.innerHeight;
    pbar.style.transform = "scaleX(" + (max > 0 ? window.scrollY / max : 0) + ")";
    var idx = currentIndex();
    dots.forEach(function (d, i) { d.classList.toggle("on", i === idx); });
  });
}
window.addEventListener("scroll", onScroll, { passive: true });
window.addEventListener("resize", onScroll);
onScroll();

/* keyboard: one section per keypress */
function goTo(idx) {
  idx = Math.max(0, Math.min(sections.length - 1, idx));
  sections[idx].scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
}
window.addEventListener("keydown", function (ev) {
  if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  var tag = (ev.target && ev.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  var idx = currentIndex();
  switch (ev.key) {
    case "ArrowDown": case "PageDown": case "j":
      ev.preventDefault(); goTo(idx + 1); break;
    case "ArrowUp": case "PageUp": case "k":
      ev.preventDefault(); goTo(idx - 1); break;
    case "Home": ev.preventDefault(); goTo(0); break;
    case "End": ev.preventDefault(); goTo(sections.length - 1); break;
  }
});

})();

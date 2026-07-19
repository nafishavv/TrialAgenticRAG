/* app.js — state machine + rendering for the two-view (landing/chat) UI.
   Model output is rendered through a tiny safe formatter (escape first, then a
   minimal markdown subset) — never raw innerHTML of model text. */
(function () {
  "use strict";

  var state = {
    sessionId: sessionStorage.getItem("session_id") || null,
    mode: sessionStorage.getItem("mode") || "enhanced",
    busy: false,
    modes: ["naive", "enhanced", "agentic"],
  };

  var $ = function (id) { return document.getElementById(id); };
  var thread = $("thread");

  /* ---------- safe rendering of model text ---------- */

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Minimal markdown subset: **bold**, bullet lists (* / -), numbered lists.
  function renderAnswer(text) {
    var lines = escapeHtml(text).split(/\r?\n/);
    var html = [], list = null; // "ul" | "ol" | null

    function closeList() {
      if (list) { html.push("</" + list + ">"); list = null; }
    }
    function inline(s) {
      return s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    }

    lines.forEach(function (line) {
      var t = line.trim();
      var mBullet = /^[*-]\s+(.*)$/.exec(t);
      var mNum = /^\d+\.\s+(.*)$/.exec(t);
      if (mBullet) {
        if (list !== "ul") { closeList(); html.push("<ul>"); list = "ul"; }
        html.push("<li>" + inline(mBullet[1]) + "</li>");
      } else if (mNum) {
        if (list !== "ol") { closeList(); html.push("<ol>"); list = "ol"; }
        html.push("<li>" + inline(mNum[1]) + "</li>");
      } else if (t === "") {
        closeList();
      } else {
        closeList();
        html.push("<p>" + inline(t) + "</p>");
      }
    });
    closeList();
    return html.join("");
  }

  /* ---------- message DOM ---------- */

  function addUserMessage(text) {
    var msg = document.createElement("div");
    msg.className = "msg user";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
    thread.appendChild(msg);
    scrollDown();
  }

  function addPending() {
    var msg = document.createElement("div");
    msg.className = "msg bot pending";
    msg.innerHTML =
      '<div class="bubble"><span class="dots"><span></span><span></span><span></span></span>' +
      '<span class="status">Memproses pertanyaan…</span></div>';
    thread.appendChild(msg);
    scrollDown();
    var status = msg.querySelector(".status");
    var t1 = setTimeout(function () { status.textContent = "Mencari dokumen terkait…"; }, 1500);
    var t2 = setTimeout(function () { status.textContent = "Menyusun jawaban…"; }, 6000);
    return {
      el: msg,
      done: function () { clearTimeout(t1); clearTimeout(t2); msg.remove(); },
    };
  }

  function formatLatency(seconds) {
    return seconds.toFixed(2).replace(".", ",") + " s";
  }

  function addBotMessage(resp) {
    var msg = document.createElement("div");
    msg.className = "msg bot";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = renderAnswer(resp.answer);
    msg.appendChild(bubble);

    // "Sumber (n) • 2,31 s" — the only statistic shown in the UI.
    var line = document.createElement("div");
    line.className = "sources-line";
    var n = resp.sources.length;
    if (n > 0) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.setAttribute("aria-expanded", "false");
      btn.textContent = "▸ Sumber (" + n + ") • " + formatLatency(resp.latency_seconds);
      var list = document.createElement("ul");
      list.className = "sources-list";
      list.hidden = true;
      resp.sources.forEach(function (s) {
        var li = document.createElement("li");
        li.textContent = s.label;
        list.appendChild(li);
      });
      btn.addEventListener("click", function () {
        var open = list.hidden;
        list.hidden = !open;
        btn.setAttribute("aria-expanded", String(open));
        btn.textContent = (open ? "▾" : "▸") + btn.textContent.slice(1);
      });
      line.appendChild(btn);
      line.appendChild(list);
    } else {
      line.textContent = formatLatency(resp.latency_seconds);
    }
    msg.appendChild(line);
    thread.appendChild(msg);
    scrollDown();
  }

  function addErrorMessage(err, retryText) {
    var msg = document.createElement("div");
    msg.className = "msg bot error";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = err && err.message
      ? err.message
      : "Maaf, terjadi kendala teknis. Silakan coba lagi.";
    msg.appendChild(bubble);
    if (retryText) {
      var retry = document.createElement("button");
      retry.className = "retry";
      retry.type = "button";
      retry.textContent = "Coba lagi";
      retry.addEventListener("click", function () {
        msg.remove();
        send(retryText, true);
      });
      msg.appendChild(retry);
    }
    thread.appendChild(msg);
    scrollDown();
  }

  function scrollDown() {
    thread.scrollTop = thread.scrollHeight;
  }

  /* ---------- core send flow ---------- */

  function enterChatView() {
    if (document.body.dataset.view !== "chat") {
      document.body.dataset.view = "chat";
      $("composer-input").focus();
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    $("composer-send").disabled = busy;
  }

  function send(text, isRetry) {
    text = (text || "").trim();
    if (!text || state.busy) return;
    enterChatView();
    if (!isRetry) addUserMessage(text);
    setBusy(true);
    var pending = addPending();

    API.postChat(text, state.sessionId, state.mode)
      .then(function (resp) {
        state.sessionId = resp.session_id;
        sessionStorage.setItem("session_id", resp.session_id);
        pending.done();
        addBotMessage(resp);
      })
      .catch(function (err) {
        pending.done();
        addErrorMessage(err, text);
      })
      .finally(function () {
        setBusy(false);
        $("composer-input").focus();
      });
  }

  function newConversation() {
    state.sessionId = null;
    sessionStorage.removeItem("session_id");
    thread.innerHTML = "";
    document.body.dataset.view = "landing";
    $("hero-input").value = "";
    window.scrollTo(0, 0);
  }

  /* ---------- bootstrap: meta-driven UI ---------- */

  function populateMeta(meta) {
    // Popular-question chips
    var chips = $("popular-chips");
    meta.popular_questions.forEach(function (q) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", function () { send(q); });
      chips.appendChild(b);
    });

    // Domain cards — navigation aids only; backend decides routing.
    var grid = $("domain-grid");
    meta.domains.forEach(function (d) {
      var card = document.createElement("article");
      card.className = "domain-card";
      var h = document.createElement("h3");
      h.textContent = d.title;
      var p = document.createElement("p");
      p.textContent = d.description;
      var ex = document.createElement("button");
      ex.type = "button";
      ex.textContent = "“" + d.example + "”";
      ex.addEventListener("click", function () { send(d.example); });
      card.appendChild(h); card.appendChild(p); card.appendChild(ex);
      grid.appendChild(card);
    });

    // Header "Layanan" dropdown
    var menu = $("nav-layanan-menu");
    meta.domains.forEach(function (d) {
      var li = document.createElement("li");
      li.setAttribute("role", "menuitem");
      li.innerHTML = "<strong></strong><span></span>";
      li.querySelector("strong").textContent = d.title;
      li.querySelector("span").textContent = d.description;
      li.addEventListener("click", function () { send(d.example); });
      menu.appendChild(li);
    });

    // Mode selector
    state.modes = meta.modes;
    var sel = $("mode-select");
    meta.modes.forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m.charAt(0).toUpperCase() + m.slice(1);
      sel.appendChild(opt);
    });
    if (meta.modes.indexOf(state.mode) === -1) state.mode = "enhanced";
    sel.value = state.mode;
  }

  /* ---------- wiring ---------- */

  document.addEventListener("DOMContentLoaded", function () {
    // Use the real hero photo when present (drop-in, no code change needed).
    var probe = new Image();
    probe.onload = function () {
      document.querySelector(".hero").classList.add("has-photo");
    };
    probe.src = "assets/img/hero-batang.webp";

    API.getMeta().then(populateMeta).catch(function () {
      // Meta failing shouldn't kill the page; chat can still work with defaults.
      var sel = $("mode-select");
      state.modes.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m.charAt(0).toUpperCase() + m.slice(1);
        sel.appendChild(opt);
      });
      sel.value = state.mode;
    });

    $("hero-form").addEventListener("submit", function (e) {
      e.preventDefault();
      send($("hero-input").value);
      $("hero-input").value = "";
    });

    $("composer-form").addEventListener("submit", function (e) {
      e.preventDefault();
      var input = $("composer-input");
      send(input.value);
      input.value = "";
      input.style.height = "auto";
    });

    // Enter = send, Shift+Enter = newline; auto-grow textarea.
    var composerInput = $("composer-input");
    composerInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        $("composer-form").requestSubmit();
      }
    });
    composerInput.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 132) + "px";
    });

    $("mode-select").addEventListener("change", function () {
      state.mode = this.value;
      sessionStorage.setItem("mode", state.mode);
    });

    $("btn-new-chat").addEventListener("click", newConversation);
    $("brand-home").addEventListener("click", function (e) {
      if (document.body.dataset.view === "chat") {
        e.preventDefault();
        newConversation();
      }
    });

    // "Layanan" dropdown open/close
    var navBtn = $("nav-layanan");
    var navMenu = $("nav-layanan-menu");
    navBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = navMenu.hidden;
      navMenu.hidden = !open;
      navBtn.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("click", function () {
      if (!navMenu.hidden) {
        navMenu.hidden = true;
        navBtn.setAttribute("aria-expanded", "false");
      }
    });
  });
})();

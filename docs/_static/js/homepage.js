/* Copy-to-clipboard for the landing page.
 *
 * sphinx-copybutton only decorates highlighted literal blocks. Everything on
 * the landing page is hand-written markup in _includes/homepage.html - the hero
 * install pill and the four <pre class="px-code"> panels - so it all gets its
 * own handler here.
 *
 * The panel buttons are injected rather than written into the markup, so a
 * button never appears unless it can actually work. This mirrors what
 * sphinx-copybutton does with the rest of the site.
 */
(function () {
  "use strict";

  var RESET_DELAY_MS = 1600;

  var ICON_COPY =
    '<svg class="px-copy-idle" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="9" y="9" width="13" height="13" rx="1"></rect>' +
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';

  var ICON_DONE =
    '<svg class="px-copy-done" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
    ' stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="20 6 9 17 4 12"></polyline></svg>';

  /* Write `text` to the clipboard, resolving either way. */
  function writeClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    // file:// and plain-http builds have no async clipboard API.
    return new Promise(function (resolve) {
      var helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      try {
        document.execCommand("copy");
      } catch (err) {
        /* nothing useful to do; the button just will not confirm */
      }
      document.body.removeChild(helper);
      resolve();
    });
  }

  /* Wire a button to copy whatever getText() returns.
   *
   * The button holds two icons and CSS shows one at a time off `is-copied`.
   * Both are aria-hidden, so the state change rides on aria-label instead.
   */
  function attachCopy(button, getText, label) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      writeClipboard(getText()).then(function () {
        button.classList.add("is-copied");
        button.setAttribute("aria-label", "Copied");
        window.setTimeout(function () {
          button.classList.remove("is-copied");
          button.setAttribute("aria-label", label);
        }, RESET_DELAY_MS);
      });
    });
  }

  /* Reduce a shell transcript to the commands alone.
   *
   * The CLI panel interleaves commands and their output. Copying it whole hands
   * back something unrunnable, so prompt lines are pulled out and the prompt
   * stripped - matching copybutton_prompt_text in conf.py, which does the same
   * for the rest of the site.
   */
  function codeText(pre) {
    var clone = pre.cloneNode(true);
    // the blinking block cursor is decoration, not code
    clone.querySelectorAll(".px-cursor").forEach(function (node) {
      node.remove();
    });

    var lines = clone.textContent.replace(/\s+$/, "").split("\n");
    var prompts = lines.filter(function (line) {
      return /^\s*\$\s+/.test(line);
    });

    if (prompts.length) {
      return prompts
        .map(function (line) {
          return line.replace(/^\s*\$\s+/, "");
        })
        .join("\n");
    }
    return lines.join("\n");
  }

  function init() {
    // 1. the hero install pill, whose button is in the markup already
    document.querySelectorAll(".px-install").forEach(function (pill) {
      var button = pill.querySelector(".px-copy");
      var command = pill.querySelector(".px-cmd");
      if (!button || !command) {
        return;
      }
      attachCopy(
        button,
        function () {
          return command.textContent.replace(/^\s*\$\s*/, "");
        },
        "Copy install command"
      );
    });

    // 2. the code panels, whose buttons are injected into the title bar
    document.querySelectorAll(".px-panel").forEach(function (panel) {
      var pre = panel.querySelector("pre.px-code");
      var head = panel.querySelector(".px-panel-head");
      if (!pre || !head || head.querySelector(".px-copy")) {
        return;
      }

      var label = "Copy code";
      var button = document.createElement("button");
      button.type = "button";
      button.className = "px-copy px-copy-panel";
      button.setAttribute("aria-label", label);
      button.title = label;
      button.innerHTML = ICON_COPY + ICON_DONE;

      head.appendChild(button);
      attachCopy(
        button,
        function () {
          return codeText(pre);
        },
        label
      );
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

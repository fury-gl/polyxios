/* Copy-to-clipboard for the landing page install command.
 *
 * sphinx-copybutton only decorates highlighted literal blocks; the hero pill in
 * _includes/homepage.html is hand-written markup, so it gets its own handler.
 */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".px-install").forEach(function (el) {
    var button = el.querySelector(".px-copy");
    var command = el.querySelector(".px-cmd");
    if (!button || !command) {
      return;
    }

    button.addEventListener("click", function (event) {
      event.preventDefault();
      var text = command.textContent.replace(/^\s*\$\s*/, "");

      var done = function () {
        button.textContent = "copied";
        button.classList.add("is-copied");
        window.setTimeout(function () {
          button.textContent = "copy";
          button.classList.remove("is-copied");
        }, 1600);
      };

      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done);
        return;
      }

      // file:// and plain-http builds have no async clipboard API.
      var helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      document.execCommand("copy");
      document.body.removeChild(helper);
      done();
    });
  });
});

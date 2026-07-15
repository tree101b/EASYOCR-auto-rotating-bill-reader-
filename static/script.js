const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropLabel = document.getElementById("drop-label");
const preview = document.getElementById("preview");
const form = document.getElementById("scan-form");
const goBtn = document.getElementById("go");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");
const summary = document.getElementById("summary");
const textEl = document.getElementById("text");
const download = document.getElementById("download");

function showPreview(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  dropLabel.textContent = file.name;
}

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => showPreview(fileInput.files[0]));

["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
  })
);
drop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    showPreview(file);
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files[0]) {
    setStatus("Please choose an image first.", true);
    return;
  }

  goBtn.disabled = true;
  setStatus("Scanning… the first scan loads the OCR model and can take a moment.");
  results.hidden = true;

  const data = new FormData();
  data.append("image", fileInput.files[0]);
  data.append("rotate", document.getElementById("rotate").value);

  try {
    const res = await fetch("/scan", { method: "POST", body: data });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || "Scan failed.");
    render(json);
    setStatus("Done.");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    goBtn.disabled = false;
  }
});

function render(r) {
  const f = r.fields || {};
  const chips = [
    chip("Lines found", r.line_count),
    chip("Avg. confidence", r.confidence + "%", r.confidence >= 60),
    f.invoice_no ? chip("Invoice #", f.invoice_no) : "",
    f.date ? chip("Date", f.date) : "",
    f.total ? chip("Total", f.total, true) : "",
  ].join("");
  summary.innerHTML = chips;
  textEl.textContent = r.text || "(no text detected)";
  download.href = "/outputs/" + r.output_file;
  results.hidden = false;
}

function chip(label, value, ok) {
  return `<div class="chip ${ok ? "ok" : ""}"><b>${label}</b><span>${value}</span></div>`;
}

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", !!isError);
}

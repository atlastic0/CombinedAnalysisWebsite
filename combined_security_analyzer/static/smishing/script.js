const messageInput = document.getElementById("messageText");
const analyzeButton = document.getElementById("analyzeButton");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceTrack = document.querySelector(".confidence-track");

messageInput.addEventListener("input", () => {
  document.getElementById("charCount").textContent =
    `${messageInput.value.length.toLocaleString()} / 10,000 characters`;
  resultBox.classList.add("hidden");
});

analyzeButton.addEventListener("click", async () => {
  const text = messageInput.value.trim();
  resultBox.classList.add("hidden");
  if (!text) {
    statusBox.textContent = "Paste a text message before analyzing it.";
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  statusBox.textContent = "Checking the message...";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text})
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error(`Unexpected server response (HTTP ${response.status}).`);
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");

    const confidence = Math.max(0, Math.min(100, Number(data.confidence) || 0));
    const isSmishing = data.classification === "Smishing";
    resultBox.classList.remove("safe", "smishing");
    resultBox.classList.add(isSmishing ? "smishing" : "safe");

    document.getElementById("classification").textContent = data.classification;
    document.getElementById("confidence").textContent = `${confidence.toFixed(2)}%`;
    confidenceBar.style.width = `${confidence}%`;
    confidenceTrack.setAttribute("aria-valuenow", confidence.toFixed(2));
    document.getElementById("explanation").textContent = data.explanation || "";
    document.getElementById("warning").textContent = data.warning || "";
    resultBox.classList.remove("hidden");
    statusBox.textContent = "Analysis complete.";
    resultBox.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze message";
  }
});

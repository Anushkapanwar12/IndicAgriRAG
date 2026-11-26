/**
 * IndicAgriRAG Frontend Application
 * Multilingual Agricultural RAG Interface
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const heroSection = document.getElementById("hero-section");
  const resultsSection = document.getElementById("results-section");
  const chatThread = document.getElementById("chat-thread");
  const queryInput = document.getElementById("query-input");
  const btnSend = document.getElementById("btn-send");
  const btnNewChat = document.getElementById("btn-new-chat");
  const btnMetrics = document.getElementById("btn-metrics");
  const btnCorpus = document.getElementById("btn-corpus");
  const btnThemeToggle = document.getElementById("btn-theme-toggle");
  const btnShuffle = document.getElementById("btn-shuffle");
  const btnCollapseExplore = document.getElementById("btn-collapse-explore");
  const exploreContainer = document.getElementById("explore-container");
  const questionCardsGrid = document.getElementById("question-cards-grid");

  const selectCorpusSource = document.getElementById("select-corpus-source");
  const selectAnswerLang = document.getElementById("select-answer-lang");
  const selectRetrievalMode = document.getElementById("select-retrieval-mode");

  const btnUploadPdfTrigger = document.getElementById("btn-upload-pdf-trigger");
  const btnExploreCorpusTrigger = document.getElementById("btn-explore-corpus-trigger");
  const btnAttach = document.getElementById("btn-attach");
  const pdfFileInput = document.getElementById("pdf-file-input");

  // Modals
  const modalMetrics = document.getElementById("modal-metrics");
  const btnCloseMetrics = document.getElementById("btn-close-metrics");
  const modalCorpus = document.getElementById("modal-corpus");
  const btnCloseCorpus = document.getElementById("btn-close-corpus");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const uploadStatus = document.getElementById("upload-status");
  const corpusSummary = document.getElementById("corpus-summary");

  // Mic Pill Elements
  const btnMicToggle = document.getElementById("btn-mic-toggle");
  const micLangLabel = document.getElementById("mic-lang-label");
  const micDropdownMenu = document.getElementById("mic-dropdown-menu");
  const micMenuItems = document.querySelectorAll(".mic-menu-item");

  // State
  let exampleBatches = [];
  let currentBatchIdx = 0;
  let speechRecognition = null;
  let isRecording = false;
  let activeMicLang = "hi-IN";

  // --- 1. Theme Management ---
  const initTheme = () => {
    const savedTheme = localStorage.getItem("indic_agri_rag_theme") || "theme-light";
    document.body.className = savedTheme;
    updateThemeIcon(savedTheme);
  };

  const updateThemeIcon = (theme) => {
    const sunIcon = document.querySelector(".icon-sun");
    const moonIcon = document.querySelector(".icon-moon");
    if (theme === "theme-dark") {
      sunIcon.style.display = "none";
      moonIcon.style.display = "block";
    } else {
      sunIcon.style.display = "block";
      moonIcon.style.display = "none";
    }
  };

  btnThemeToggle.addEventListener("click", () => {
    const isDark = document.body.classList.contains("theme-dark");
    const newTheme = isDark ? "theme-light" : "theme-dark";
    document.body.className = newTheme;
    localStorage.setItem("indic_agri_rag_theme", newTheme);
    updateThemeIcon(newTheme);
  });

  // --- 2. Load Example Questions & Shuffle ---
  const loadExamples = async () => {
    try {
      const res = await fetch("/api/examples");
      const data = await res.json();
      if (data.batches && data.batches.length > 0) {
        exampleBatches = data.batches;
        renderExampleBatch(0);
      }
    } catch (err) {
      console.error("Failed to fetch examples:", err);
    }
  };

  const renderExampleBatch = (idx) => {
    if (!exampleBatches[idx]) return;
    questionCardsGrid.innerHTML = "";
    exampleBatches[idx].forEach((item) => {
      const card = document.createElement("div");
      card.className = "q-card";
      card.innerHTML = `
        <span class="q-badge">${item.badge}</span>
        <div class="q-text">${item.question}</div>
      `;
      card.addEventListener("click", () => {
        queryInput.value = item.question;
        queryInput.focus();
        sendQuery(item.question);
      });
      questionCardsGrid.appendChild(card);
    });
  };

  btnShuffle.addEventListener("click", () => {
    if (exampleBatches.length > 1) {
      currentBatchIdx = (currentBatchIdx + 1) % exampleBatches.length;
      renderExampleBatch(currentBatchIdx);
    }
  });

  btnCollapseExplore.addEventListener("click", () => {
    exploreContainer.classList.toggle("collapsed");
  });

  // --- 3. Speech Recognition (Web Speech API) ---
  const initSpeech = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      speechRecognition = new SpeechRecognition();
      speechRecognition.continuous = false;
      speechRecognition.interimResults = false;
      speechRecognition.lang = activeMicLang;

      speechRecognition.onstart = () => {
        isRecording = true;
        btnMicToggle.classList.add("recording");
      };

      speechRecognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        queryInput.value = transcript;
        queryInput.focus();
      };

      speechRecognition.onerror = (e) => {
        console.warn("Speech error:", e.error);
        stopSpeech();
      };

      speechRecognition.onend = () => {
        stopSpeech();
      };
    }
  };

  const startSpeech = () => {
    if (!speechRecognition) {
      alert("Speech recognition is not supported in this browser. Please use Google Chrome or Edge.");
      return;
    }
    speechRecognition.lang = activeMicLang;
    try {
      speechRecognition.start();
    } catch (e) {
      stopSpeech();
    }
  };

  const stopSpeech = () => {
    isRecording = false;
    btnMicToggle.classList.remove("recording");
    if (speechRecognition) {
      try { speechRecognition.stop(); } catch (e) {}
    }
  };

  btnMicToggle.addEventListener("click", (e) => {
    // If clicking on chevron area, toggle menu; else toggle record
    if (e.target.closest("svg") && e.target.closest("svg").querySelector("polyline")) {
      micDropdownMenu.classList.toggle("show");
      return;
    }
    if (isRecording) {
      stopSpeech();
    } else {
      startSpeech();
    }
  });

  micMenuItems.forEach((item) => {
    item.addEventListener("click", () => {
      micMenuItems.forEach(i => i.classList.remove("active"));
      item.classList.add("active");
      activeMicLang = item.getAttribute("data-lang");
      micLangLabel.textContent = item.getAttribute("data-label");
      micDropdownMenu.classList.remove("show");
      if (isRecording) stopSpeech();
    });
  });

  document.addEventListener("click", (e) => {
    if (!btnMicToggle.contains(e.target) && !micDropdownMenu.contains(e.target)) {
      micDropdownMenu.classList.remove("show");
    }
  });

  // --- 4. Send Query & Answer Rendering ---
  const sendQuery = async (queryText) => {
    const q = (queryText || queryInput.value || "").trim();
    if (!q) return;

    // Switch view to Results
    heroSection.style.display = "none";
    resultsSection.style.display = "block";
    queryInput.value = "";

    // 1. Render User Message Card
    const userMsg = document.createElement("div");
    userMsg.className = "chat-message";
    userMsg.innerHTML = `<div class="user-query-card">${escapeHtml(q)}</div>`;
    chatThread.appendChild(userMsg);

    // 2. Render Loading Assistant Card
    const assistantMsg = document.createElement("div");
    assistantMsg.className = "chat-message";
    assistantMsg.innerHTML = `
      <div class="assistant-response-card">
        <div class="response-header">
          <div class="response-status-badge status-answerable">
            <span class="spinner-dot">●</span> Retrieving ICAR Evidence & Reasoning...
          </div>
          <div class="response-meta-pills">
            <span class="meta-pill">${selectRetrievalMode.value.toUpperCase()}</span>
          </div>
        </div>
        <div class="response-body" style="color:var(--text-muted);">
          Consulting 2,813 ICAR Kharif Agro-Advisory passages across English and Indic languages...
        </div>
      </div>
    `;
    chatThread.appendChild(assistantMsg);
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });

    // 3. Make API Call to /api/ask
    try {
      const payload = {
        query: q,
        top_k: 6,
        mode: selectRetrievalMode.value,
        target_lang: selectAnswerLang.value,
        source: selectCorpusSource.value
      };

      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      // Render actual grounded response
      renderAssistantResponse(assistantMsg, q, data);
    } catch (err) {
      assistantMsg.innerHTML = `
        <div class="assistant-response-card">
          <div class="response-header">
            <div class="response-status-badge status-unanswerable">Server Error</div>
          </div>
          <div class="response-body">Failed to process request: ${escapeHtml(err.message)}</div>
        </div>
      `;
    }
  };

  const renderAssistantResponse = (container, question, data) => {
    const isAns = data.is_answerable;
    const statusClass = isAns ? "status-answerable" : "status-unanswerable";
    const statusText = isAns ? "✓ <<ANSWERABLE>> Grounded in Evidence" : "⚠ <<UNANSWERABLE>> Insufficient Evidence";
    const modeLabel = (data.mode || "Hybrid").toUpperCase();
    const latency = data.latency_seconds ? `⚡ ${data.latency_seconds}s` : "";
    const generator = data.generator || "IndicAgriRAG";

    // Format citations to interactive pill tags
    let formattedAnswer = escapeHtml(data.answer);
    formattedAnswer = formattedAnswer.replace(/\[(ICAR[^,\]]+),\s*p\.\s*(\d+)\]/g, (match, src, page) => {
      return `<a href="javascript:void(0)" class="citation-tag" data-page="${page}" title="View source evidence">[${src}, p. ${page}]</a>`;
    });
    // Support bold formatting
    formattedAnswer = formattedAnswer.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    const passages = data.passages || [];

    container.innerHTML = `
      <div class="assistant-response-card">
        <div class="response-header">
          <div class="response-status-badge ${statusClass}">${statusText}</div>
          <div class="response-meta-pills">
            <span class="meta-pill">${modeLabel}</span>
            <span class="meta-pill">${generator}</span>
            ${latency ? `<span class="meta-pill">${latency}</span>` : ""}
          </div>
          <div class="response-actions">
            <button class="action-icon-btn btn-tts" title="Listen in audio">
              🔊 Listen
            </button>
            <button class="action-icon-btn btn-copy" title="Copy answer">
              📋 Copy
            </button>
          </div>
        </div>

        <div class="response-body">${formattedAnswer}</div>

        <!-- Evidence Accordion -->
        <div class="evidence-drawer">
          <div class="evidence-header">
            <span>📚 Grounded Passages (${passages.length} Retrieved)</span>
            <span class="evidence-toggle-text">▾ Expand Evidence</span>
          </div>
          <div class="evidence-list" style="display:none;">
            ${passages.map((p, i) => `
              <div class="evidence-card" id="passage-${p.page || i}">
                <div class="evidence-meta">
                  <span>#${i+1} · ${escapeHtml(p.source)} (Page ${p.page})</span>
                  <span class="evidence-score">Relevance: ${(p.score || 0).toFixed(4)}</span>
                </div>
                <div class="evidence-text">${escapeHtml(p.text)}</div>
              </div>
            `).join("")}
          </div>
        </div>
      </div>
    `;

    // Hook up Text-to-Speech
    const btnTts = container.querySelector(".btn-tts");
    btnTts.addEventListener("click", () => {
      speakText(data.answer);
    });

    // Hook up Copy
    const btnCopy = container.querySelector(".btn-copy");
    btnCopy.addEventListener("click", () => {
      navigator.clipboard.writeText(data.answer).then(() => {
        btnCopy.textContent = "✓ Copied";
        setTimeout(() => { btnCopy.textContent = "📋 Copy"; }, 2000);
      });
    });

    // Hook up Evidence Toggle
    const evHeader = container.querySelector(".evidence-header");
    const evList = container.querySelector(".evidence-list");
    const evToggleText = container.querySelector(".evidence-toggle-text");
    evHeader.addEventListener("click", () => {
      const isHidden = evList.style.display === "none";
      evList.style.display = isHidden ? "flex" : "none";
      evToggleText.textContent = isHidden ? "▴ Collapse Evidence" : "▾ Expand Evidence";
    });

    // Hook up Clickable Citation Tags
    container.querySelectorAll(".citation-tag").forEach((cTag) => {
      cTag.addEventListener("click", () => {
        evList.style.display = "flex";
        evToggleText.textContent = "▴ Collapse Evidence";
        const page = cTag.getAttribute("data-page");
        const targetCard = container.querySelector(`#passage-${page}`);
        if (targetCard) {
          targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
          targetCard.style.borderColor = "var(--accent-purple)";
          targetCard.style.boxShadow = "var(--shadow-card-hover)";
          setTimeout(() => {
            targetCard.style.borderColor = "var(--border-subtle)";
            targetCard.style.boxShadow = "none";
          }, 2500);
        }
      });
    });

    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  };

  // Text-to-Speech
  const speakText = (text) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/\[.*?\]/g, "").replace(/\*\*/g, "").trim();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    const isHindi = /[\u0900-\u097F]/.test(cleanText);
    utterance.lang = isHindi ? "hi-IN" : "en-IN";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  // Send Event Listeners
  btnSend.addEventListener("click", () => sendQuery());
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendQuery();
    }
  });

  // Reset / New Chat
  btnNewChat.addEventListener("click", () => {
    chatThread.innerHTML = "";
    resultsSection.style.display = "none";
    heroSection.style.display = "flex";
    queryInput.value = "";
    queryInput.focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // --- 5. Metrics & Evaluation Modal ---
  btnMetrics.addEventListener("click", async () => {
    modalMetrics.style.display = "flex";
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      renderMetricsModal(data);
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  });

  btnCloseMetrics.addEventListener("click", () => {
    modalMetrics.style.display = "none";
  });

  const renderMetricsModal = (data) => {
    const m = data.metrics || {};
    if (m.answerability_accuracy) {
      document.getElementById("kpi-acc").textContent = `${(m.answerability_accuracy * 100).toFixed(1)}%`;
      document.getElementById("kpi-f1").textContent = `${((m.answerable_f1 || 0) * 100).toFixed(1)}%`;
      document.getElementById("kpi-citation").textContent = `${((m.citation_coverage || 0) * 100).toFixed(1)}%`;
      document.getElementById("kpi-latency").textContent = `${(m.latency_seconds_median || 0).toFixed(1)}s`;
    }

    const plots = data.plots || [];
    const nav = document.getElementById("plot-tabs-nav");
    const activeImg = document.getElementById("active-plot-img");
    const activeDesc = document.getElementById("active-plot-desc");
    nav.innerHTML = "";

    if (plots.length > 0) {
      plots.forEach((p, i) => {
        const btn = document.createElement("button");
        btn.className = `plot-tab-btn ${i === 0 ? "active" : ""}`;
        btn.textContent = p.title;
        btn.addEventListener("click", () => {
          nav.querySelectorAll(".plot-tab-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeImg.src = p.url;
          activeDesc.textContent = p.description;
        });
        nav.appendChild(btn);
      });
      activeImg.src = plots[0].url;
      activeDesc.textContent = plots[0].description;
    }
  };

  // --- 6. Corpus & PDF Upload Modal ---
  const openCorpusModal = async () => {
    modalCorpus.style.display = "flex";
    try {
      const res = await fetch("/api/corpus");
      const data = await res.json();
      renderCorpusSummary(data);
    } catch (err) {
      console.error("Failed to load corpus stats:", err);
    }
  };

  btnCorpus.addEventListener("click", openCorpusModal);
  btnExploreCorpusTrigger.addEventListener("click", openCorpusModal);
  btnCloseCorpus.addEventListener("click", () => {
    modalCorpus.style.display = "none";
  });

  const renderCorpusSummary = (data) => {
    const sourcesHtml = Object.entries(data.sources || {}).map(([src, count]) => `
      <div>📄 <strong>${escapeHtml(src)}</strong>: ${count} passages</div>
    `).join("");

    corpusSummary.innerHTML = `
      <div style="margin-bottom:12px; font-weight:700; color:var(--text-main); font-size:1.05rem;">
        Corpus Index Overview
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:14px;">
        <div><strong>Total Indexed Passages:</strong> ${data.total_chunks || 2813}</div>
        <div><strong>Dense Embedding Model:</strong> ${data.embedding_model}</div>
        <div><strong>Lexical Sparse Engine:</strong> BM25Okapi</div>
        <div><strong>Generation Model:</strong> ${data.generation_model}</div>
      </div>
      <div style="border-top:1px solid var(--border-subtle); padding-top:10px;">
        <span style="font-size:0.8rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Indexed Documents:</span>
        <div style="margin-top:6px; display:flex; flex-direction:column; gap:4px;">${sourcesHtml}</div>
      </div>
    `;
  };

  // PDF Upload Handlers
  btnUploadPdfTrigger.addEventListener("click", () => pdfFileInput.click());
  btnAttach.addEventListener("click", () => pdfFileInput.click());
  uploadDropzone.addEventListener("click", () => pdfFileInput.click());

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });
  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });
  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handlePdfUpload(e.dataTransfer.files[0]);
    }
  });

  pdfFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handlePdfUpload(e.target.files[0]);
    }
  });

  const handlePdfUpload = async (file) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please upload a valid PDF document.");
      return;
    }
    uploadStatus.style.display = "block";
    uploadStatus.style.background = "var(--accent-purple-light)";
    uploadStatus.style.color = "var(--accent-purple)";
    uploadStatus.textContent = `Processing and chunking ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload-pdf", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        uploadStatus.style.background = "var(--accent-green-bg)";
        uploadStatus.style.color = "var(--accent-green)";
        uploadStatus.textContent = `✓ Indexed ${data.chunks_created} passages from ${file.name}.`;

        // Add to corpus dropdown
        const opt = document.createElement("option");
        opt.value = data.upload_id;
        opt.textContent = `Custom: ${file.name}`;
        selectCorpusSource.appendChild(opt);
        selectCorpusSource.value = data.upload_id;

        setTimeout(() => {
          modalCorpus.style.display = "none";
          queryInput.focus();
        }, 1800);
      } else {
        uploadStatus.style.background = "var(--accent-amber-bg)";
        uploadStatus.style.color = "var(--accent-amber)";
        uploadStatus.textContent = `Error: ${data.error || "Upload failed"}`;
      }
    } catch (err) {
      uploadStatus.style.background = "var(--accent-amber-bg)";
      uploadStatus.style.color = "var(--accent-amber)";
      uploadStatus.textContent = `Network error: ${err.message}`;
    }
  };

  // Close modals on backdrop click
  window.addEventListener("click", (e) => {
    if (e.target === modalMetrics) modalMetrics.style.display = "none";
    if (e.target === modalCorpus) modalCorpus.style.display = "none";
  });

  // Utilities
  const escapeHtml = (text) => {
    if (!text) return "";
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  };

  // Initialize
  initTheme();
  loadExamples();
  initSpeech();
});

/******************************************************
 * scripts.js (Production-Ready)
 ******************************************************/

/******************************************************
 * 1) Globals & Auto-Refresh
 ******************************************************/
let currentPage = 1; // for pagination
setInterval(refreshSnapshot, 60 * 1000);

document.addEventListener('DOMContentLoaded', function() {
  // Immediately refresh the snapshot
  refreshSnapshot();

  // Sidebar toggle
  const sidebarToggleBtn = document.getElementById('sidebarToggle');
  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener('click', () => {
      const sidebar = document.querySelector('.sidebar');
      if (sidebar) {
        sidebar.classList.toggle('collapsed');
      }
    });
  }

  // Robot dropdown logic
  const robotSelector = document.getElementById('robotSelector');
  if (robotSelector) {
    robotSelector.addEventListener('change', () => {
      currentPage = 1; // reset to page 1 whenever user changes robot
      refreshSnapshot();
    });
  }

  // Pagination buttons logic
  const prevPageBtn = document.getElementById('prevPageBtn');
  const nextPageBtn = document.getElementById('nextPageBtn');
  if (prevPageBtn && nextPageBtn) {
    prevPageBtn.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        refreshSnapshot();
      }
    });
    nextPageBtn.addEventListener('click', () => {
      currentPage++;
      refreshSnapshot();
      // After a short delay, if no sensor data is loaded, revert the page number
      setTimeout(() => {
        const tableBody = document.querySelector('#sensorTable tbody');
        if (tableBody && tableBody.children.length === 0) {
          console.warn("No sensor data on next page; reverting page number.");
          currentPage--;
        }
      }, 2000);
    });
  }

  // Initialize Chat + Voice logic
  initChat();
  initVoice();

  // If on a page with camera feeds
  if (document.getElementById('cameraFeed1') ||
      document.getElementById('cameraFeed2') ||
      document.getElementById('cameraFeed3') ||
      document.getElementById('cameraFeed4')) {
    console.log("Camera feeds page detected. Additional setup could go here.");
  }
});

/******************************************************
 * 2) Refresh Snapshot
 ******************************************************/
function refreshSnapshot() {
  const loadingIndicator = document.getElementById('loading');
  if (loadingIndicator) {
    loadingIndicator.style.display = 'block';
  }

  // 1) Get token from hidden field or localStorage
  const tokenElem = document.getElementById('userToken');
  let token = '';
  if (tokenElem) {
    token = tokenElem.value.trim();
  } 
  // Or, if you prefer localStorage:
  // let token = localStorage.getItem('access_token') || '';

  if (!token) {
    console.error("No token found. Aborting /snapshot request.");
    if (loadingIndicator) loadingIndicator.style.display = 'none';
    return;
  }

  // 2) Get selected robot & page for query params
  const robotValStr = document.getElementById('robotSelector')?.value || '';
  const robotVal = robotValStr === '' ? null : parseInt(robotValStr, 10);
  const pageParam = currentPage;

  // 3) Construct the snapshot URL
  const snapshotUrl = `/snapshot?robot_number=${robotVal}&page=${pageParam}`;

  // 4) Make the GET request with the Authorization header
  fetch(snapshotUrl, {
    method: 'GET',
    headers: {
      // If you do POST or JSON, add 'Content-Type': 'application/json'
      'Authorization': 'Bearer ' + token
    }
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    // 5) Update the various dashboard elements
    updateUseCaseDropdown(data);
    updateSystemHealth(data);
    updateSensorTable(data);
    updateLastUpdated(data);
    updateKpiMetrics(data);
    updateMilestone(data);

    // If the backend indicates an anomaly or issue, post it in the chat
    if (data.has_issue && data.issue_message) {
      appendChatMessage("System [Alert]", data.issue_message);
    }

    // If there's a predictive maintenance notification
    if (data.maintenance_needed && data.maintenance_message) {
      appendChatMessage("System [Maintenance]", data.maintenance_message);
    }
  })
  .catch(err => {
    console.error("Error fetching snapshot:", err);
    const systemHealthElem = document.getElementById('systemHealth');
    if (systemHealthElem) {
      systemHealthElem.textContent = "Error fetching data.";
      systemHealthElem.className =
        "system-health bg-secondary text-white p-3 mb-4 text-center text-uppercase font-weight-bold";
    }
  })
  .finally(() => {
    if (loadingIndicator) {
      loadingIndicator.style.display = 'none';
    }
  });
}

/******************************************************
 * 3) Dashboard Update Helpers
 ******************************************************/
function updateUseCaseDropdown(data) {
  if (data.use_case) {
    const useCaseSelect = document.getElementById("useCaseSelect");
    if (useCaseSelect) {
      useCaseSelect.value = data.use_case.toLowerCase();
    }
  }
}

function formatToLocalTime(utcString) {
  if (!utcString) return '';
  const dateObj = new Date(utcString + 'Z');
  return dateObj.toLocaleString();
}

function updateSystemHealth(data) {
  const systemHealthElem = document.getElementById('systemHealth');
  if (!systemHealthElem) return;

  // Default to "green" if no issue, "red" if there's an issue
  const defaultStatusColor = data.has_issue ? "red" : "green";
  const statusColorClass = data.status_color || defaultStatusColor;

  systemHealthElem.innerHTML = data.system_status || "Unknown System Status";
  systemHealthElem.className = "system-health " + statusColorClass + " p-3 mb-2 text-center text-uppercase font-weight-bold";

  // Remove any old appended child nodes
  while (systemHealthElem.childNodes.length > 1) {
    systemHealthElem.removeChild(systemHealthElem.lastChild);
  }

  const extraDiv = document.createElement("div");
  extraDiv.className = "mt-2";
  if (data.has_issue) {
    extraDiv.innerHTML = `<small><strong>Recommended Action:</strong> ${data.issue_message || 'Review sensor logs.'}</small>`;
  } else {
    extraDiv.innerHTML = `<small>All conditions normal, no user intervention required.</small>`;
  }
  systemHealthElem.appendChild(extraDiv);
}

function updateSensorTable(data) {
  const tableBody = document.querySelector('#sensorTable tbody');
  if (!tableBody) return;
  tableBody.innerHTML = "";

  (data.sensors || []).forEach(sensor => {
    const tr = document.createElement("tr");

    // Status
    const tdStatus = document.createElement("td");
    if (sensor.status && sensor.status.toLowerCase() === 'operational') {
      tdStatus.innerHTML = '<span class="green-circle" title="Operational Sensor"></span>';
    } else {
      tdStatus.innerHTML = '<span class="red-circle" title="Sensor Issue"></span>';
    }
    tr.appendChild(tdStatus);

    // Sensor name
    const tdName = document.createElement("td");
    tdName.textContent = sensor.sensor_name || "Unknown";
    tr.appendChild(tdName);

    // Sensor output
    const tdOutput = document.createElement("td");
    if (sensor.sensor_output == null) {
      tdOutput.textContent = "N/A";
    } else {
      const num = parseFloat(sensor.sensor_output);
      tdOutput.textContent = !isNaN(num) ? num.toFixed(2) : sensor.sensor_output;
    }
    tr.appendChild(tdOutput);

    // Robot number
    const tdRobot = document.createElement("td");
    tdRobot.textContent = sensor.robot_number != null ? sensor.robot_number : "";
    tr.appendChild(tdRobot);

    tableBody.appendChild(tr);
  });
}

function updateLastUpdated(data) {
  const lastUpdatedElem = document.getElementById("lastUpdated");
  if (!lastUpdatedElem) return;

  if (data.last_updated) {
    lastUpdatedElem.textContent = formatToLocalTime(data.last_updated);
  } else {
    lastUpdatedElem.textContent = new Date().toLocaleTimeString();
  }
}

function updateKpiMetrics(data) {
  const metrics = data.summary_metrics || {};
  const effElem = document.getElementById("avgEfficiencyVal");
  const savedElem = document.getElementById("totalEnergySavedVal");
  const anomElem = document.getElementById("totalAnomaliesVal");

  if (effElem) {
    if (typeof metrics.avg_energy_efficiency === "number") {
      effElem.textContent = metrics.avg_energy_efficiency.toFixed(1) + "%";
    } else {
      effElem.textContent = "N/A";
    }
  }
  if (savedElem) {
    if (typeof metrics.total_energy_saved === "number") {
      savedElem.textContent = metrics.total_energy_saved + " units";
    } else {
      savedElem.textContent = "N/A";
    }
  }
  if (anomElem) {
    if (typeof metrics.total_anomalies === "number") {
      anomElem.textContent = metrics.total_anomalies;
    } else {
      anomElem.textContent = "N/A";
    }
  }
}

function updateMilestone(data) {
  const milestoneElem = document.getElementById("milestoneBanner");
  if (!milestoneElem) return;

  if (data.milestone) {
    milestoneElem.style.display = "block";
    milestoneElem.innerHTML = '<strong>🎉 ' + data.milestone + ' 🎉</strong>';
  } else {
    milestoneElem.style.display = "block";
    milestoneElem.innerHTML = '<strong>No milestones yet</strong>';
  }
}

/******************************************************
 * 4) Voice Integration
 ******************************************************/
let recognition = null;
let isListening = false;

function initVoice() {
  const voiceBtn = document.getElementById('voiceBtn');
  const voiceIndicator = document.getElementById('voiceIndicator');
  if (!voiceBtn || !voiceIndicator) return;

  if (!('webkitSpeechRecognition' in window)) {
    console.warn("Speech recognition not supported in this browser.");
    voiceBtn.disabled = true;
    return;
  }

  recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = function() {
    console.log("Voice recognition started.");
    voiceIndicator.style.display = 'block';
  };

  recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript.trim();
    console.log("Heard:", transcript);

    appendChatMessage("You [Voice]", transcript);

    const tokenElem = document.getElementById('userToken');
    const token = tokenElem ? tokenElem.value.trim() : "";
    if (!token) {
      console.error("No token found for voice command. Aborting fetch.");
      return;
    }

    fetch("/voice_command", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify({ transcript })
    })
    .then(res => res.json())
    .then(data => {
      if (data.reply) {
        appendChatMessage("System", data.reply);
        speakText(data.reply);
      } else {
        console.warn("No reply in /voice_command response:", data);
      }
    })
    .catch(err => console.error("Voice command error:", err));
  };

  recognition.onerror = function(e) {
    console.error("Speech recognition error:", e);
  };

  recognition.onend = function() {
    console.log("Voice recognition ended.");
    isListening = false;
    voiceIndicator.style.display = 'none';
  };

  voiceBtn.addEventListener('click', toggleVoice);
}

function toggleVoice() {
  if (!recognition) return;
  if (!isListening) {
    recognition.start();
    isListening = true;
  } else {
    recognition.stop();
    isListening = false;
  }
}

/******************************************************
 * 5) Text-to-Speech Helper
 ******************************************************/
function speakText(text) {
  if (!window.speechSynthesis) {
    console.warn("Text-to-speech not supported in this browser.");
    return;
  }

  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.onstart = () => console.log("Speech started for:", text);
  utterance.onend = () => console.log("Speech ended for:", text);
  utterance.onerror = (err) => console.log("Speech error:", err);

  let voices = speechSynthesis.getVoices();
  if (!voices.length) {
    speechSynthesis.onvoiceschanged = () => {
      voices = speechSynthesis.getVoices();
      setPreferredVoice(utterance, voices);
      speechSynthesis.speak(utterance);
    };
  } else {
    setPreferredVoice(utterance, voices);
    speechSynthesis.speak(utterance);
  }
}

function setPreferredVoice(utterance, voices) {
  const preferred = voices.find(v =>
    (v.lang === 'en-US' || v.lang.startsWith('en')) &&
    (v.name.includes('Google') || v.name.includes('Microsoft'))
  );
  if (preferred) {
    utterance.voice = preferred;
  } else {
    const fallback = voices.find(v => v.lang.startsWith('en'));
    if (fallback) {
      utterance.voice = fallback;
    }
  }
  utterance.pitch = 1.0;
  utterance.rate = 1.0;
  utterance.lang = 'en-US';
}

/******************************************************
 * 6) Chat Integration
 ******************************************************/
function initChat() {
  const sendChatBtn = document.getElementById('sendChatBtn');
  const chatInput = document.getElementById('chatInput');
  const chatContainer = document.getElementById('chatContainer');

  if (!sendChatBtn || !chatInput || !chatContainer) {
    return;
  }

  loadChatHistory();

  sendChatBtn.addEventListener('click', () => {
    const userText = chatInput.value.trim();
    if (!userText) return;

    appendChatMessage('You', userText);

    fetch('/chat_endpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userText })
    })
    .then(res => res.json())
    .then(data => {
      if (data.reply) {
        appendChatMessage('System', data.reply);
        speakText(data.reply);
      } else {
        appendChatMessage('System', '[No response]');
      }
    })
    .catch(err => console.error("Chat endpoint error:", err));

    chatInput.value = '';
  });

  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendChatBtn.click();
    }
  });
}

function loadChatHistory() {
  const tokenElem = document.getElementById('userToken');
  if (!tokenElem) {
    console.warn("No userToken found; cannot load chat history.");
    return;
  }
  const token = tokenElem.value.trim();
  if (!token) {
    console.error("No token for chat history. Skipping fetch.");
    return;
  }

  fetch('/chat_history', {
    method: 'GET',
    headers: { 'Authorization': 'Bearer ' + token }
  })
  .then(res => {
    if (!res.ok) {
      throw new Error(`Failed to load chat history: ${res.status}`);
    }
    return res.json();
  })
  .then(history => {
    history.forEach(msg => {
      appendChatMessage(msg.speaker, msg.message);
    });
  })
  .catch(err => console.error("Error loading chat history:", err));
}

function appendChatMessage(speaker, text) {
  const chatContainer = document.getElementById('chatContainer');
  if (!chatContainer) return;

  // Create a container for this chat message
  const bubbleContainer = document.createElement('div');
  bubbleContainer.classList.add('chat-bubble-container');

  // Create the chat bubble element
  const bubble = document.createElement('div');
  let bubbleClass = '';

  // Determine alignment: system messages on the left, user/you messages on the right
  if (speaker.toLowerCase().includes('system')) {
    bubbleClass = 'chat-bubble-left';
  } else {
    bubbleClass = 'chat-bubble-right';
  }
  bubble.classList.add('chat-bubble', bubbleClass);

  // Format the speaker label (set to "System" or "You") with teal color for System
  let formattedSpeaker = speaker.toLowerCase().includes('system')
    ? '<span style="color:#00ffe0;">System</span>'
    : 'You';

  bubble.innerHTML = `<strong>${formattedSpeaker}:</strong> ${text}`;
  bubbleContainer.appendChild(bubble);
  chatContainer.appendChild(bubbleContainer);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/******************************************************
 * 7) Camera Feeds Integration
 ******************************************************/
function initCameraFeeds() {
  const cameraContainer = document.getElementById('cameraContainer');
  if (!cameraContainer) return;

  const tokenElem = document.getElementById('userToken');
  if (!tokenElem) {
    console.warn("No userToken found; cannot load camera feeds.");
    return;
  }
  const token = tokenElem.value.trim();
  if (!token) {
    console.error("No token for camera feeds. Skipping fetch.");
    return;
  }

  fetch('/api/camera_feeds', {
    method: 'GET',
    headers: { 'Authorization': 'Bearer ' + token }
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    cameraContainer.innerHTML = '';
    (data.feeds || []).forEach((feed, index) => {
      const feedDiv = document.createElement('div');
      feedDiv.classList.add('camera-feed');
      feedDiv.innerHTML = `
        <h4>${feed.name || 'Camera ' + (index + 1)}</h4>
        <iframe 
          src="${feed.url}"
          width="100%"
          height="300"
          frameborder="0"
          allowfullscreen
        ></iframe>
      `;
      cameraContainer.appendChild(feedDiv);
    });
  })
  .catch(err => {
    console.error("Error fetching camera feeds:", err);
  });
}

/******************************************************
 * 8) API Access
 ******************************************************/
function initAPIAccess() {
  const apiTokenDisplay = document.getElementById('apiTokenDisplay');
  const generateTokenBtn = document.getElementById('generateTokenBtn');
  if (!apiTokenDisplay || !generateTokenBtn) {
    return;
  }

  generateTokenBtn.addEventListener('click', () => {
    if (!confirm("Are you sure you want to generate (or rotate) your API token?")) {
      return;
    }

    fetch('/generate_api_token', {
      method: 'POST'
    })
    .then(res => {
      if (!res.ok) {
        throw new Error(`Failed to generate token: ${res.status}`);
      }
      window.location.reload();
    })
    .catch(err => console.error("Error generating token:", err));
  });
}

/******************************************************
 * 9) Alerts Toggle Event Listener
 ******************************************************/
const alertsToggle = document.getElementById('alertToggle');
if (alertsToggle) {
  alertsToggle.addEventListener('change', () => {
    const enabled = alertsToggle.checked;
    const tokenElem = document.getElementById('userToken');
    let token = "";
    if (tokenElem) {
      token = tokenElem.value.trim();
    }
    fetch('/update_alert_toggle', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify({ alerts_enabled: enabled })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Alert toggle updated:', data);
    })
    .catch(err => {
      console.error('Error updating alert toggle:', err);
    });
  });
}

/******************************************************
 * 10) Raw Data Page Integration
 ******************************************************/
function refreshRawData() {
  // Retrieve the current token from a hidden field (assumed to be present on raw_data.html)
  const tokenElem = document.getElementById('userToken');
  let token = '';
  if (tokenElem) {
    token = tokenElem.value.trim();
  }
  if (!token) {
    console.error("No token found. Aborting raw_data fetch.");
    return;
  }
  
  // Fetch aggregated raw data from the /raw_data endpoint
  fetch('/raw_data', {
    method: 'GET',
    headers: {
      'Authorization': 'Bearer ' + token
    }
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('HTTP error! status: ' + response.status);
    }
    return response.json();
  })
  .then(data => {
    // Update each container with the corresponding endpoint data
    updateRawDataContainer('raw_health', data.health);
    updateRawDataContainer('raw_metrics', data.metrics);
    updateRawDataContainer('raw_snapshot', data.snapshot);
    updateRawDataContainer('raw_rl_predict', data.rl_predict);
    updateRawDataContainer('raw_real_time_control', data.real_time_control);
  })
  .catch(error => {
    console.error("Error fetching raw data:", error);
  });
}

function updateRawDataContainer(containerId, endpointData) {
  const container = document.getElementById(containerId);
  if (container) {
    // Format the JSON with 2-space indentation for readability
    container.textContent = JSON.stringify(endpointData, null, 2);
  }
}

// When the DOM is fully loaded, check if we're on the raw_data page
// (by testing for a key container element) and then refresh raw data.
document.addEventListener('DOMContentLoaded', function() {
  if (document.getElementById('raw_health')) {
    refreshRawData();
    // Auto-refresh raw data every 60 seconds
    setInterval(refreshRawData, 60000);
  }
});


/******************************************************
 * 11) Final Loader
 ******************************************************/

document.addEventListener('DOMContentLoaded', () => {
  initAPIAccess();
});
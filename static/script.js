/**
 * EdgeGuard: Industrial Machine Health Monitoring & Anomaly Detection
 * High-Performance Real-Time Telemetry & TinyML Dashboard Engine
 */

// ==============================================================================
// GLOBAL STATE & CONSTANTS
// ==============================================================================
const POLLING_INTERVAL_MS = 200; // 5 Hz refresh rate
let pollingTimer = null;
let isFetching = false;

// DOM Element References Cache
const dom = {
    // Top Bar
    systemClock: document.getElementById("systemClock"),
    connectionDot: document.getElementById("connectionDot"),
    connectionStatusText: document.getElementById("connectionStatusText"),
    connectionDetailText: document.getElementById("connectionDetailText"),

    // Main Hero Status Card
    heroStatusCard: document.getElementById("heroStatusCard"),
    heroEmoji: document.getElementById("heroEmoji"),
    machineStatusText: document.getElementById("machineStatusText"),
    machineStatusDesc: document.getElementById("machineStatusDesc"),
    heroConfidence: document.getElementById("heroConfidence"),
    heroWindow: document.getElementById("heroWindow"),
    alarmTitle: document.getElementById("alarmTitle"),
    alarmMessage: document.getElementById("alarmMessage"),

    // Acceleration
    valAx: document.getElementById("valAx"),
    valAy: document.getElementById("valAy"),
    valAz: document.getElementById("valAz"),
    valAccelMag: document.getElementById("valAccelMag"),
    magAccelBar: document.getElementById("magAccelBar"),

    // Gyroscope
    valGx: document.getElementById("valGx"),
    valGy: document.getElementById("valGy"),
    valGz: document.getElementById("valGz"),
    valGyroMag: document.getElementById("valGyroMag"),
    magGyroBar: document.getElementById("magGyroBar"),

    // TinyML Prediction Card
    valMlClass: document.getElementById("valMlClass"),
    valMlConfidence: document.getElementById("valMlConfidence"),
    valMlWindow: document.getElementById("valMlWindow"),

    // Oscilloscope Legend
    legendAx: document.getElementById("legendAx"),
    legendAy: document.getElementById("legendAy"),
    legendAz: document.getElementById("legendAz"),

    // Canvas
    vibrationCanvas: document.getElementById("vibrationCanvas"),

    // History Table
    historyTableBody: document.getElementById("historyTableBody")
};

// ==============================================================================
// SYSTEM CLOCK UPDATE
// ==============================================================================
function updateSystemClock() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");
    if (dom.systemClock) {
        dom.systemClock.textContent = `${hours}:${minutes}:${seconds}`;
    }
}
setInterval(updateSystemClock, 1000);
updateSystemClock();

// ==============================================================================
// HIGH-PERFORMANCE VIBRATION OSCILLOSCOPE (CANVAS ENGINE)
// ==============================================================================
class VibrationOscilloscope {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas ? canvas.getContext("2d") : null;
        this.dpr = window.devicePixelRatio || 1;
        this.resize();

        window.addEventListener("resize", () => this.resize());
    }

    resize() {
        if (!this.canvas || !this.ctx) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height || 240;

        this.canvas.width = this.width * this.dpr;
        this.canvas.height = this.height * this.dpr;
        this.ctx.scale(this.dpr, this.dpr);
    }

    render(points) {
        if (!this.ctx || !this.width || !this.height) return;
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        // Clear canvas
        ctx.clearRect(0, 0, w, h);

        // Draw industrial oscilloscope grid
        ctx.strokeStyle = "rgba(30, 41, 59, 0.6)";
        ctx.lineWidth = 1;

        // Horizontal gridlines (5 divisions)
        const vSteps = 4;
        for (let i = 1; i <= vSteps; i++) {
            const y = (h / (vSteps + 1)) * i;
            ctx.beginPath();
            ctx.setLineDash([4, 4]);
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Center zero line
        ctx.strokeStyle = "rgba(51, 65, 85, 0.8)";
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // If no points yet, display waiting message
        if (!points || points.length < 2) {
            ctx.fillStyle = "#64748b";
            ctx.font = "13px 'JetBrains Mono', monospace";
            ctx.textAlign = "center";
            ctx.fillText("Waiting for acceleration stream...", w / 2, h / 2 - 10);
            return;
        }

        // Calculate dynamic Y min/max across all visible channels
        let minVal = -2000;
        let maxVal = 2000;

        for (const pt of points) {
            if (pt.ax !== undefined) {
                minVal = Math.min(minVal, pt.ax, pt.ay, pt.az);
                maxVal = Math.max(maxVal, pt.ax, pt.ay, pt.az);
            }
        }

        // Add 15% padding so waveforms don't clip
        const range = Math.max(maxVal - minVal, 1000);
        const yPadding = range * 0.15;
        const yMin = minVal - yPadding;
        const yMax = maxVal + yPadding;

        // Helper to convert data value to Y coordinate
        const getY = (val) => {
            const normalized = (val - yMin) / (yMax - yMin);
            return h - (normalized * h);
        };

        const totalPoints = points.length;
        const xStep = w / (totalPoints - 1);

        // Function to draw one continuous channel line
        const drawChannel = (key, strokeColor, shadowColor) => {
            ctx.save();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.shadowColor = shadowColor;
            ctx.shadowBlur = 6;
            ctx.beginPath();

            for (let i = 0; i < totalPoints; i++) {
                const x = i * xStep;
                const val = points[i][key];
                const y = getY(val);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            ctx.restore();
        };

        // Render AX (Cyan), AY (Purple), AZ (Emerald)
        drawChannel("ax", "#38bdf8", "rgba(56, 189, 248, 0.4)");
        drawChannel("ay", "#c084fc", "rgba(192, 132, 252, 0.4)");
        drawChannel("az", "#34d399", "rgba(52, 211, 153, 0.4)");
    }
}

const oscilloscope = new VibrationOscilloscope(dom.vibrationCanvas);

// ==============================================================================
// TELEMETRY UPDATE ENGINE
// ==============================================================================
async function fetchTelemetry() {
    if (isFetching) return;
    isFetching = true;

    try {
        const response = await fetch("/api/data", {
            cache: "no-store",
            headers: { "Accept": "application/json" }
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        renderDashboard(data);
    } catch (err) {
        handleOfflineState(err);
    } finally {
        isFetching = false;
    }
}

// ==============================================================================
// RENDER TELEMETRY DATA ON DASHBOARD
// ==============================================================================
function renderDashboard(data) {
    if (!data) return;

    const isConnected = data.connected === true;
    const sensor = data.sensor || {};
    const pred = data.prediction || {};
    const history = data.history || [];
    const points = data.vibration_points || [];

    // 1. Connection Status Badge
    if (dom.connectionDot) {
        dom.connectionDot.className = `status-indicator-dot ${isConnected ? "connected" : "disconnected"}`;
    }
    if (dom.connectionStatusText) {
        dom.connectionStatusText.textContent = isConnected ? "ESP32 Connected" : "ESP32 Disconnected";
    }
    if (dom.connectionDetailText) {
        dom.connectionDetailText.textContent = data.connection_detail || (isConnected ? "COM3 @ 115200 baud" : "Disconnected");
    }

    // 2. Main Hero Machine Status Card
    const status = (pred.status || "WAITING").toUpperCase();
    const heroCard = dom.heroStatusCard;

    if (heroCard) {
        heroCard.classList.remove("is-normal", "is-anomaly", "is-waiting");

        if (status === "NORMAL") {
            heroCard.classList.add("is-normal");
            dom.heroEmoji.textContent = "🟢";
            dom.machineStatusText.textContent = "NORMAL CONDITION";
            dom.machineStatusDesc.textContent = "Machine operating within nominal vibration thresholds. No structural or harmonic anomalies detected.";
            dom.alarmTitle.textContent = "Nominal Machine Condition";
            dom.alarmMessage.textContent = "Decision Tree classifier confirms normal mechanical behavior.";
        } else if (status === "ANOMALY") {
            heroCard.classList.add("is-anomaly");
            dom.heroEmoji.textContent = "🔴";
            dom.machineStatusText.textContent = "ANOMALY DETECTED ⚠️";
            dom.machineStatusDesc.textContent = "CRITICAL: Abnormal vibration signature detected! Inspect motor bearings, shaft alignment, or mechanical load.";
            dom.alarmTitle.textContent = "Anomaly Alert Active";
            dom.alarmMessage.textContent = "High harmonic deviation detected by TinyML model on current 10-sample window.";
        } else {
            heroCard.classList.add("is-waiting");
            dom.heroEmoji.textContent = "⏳";
            dom.machineStatusText.textContent = isConnected ? "WAITING FOR PREDICTION" : "ESP32 DISCONNECTED";
            dom.machineStatusDesc.textContent = isConnected
                ? `Collecting sensor window (${pred.window_progress || 0} / 10 samples)... Awaiting feature extraction.`
                : "COM3 serial link offline. Plug in ESP32 to start telemetry stream.";
            dom.alarmTitle.textContent = isConnected ? "Calibrating Feature Window" : "Serial Connection Inactive";
            dom.alarmMessage.textContent = isConnected
                ? "Extracting 24 statistical features once 10 samples are buffered."
                : "Waiting for ESP32 connection on COM3 @ 115200 baud.";
        }
    }

    // Hero metadata
    if (dom.heroConfidence) {
        dom.heroConfidence.textContent = (pred.confidence !== null && pred.confidence !== undefined)
            ? `${pred.confidence}%`
            : "--";
    }
    if (dom.heroWindow) {
        dom.heroWindow.textContent = `${pred.window_progress || 0} / ${pred.window_size || 10}`;
    }

    // 3. Live Acceleration Sensor Readouts
    if (sensor.ax !== null && sensor.ax !== undefined) {
        dom.valAx.textContent = sensor.ax;
        dom.valAy.textContent = sensor.ay;
        dom.valAz.textContent = sensor.az;
        dom.valAccelMag.textContent = sensor.accel_mag !== undefined ? sensor.accel_mag : "--";

        // Progress bar for magnitude (nominal max ~25000 LSB)
        const accelPercent = Math.min(Math.round(((sensor.accel_mag || 0) / 25000) * 100), 100);
        if (dom.magAccelBar) dom.magAccelBar.style.width = `${accelPercent}%`;
    } else {
        dom.valAx.textContent = "--";
        dom.valAy.textContent = "--";
        dom.valAz.textContent = "--";
        dom.valAccelMag.textContent = "--";
        if (dom.magAccelBar) dom.magAccelBar.style.width = "0%";
    }

    // 4. Live Gyroscope Sensor Readouts
    if (sensor.gx !== null && sensor.gx !== undefined) {
        dom.valGx.textContent = sensor.gx;
        dom.valGy.textContent = sensor.gy;
        dom.valGz.textContent = sensor.gz;
        dom.valGyroMag.textContent = sensor.gyro_mag !== undefined ? sensor.gyro_mag : "--";

        // Progress bar for magnitude (nominal max ~2000 LSB)
        const gyroPercent = Math.min(Math.round(((sensor.gyro_mag || 0) / 2000) * 100), 100);
        if (dom.magGyroBar) dom.magGyroBar.style.width = `${gyroPercent}%`;
    } else {
        dom.valGx.textContent = "--";
        dom.valGy.textContent = "--";
        dom.valGz.textContent = "--";
        dom.valGyroMag.textContent = "--";
        if (dom.magGyroBar) dom.magGyroBar.style.width = "0%";
    }

    // 5. TinyML Prediction Card
    if (dom.valMlClass) {
        dom.valMlClass.className = "ml-class-display";
        if (status === "NORMAL") {
            dom.valMlClass.textContent = "NORMAL";
            dom.valMlClass.classList.add("normal");
        } else if (status === "ANOMALY") {
            dom.valMlClass.textContent = "ANOMALY";
            dom.valMlClass.classList.add("anomaly");
        } else {
            dom.valMlClass.textContent = isConnected ? "WAITING" : "OFFLINE";
        }
    }

    if (dom.valMlConfidence) {
        dom.valMlConfidence.textContent = (pred.confidence !== null && pred.confidence !== undefined)
            ? `${pred.confidence}%`
            : "--";
    }
    if (dom.valMlWindow) {
        dom.valMlWindow.textContent = `${pred.window_progress || 0} / 10 Samples`;
    }

    // 6. Oscilloscope Legend Values
    if (dom.legendAx) dom.legendAx.textContent = sensor.ax !== null && sensor.ax !== undefined ? sensor.ax : "--";
    if (dom.legendAy) dom.legendAy.textContent = sensor.ay !== null && sensor.ay !== undefined ? sensor.ay : "--";
    if (dom.legendAz) dom.legendAz.textContent = sensor.az !== null && sensor.az !== undefined ? sensor.az : "--";

    // 7. Render Oscilloscope Waveform
    oscilloscope.render(points);

    // 8. Recent Predictions History Table
    renderHistoryTable(history);
}

// ==============================================================================
// RENDER PREDICTIONS TABLE (RESPONSIVE TABLE CONTAINER)
// ==============================================================================
function renderHistoryTable(historyList) {
    if (!dom.historyTableBody) return;

    if (!historyList || historyList.length === 0) {
        dom.historyTableBody.innerHTML = `
            <tr>
                <td colspan="4" class="table-empty-cell">Waiting for ESP32 telemetry stream...</td>
            </tr>
        `;
        return;
    }

    let rowsHtml = "";
    historyList.slice(0, 10).forEach((item, index) => {
        const isAnomaly = (item.status || "").toUpperCase() === "ANOMALY";
        const pillClass = isAnomaly ? "anomaly" : "normal";
        const statusText = isAnomaly ? "ANOMALY ⚠️" : "NORMAL 🟢";

        rowsHtml += `
            <tr>
                <td>#${index + 1}</td>
                <td>${item.timestamp || "--"}</td>
                <td>
                    <span class="status-pill ${pillClass}">
                        ${statusText}
                    </span>
                </td>
                <td>${item.confidence || "--"}</td>
            </tr>
        `;
    });

    dom.historyTableBody.innerHTML = rowsHtml;
}

// ==============================================================================
// OFFLINE & ERROR HANDLER
// ==============================================================================
function handleOfflineState(error) {
    if (dom.connectionDot) {
        dom.connectionDot.className = "status-indicator-dot disconnected";
    }
    if (dom.connectionStatusText) {
        dom.connectionStatusText.textContent = "Server Offline";
    }
    if (dom.connectionDetailText) {
        dom.connectionDetailText.textContent = "Flask server unreachable on port 5000";
    }
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
    // Initial fetch
    fetchTelemetry();
    // Start continuous polling loop
    pollingTimer = setInterval(fetchTelemetry, POLLING_INTERVAL_MS);
});


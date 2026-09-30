const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const bars = document.getElementById("bars");
const status = document.getElementById("status");

let drawing = false;
let lastX = 0;
let lastY = 0;
let timer = null;

ctx.fillStyle = "#000";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 18;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#fff";

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const point = event.touches ? event.touches[0] : event;
    return {
        x: (point.clientX - rect.left) * scaleX,
        y: (point.clientY - rect.top) * scaleY
    };
}

function startDrawing(event) {
    event.preventDefault();
    drawing = true;
    const p = getPosition(event);
    lastX = p.x;
    lastY = p.y;

    ctx.beginPath();
    ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();

    schedulePrediction();
}

function draw(event) {
    if (!drawing) return;
    event.preventDefault();

    const p = getPosition(event);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();

    lastX = p.x;
    lastY = p.y;
    schedulePrediction();
}

function stopDrawing() {
    drawing = false;
}

function schedulePrediction() {
    clearTimeout(timer);
    timer = setTimeout(predict, 120);
}

async function predict() {
    canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append("image", blob, "digit.png");

        status.textContent = "Predicting...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed");
            }

            prediction.textContent = data.digit;
            confidence.textContent = `${data.confidence}% confidence`;
            renderBars(data.probabilities);
            status.textContent = "Prediction updated";
        } catch (error) {
            status.textContent = error.message;
        }
    }, "image/png");
}

function renderBars(probabilities) {
    bars.innerHTML = "";

    probabilities.forEach((value, digit) => {
        const row = document.createElement("div");
        row.className = "bar-row";

        row.innerHTML = `
            <span>${digit}</span>
            <div class="bar">
                <div class="fill" style="width: ${value}%"></div>
            </div>
            <span>${value.toFixed(1)}%</span>
        `;

        bars.appendChild(row);
    });
}

function clearCanvas() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    prediction.textContent = "—";
    confidence.textContent = "Draw a digit to begin";
    bars.innerHTML = "";
    status.textContent = "Ready";
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

clearBtn.addEventListener("click", clearCanvas);

import * as inf from './inference.js';

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const status = $('status');
const placeholder = document.querySelector('.placeholder');
const startButton = $('btn-camera');
const confidence = $('conf-slider');
let stream = null;
let running = false;
let busy = false;
let raf = 0;
const detectionColor = '#ED145B';

function setStatus(message, kind = '') { status.textContent = message; status.dataset.kind = kind; }
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function draw(detections) {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!video.videoWidth || !video.videoHeight) return;
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const xOffset = (rect.width - video.videoWidth * scale) / 2;
  const yOffset = (rect.height - video.videoHeight * scale) / 2;
  for (const d of detections) {
    const x = xOffset + d.x1 * scale, y = yOffset + d.y1 * scale;
    const w = (d.x2 - d.x1) * scale, h = (d.y2 - d.y1) * scale;
    if (w <= 0 || h <= 0) continue;
    const color = detectionColor;
    const label = `${inf.CLASS_NAMES[d.classId]} ${(d.conf * 100).toFixed(0)}%`;
    ctx.lineWidth = 2.5; ctx.strokeStyle = color; ctx.strokeRect(x, y, w, h);
    ctx.font = '600 13px system-ui';
    const labelWidth = ctx.measureText(label).width + 12;
    const labelY = Math.max(0, y - 25);
    ctx.fillStyle = color; ctx.fillRect(x, labelY, labelWidth, 23);
    ctx.fillStyle = '#101820'; ctx.fillText(label, x + 6, labelY + 16);
  }
}

async function frame() {
  if (!running) return;
  if (!busy && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    busy = true;
    try {
      const now = performance.now();
      const input = inf.preprocess(video);
      const session = inf.getSession();
      const outputs = await session.run({ [session.inputNames[0]]: input });
      const detections = inf.postprocess(outputs[session.outputNames[0]]);
      draw(detections);
      $('detection-count').textContent = `${detections.length} detecção${detections.length === 1 ? '' : 'ões'}`;
      const elapsed = Math.round(performance.now() - now);
      $('latency').textContent = `${elapsed} ms`;
      setStatus('Câmera ativa · inferência no dispositivo', 'live');
    } catch (error) {
      console.error(error);
      setStatus(`Falha na inferência: ${error.message}`, 'error');
    } finally { busy = false; }
  }
  raf = requestAnimationFrame(frame);
}

async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('A câmera exige HTTPS (ou localhost) e um navegador compatível.');
  }
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error('Não foi possível iniciar o vídeo da câmera.'));
  });
  await video.play();
  video.hidden = false;
  placeholder.hidden = true;
  resizeCanvas();
  running = true;
  startButton.textContent = 'Parar detecção';
  startButton.dataset.running = 'true';
  raf = requestAnimationFrame(frame);
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(raf);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  video.hidden = true;
  placeholder.hidden = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startButton.textContent = 'Iniciar detecção';
  startButton.dataset.running = 'false';
  $('detection-count').textContent = '0 detecções';
  setStatus(inf.getSession() ? 'Pronto para iniciar' : 'Modelo não carregado', '');
}

startButton.addEventListener('click', async () => {
  if (running) { stopCamera(); return; }
  startButton.disabled = true;
  try { await startCamera(); }
  catch (error) {
    console.error(error);
    setStatus(error.name === 'NotAllowedError' ? 'Permissão da câmera negada. Autorize a câmera nas configurações do Safari.' : error.message, 'error');
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
  } finally { startButton.disabled = false; }
});

confidence.addEventListener('input', () => {
  const value = Number(confidence.value) / 100;
  inf.setConfidence(value);
  $('conf-value').textContent = value.toFixed(2);
});
window.addEventListener('resize', resizeCanvas);
window.addEventListener('pagehide', stopCamera);

async function init() {
  resizeCanvas();
  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('/sw.js').catch((error) => console.warn('Service Worker indisponível:', error));
  }
  startButton.disabled = true;
  setStatus('Carregando modelo treinado (640 × 640)…');
  try {
    await inf.loadModel();
    startButton.disabled = false;
    $('model-state').textContent = 'Modelo carregado · 6 classes';
    setStatus('Modelo carregado. Autorize a câmera para começar.', 'ready');
  } catch (error) {
    console.error(error);
    $('model-state').textContent = 'Modelo ausente ou incompatível';
    setStatus(`Não foi possível carregar o modelo: ${error.message} Copie o ONNX treinado para public/models/auditoria_640.onnx.`, 'error');
  }
}
init();

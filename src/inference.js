import * as ort from 'onnxruntime-web';

export const MODEL_SIZE = 640;
export const CLASS_NAMES = ['animal', 'traffic_lights', 'waste_container', 'pothole', 'cracks', 'open_manhole'];
export const EXPECTED_OUTPUT = [1, 10, 8400];

let session = null;
let confidenceThreshold = 0.35;
let iouThreshold = 0.45;
let transform = null;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

export async function loadModel(path = '/models/auditoria_640.onnx') {
  const next = await ort.InferenceSession.create(path, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  try {
    if (next.inputNames.length !== 1 || next.outputNames.length !== 1) throw new Error('Esperada uma entrada e uma saída ONNX.');
    const probe = new ort.Tensor('float32', new Float32Array(3 * MODEL_SIZE * MODEL_SIZE), [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const result = await next.run({ [next.inputNames[0]]: probe });
    const output = result[next.outputNames[0]];
    if (!output || output.dims.join(',') !== EXPECTED_OUTPUT.join(',')) {
      throw new Error(`Saída ONNX incompatível. Esperado [${EXPECTED_OUTPUT}], recebido [${output?.dims ?? 'indisponível'}].`);
    }
  } catch (error) {
    await next.release();
    throw error;
  }
  session = next;
  return next;
}

export function getSession() { return session; }
export function setConfidence(value) { confidenceThreshold = Number(value); }

export function preprocess(source) {
  const srcW = source.videoWidth || source.naturalWidth || source.width;
  const srcH = source.videoHeight || source.naturalHeight || source.height;
  if (!srcW || !srcH) throw new Error('A imagem de entrada ainda não está pronta.');
  const geometry = getLetterboxGeometry(srcW, srcH);
  const canvas = document.createElement('canvas');
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  // Ultralytics LetterBox uses a constant 114 gray border.
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(source, 0, 0, srcW, srcH, geometry.padX, geometry.padY, geometry.width, geometry.height);
  const pixels = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const data = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    data[i] = pixels[p] / 255;
    data[plane + i] = pixels[p + 1] / 255;
    data[plane * 2 + i] = pixels[p + 2] / 255;
  }
  transform = { srcW, srcH, ...geometry };
  return new ort.Tensor('float32', data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

export function getLetterboxGeometry(srcW, srcH, size = MODEL_SIZE) {
  const scale = Math.min(size / srcW, size / srcH);
  const width = roundLikePython(srcW * scale);
  const height = roundLikePython(srcH * scale);
  const padW = size - width;
  const padH = size - height;
  return {
    scale, width, height,
    padX: roundLikePython(padW / 2 - 0.1),
    padY: roundLikePython(padH / 2 - 0.1),
    padRight: roundLikePython(padW / 2 + 0.1),
    padBottom: roundLikePython(padH / 2 + 0.1),
  };
}

function roundLikePython(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function mapBoxToSource(box) {
  if (!transform) return box;
  return mapBox(box, transform);
}

export function mapBox(box, state) {
  const { srcW, srcH, scale, padX, padY } = state;
  return {
    x1: clamp((box.x1 - padX) / scale, 0, srcW),
    y1: clamp((box.y1 - padY) / scale, 0, srcH),
    x2: clamp((box.x2 - padX) / scale, 0, srcW),
    y2: clamp((box.y2 - padY) / scale, 0, srcH),
  };
}

export function postprocess(output, confidence = confidenceThreshold, iou = iouThreshold) {
  const [batch, channels, count] = output.dims;
  if (batch !== EXPECTED_OUTPUT[0] || channels !== EXPECTED_OUTPUT[1] || count !== EXPECTED_OUTPUT[2]) {
    throw new Error(`Saída ONNX incompatível: esperado [${EXPECTED_OUTPUT}], recebido [${output.dims}]`);
  }
  const values = output.data;
  const candidates = [];
  for (let i = 0; i < count; i++) {
    const cx = values[i], cy = values[count + i], w = values[2 * count + i], h = values[3 * count + i];
    if (![cx, cy, w, h].every(Number.isFinite) || w <= 0 || h <= 0) continue;
    let score = -Infinity, classId = -1;
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const candidate = values[(4 + c) * count + i];
      if (candidate > score) { score = candidate; classId = c; }
    }
    if (!Number.isFinite(score) || score < confidence) continue;
    candidates.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, classId, conf: score });
  }
  return classAwareNms(candidates, iou).map((box) => ({ ...box, ...mapBoxToSource(box) }));
}

export function classAwareNms(boxes, threshold) {
  const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
  const kept = [];
  while (sorted.length) {
    const best = sorted.shift();
    kept.push(best);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i].classId === best.classId && computeIoU(sorted[i], best) > threshold) sorted.splice(i, 1);
    }
  }
  return kept;
}

function computeIoU(a, b) {
  const iw = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const ih = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = iw * ih;
  return inter / Math.max(inter + Math.max(0, (a.x2 - a.x1) * (a.y2 - a.y1)) + Math.max(0, (b.x2 - b.x1) * (b.y2 - b.y1)) - inter, 1e-8);
}

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

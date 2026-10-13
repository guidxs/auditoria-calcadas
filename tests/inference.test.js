import test from 'node:test';
import assert from 'node:assert/strict';
import { CLASS_NAMES, EXPECTED_OUTPUT, getLetterboxGeometry, mapBox, postprocess } from '../src/inference.js';

function outputWith(boxes) {
  const [, channels, count] = EXPECTED_OUTPUT;
  const data = new Float32Array(channels * count);
  for (const box of boxes) {
    const i = box.index;
    data[i] = box.cx; data[count + i] = box.cy;
    data[2 * count + i] = box.w; data[3 * count + i] = box.h;
    data[(4 + box.classId) * count + i] = box.score;
  }
  return { dims: EXPECTED_OUTPUT, data };
}

test('declares the six trained class IDs in order', () => {
  assert.deepEqual(CLASS_NAMES, ['animal', 'traffic_lights', 'waste_container', 'pothole', 'cracks', 'open_manhole']);
});

test('decodes channel-major YOLO output and applies confidence filtering', () => {
  const out = outputWith([
    { index: 0, cx: 100, cy: 100, w: 40, h: 50, classId: 3, score: 0.81 },
    { index: 1, cx: 200, cy: 200, w: 50, h: 50, classId: 5, score: 0.20 },
  ]);
  const detections = postprocess(out, 0.35, 0.45);
  assert.equal(detections.length, 1);
  assert.equal(detections[0].classId, 3);
  assert.equal(detections[0].conf, Math.fround(0.81));
  assert.deepEqual([detections[0].x1, detections[0].y1, detections[0].x2, detections[0].y2], [80, 75, 120, 125]);
});

test('NMS suppresses overlap in a class but keeps an overlapping other class', () => {
  const out = outputWith([
    { index: 0, cx: 100, cy: 100, w: 60, h: 60, classId: 3, score: 0.9 },
    { index: 1, cx: 102, cy: 101, w: 60, h: 60, classId: 3, score: 0.8 },
    { index: 2, cx: 101, cy: 100, w: 60, h: 60, classId: 4, score: 0.7 },
  ]);
  const detections = postprocess(out, 0.35, 0.45);
  assert.deepEqual(detections.map((d) => d.classId), [3, 4]);
});

test('maps letterboxed coordinates back to source dimensions and clips borders', () => {
  const state = { srcW: 1280, srcH: 720, scale: 0.5, padX: 0, padY: 140 };
  assert.deepEqual(mapBox({ x1: -10, y1: 140, x2: 640, y2: 500 }, state), {
    x1: 0, y1: 0, x2: 1280, y2: 720,
  });
});

test('matches Ultralytics letterbox rounding for odd padding', () => {
  const g = getLetterboxGeometry(1280, 721);
  assert.equal(g.scale, 0.5);
  assert.deepEqual([g.width, g.height], [640, 360]);
  assert.deepEqual([g.padX, g.padRight, g.padY, g.padBottom], [0, 0, 140, 140]);
});

test('rejects unexpected tensor shape', () => {
  assert.throws(() => postprocess({ dims: [1, 10, 756], data: new Float32Array(7560) }), /esperado/);
});

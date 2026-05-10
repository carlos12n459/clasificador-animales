const TIPOS = {
  1: {
    nombre: 'Mam\u00edfero',
    emoji: '\ud83d\udc3b',
    desc: 'Vertebrado con pelo que produce leche.',
    color: '#4CAF50'
  },
  2: {
    nombre: 'Ave',
    emoji: '\ud83e\udda5',
    desc: 'Vertebrado con plumas, generalmente volador.',
    color: '#2196F3'
  },
  3: {
    nombre: 'Reptil',
    emoji: '\ud83d\udc0d',
    desc: 'Vertebrado de sangre fr\u00eda con escamas.',
    color: '#795548'
  },
  4: {
    nombre: 'Pez',
    emoji: '\ud83d\udc1f',
    desc: 'Vertebrado acu\u00e1tico con aletas.',
    color: '#00BCD4'
  },
  5: {
    nombre: 'Anfibio',
    emoji: '\ud83d\udc38',
    desc: 'Vertebrado que vive en agua y tierra.',
    color: '#9C27B0'
  },
  6: {
    nombre: 'Insecto',
    emoji: '\ud83e\uddb8',
    desc: 'Invertebrado con 6 patas y exoesqueleto.',
    color: '#FF9800'
  },
  7: {
    nombre: 'Invertebrado',
    emoji: '\ud83e\udd9e',
    desc: 'Sin columna vertebral (moluscos, ar\u00e1cnidos, etc.).',
    color: '#F44336'
  }
};
const ALL_FEATURES = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize'];
const TOP7 = ['hair', 'eggs', 'milk', 'aquatic', 'toothed', 'feathers', 'fins'];
const DEFAULTS = {
  airborne: 0,
  predator: 0,
  backbone: 1,
  breathes: 1,
  venomous: 0,
  legs: 4,
  tail: 0,
  domestic: 0,
  catsize: 0
};
let currentModel = 'logistic',
  currentBatchModel = 'logistic';

function estandarizar(vec) {
  return vec.map((v, i) => (v - PESOS.scaler.mean[i]) / PESOS.scaler.scale[i]);
}
const softmax = arr => {
  const mx = Math.max(...arr),
    e = arr.map(x => Math.exp(x - mx)),
    s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
};
const tanh = x => Math.tanh(x);

function predRL(vec) {
  const {
    coef,
    intercept,
    classes
  } = PESOS.logistic;
  const scores = intercept.map((b, k) => {
    let s = b;
    for (let i = 0; i < vec.length; i++) s += coef[k][i] * vec[i];
    return s;
  });
  const proba = softmax(scores);
  return {
    pred: classes[proba.indexOf(Math.max(...proba))],
    proba,
    classes
  };
}

function predRN(vec) {
  const {
    coefs,
    intercepts
  } = PESOS.neural;
  let a = vec.slice();
  for (let l = 0; l < coefs.length - 1; l++) {
    const W = coefs[l],
      b = intercepts[l];
    let next = [];
    for (let j = 0; j < W[0].length; j++) {
      let s = b[j];
      for (let i = 0; i < a.length; i++) s += a[i] * W[i][j];
      next.push(tanh(s));
    }
    a = next;
  }
  const Wo = coefs[coefs.length - 1],
    bo = intercepts[intercepts.length - 1];
  let scores = [];
  for (let j = 0; j < Wo[0].length; j++) {
    let s = bo[j];
    for (let i = 0; i < a.length; i++) s += a[i] * Wo[i][j];
    scores.push(s);
  }
  const proba = softmax(scores);
  return {
    pred: METRICAS.classes[proba.indexOf(Math.max(...proba))],
    proba,
    classes: METRICAS.classes
  };
}

function predecir(muestra, modelo) {
  const vec = ALL_FEATURES.map(f => parseFloat(muestra[f] ?? DEFAULTS[f] ?? 0));
  const norm = estandarizar(vec);
  return modelo === 'neural' ? predRN(norm) : predRL(norm);
}

function getMuestra() {
  const m = {
    ...DEFAULTS
  };
  TOP7.forEach(f => {
    m[f] = document.getElementById('f-' + f).checked ? 1 : 0;
  });
  return m;
}

function setModel(m, btn) {
  currentModel = m;
  document.querySelectorAll('#tab-ind .mbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function setBatchModel(m, btn) {
  currentBatchModel = m;
  document.querySelectorAll('#tab-lot .mbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function predictIndividual() {
  const {
    pred,
    proba,
    classes
  } = predecir(getMuestra(), currentModel);
  const tipo = TIPOS[pred];
  const card = document.getElementById('result-card');
  document.getElementById('result-badge').innerHTML = tipo.emoji + ' ' + tipo.nombre + ' \u2014 Tipo ' + pred;
  document.getElementById('result-badge').style.cssText = 'background:' + tipo.color + '22;color:' + tipo.color + ';border:1px solid ' + tipo.color + '55';
  document.getElementById('result-desc').textContent = tipo.desc;
  document.getElementById('result-model-tag').textContent = currentModel === 'logistic' ? 'Regresi\u00f3n Log\u00edstica' : 'Red Neuronal MLP';
  const sorted = classes.map((c, i) => ({
    c,
    p: proba[i]
  })).sort((a, b) => b.p - a.p);
  document.getElementById('prob-bars').innerHTML = sorted.map(x => '<div class="prob-row"><div class="prob-label">' + TIPOS[x.c].emoji + ' ' + TIPOS[x.c].nombre + '</div><div class="prob-outer"><div class="prob-fill" id="pb-' + x.c + '" style="background:' + TIPOS[x.c].color + ';width:0%"></div></div><div class="prob-val">' + (x.p * 100).toFixed(1) + '%</div></div>').join('');
  card.classList.remove('show');
  void card.offsetHeight;
  card.classList.add('show');
  setTimeout(() => {
    sorted.forEach(x => {
      const el = document.getElementById('pb-' + x.c);
      if (el) el.style.width = (x.p * 100) + '%';
    });
  }, 60);
  card.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest'
  });
}
let csvData = null;

function downloadSample() {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([SAMPLE_CSV], {
    type: 'text/csv'
  }));
  a.download = 'muestra_zoo.csv';
  a.click();
}
const REQUIRED_COLS = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize'];

function validateCSV(txt) {
  const lines = txt.trim().split('\n');
  if (lines.length < 2) return 'El CSV debe tener al menos una fila de datos.';
  const hdrs = lines[0].split(',').map(h => h.trim().toLowerCase());
  const missing = REQUIRED_COLS.filter(c => !hdrs.includes(c));
  if (missing.length) return 'Columnas faltantes: ' + missing.join(', ');
  return null;
}

function handleFile(input) {
  const file = input.files[0];
  if (!file) return;
  document.getElementById('fname').textContent = '\ud83d\udcc4 ' + file.name;
  const r = new FileReader();
  r.onload = e => {
    csvData = e.target.result;
    const err = validateCSV(csvData);
    const errEl = document.getElementById('csv-error');
    if (err) {
      errEl.style.display = 'block';
      errEl.textContent = '\u274c ' + err;
      csvData = null;
    } else errEl.style.display = 'none';
  };
  r.readAsText(file);
}
const zone = document.getElementById('drop-zone');
zone.addEventListener('dragover', e => {
  e.preventDefault();
  zone.classList.add('drag');
});
zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (!f) return;
  document.getElementById('fname').textContent = '\ud83d\udcc4 ' + f.name;
  const r = new FileReader();
  r.onload = ev => {
    csvData = ev.target.result;
    const err = validateCSV(csvData);
    const errEl = document.getElementById('csv-error');
    if (err) {
      errEl.style.display = 'block';
      errEl.textContent = '\u274c ' + err;
      csvData = null;
    } else errEl.style.display = 'none';
  };
  r.readAsText(f);
});

function parseCSV(txt) {
  const lines = txt.trim().split('\n');
  const hdrs = lines[0].split(',').map(h => h.trim().toLowerCase());
  return lines.slice(1).filter(l => l.trim()).map(line => {
    const vals = line.split(',').map(v => v.trim());
    const row = {};
    hdrs.forEach((h, i) => row[h] = vals[i] || '0');
    return row;
  });
}

function predictBatch() {
  const src = csvData || SAMPLE_CSV;
  if (!csvData) document.getElementById('csv-error').style.display = 'none';
  const filas = parseCSV(src);
  if (!filas.length) {
    alert('No se encontraron datos.');
    return;
  }
  const preds = [],
    probas = [];
  for (const f of filas) {
    const r = predecir(f, currentBatchModel);
    preds.push(r.pred);
    probas.push(r.proba);
  }
  document.getElementById('batch-res').style.display = 'block';
  const sm = METRICAS[currentBatchModel];
  document.getElementById('metrics-row').innerHTML = metCard('Exactitud', (sm.accuracy * 100).toFixed(1) + '%') + metCard('Precisi\u00f3n', (sm.precision * 100).toFixed(1) + '%') + metCard('Recall', (sm.recall * 100).toFixed(1) + '%') + metCard('F1-Score', (sm.f1 * 100).toFixed(1) + '%');
  drawCM(sm.confusion_matrix);
  const tb = document.getElementById('pred-tbody');
  tb.innerHTML = '';
  filas.forEach((f, i) => {
    const p = preds[i],
      tipo = TIPOS[p],
      conf = (Math.max(...probas[i]) * 100).toFixed(1);
    tb.innerHTML += '<tr><td>' + (i + 1) + '</td><td>' + (f.hair === '1' ? 'S\u00ed' : 'No') + '</td><td>' + (f.eggs === '1' ? 'S\u00ed' : 'No') + '</td><td>' + (f.feathers === '1' ? 'S\u00ed' : 'No') + '</td><td><span style="display:inline-block;padding:.13rem .45rem;border-radius:4px;font-size:.67rem;font-weight:600;background:' + tipo.color + '22;color:' + tipo.color + ';border:1px solid ' + tipo.color + '44">' + tipo.emoji + ' ' + tipo.nombre + '</span></td><td>' + conf + '%</td></tr>';
  });
  document.getElementById('batch-res').scrollIntoView({
    behavior: 'smooth'
  });
}

function metCard(name, val) {
  return '<div class="met-card"><div class="met-val">' + val + '</div><div class="met-name">' + name + '</div></div>';
}

function drawCM(cm) {
  const labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];
  let h = '<div class="cm-scroll"><table class="cm-tbl"><thead><tr><th></th>';
  for (const l of labels) h += '<th>' + l + '</th>';
  h += '</tr></thead><tbody>';
  for (let i = 0; i < 7; i++) {
    h += '<tr><td class="cm-rl">' + labels[i] + '</td>';
    for (let j = 0; j < 7; j++) {
      const v = cm[i]?.[j] ?? 0;
      h += '<td class="' + (v === 0 ? 'cm-z' : i === j ? 'cm-d' : 'cm-e') + '">' + v + '</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table></div>';
  document.getElementById('cm-box').innerHTML = h;
}

function switchTab(t, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + t).classList.add('active');
  btn.classList.add('active');
}

function buildChart() {
  const metrics = [{
    label: 'Exactitud',
    rl: METRICAS.logistic.accuracy,
    rn: METRICAS.neural.accuracy
  }, {
    label: 'Precisi\u00f3n',
    rl: METRICAS.logistic.precision,
    rn: METRICAS.neural.precision
  }, {
    label: 'Recall',
    rl: METRICAS.logistic.recall,
    rn: METRICAS.neural.recall
  }, {
    label: 'F1-Score',
    rl: METRICAS.logistic.f1,
    rn: METRICAS.neural.f1
  }];
  document.getElementById('chart-bars').innerHTML = metrics.map(m => '<div class="bar-row"><div class="lbl">' + m.label + '</div><div><div style="font-size:.59rem;color:var(--muted);margin-bottom:2px">RL</div><div class="bar-outer"><div class="bar-fill bar-rl" style="width:' + (m.rl * 100) + '%">' + (m.rl * 100).toFixed(0) + '%</div></div></div><div><div style="font-size:.59rem;color:var(--muted);margin-bottom:2px">RN</div><div class="bar-outer"><div class="bar-fill bar-rn" style="width:' + (m.rn * 100) + '%">' + (m.rn * 100).toFixed(0) + '%</div></div></div></div>').join('');
}
buildChart();
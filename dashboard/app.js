(() => {
  const data = window.RAVAN_DASHBOARD_DATA;
  if (!data || !Array.isArray(data.episodes) || !data.episodes.length) {
    document.body.innerHTML = '<main><h1>Dashboard data is missing</h1><p>Run the Phase 9 dashboard builder from the project root.</p></main>';
    return;
  }

  const $ = (id) => document.getElementById(id);
  const select = $('episode-select');
  const slider = $('step-slider');
  const episodes = data.episodes;
  let episodeIndex = 0;
  let sampleIndex = 0;

  const scenarioNames = {
    S1: 'Steady following', S2: 'Closing on a steady lead', S3: 'Lead braking, no ego response',
    S4: 'Lead braking, delayed ego response', S5: 'Safe braking control', S6: 'Non-closing pair', S7: 'Vehicles braking to rest'
  };
  const actionLabels = {
    monitor: 'Monitor', caution: 'Caution', increase_following_gap: 'Increase following gap',
    prepare_to_slow: 'Prepare to slow', critical_warning: 'Critical warning'
  };
  const reasonLabels = {
    CURRENT_CONTACT: 'Contact is present in the current sample.',
    REQUIRED_INPUT_MISSING_OR_INVALID: 'A required state input is missing or invalid; use caution.',
    PHYSICS_TTC_CRITICAL: 'Current valid TTC is at or below 1.5 seconds.',
    PHYSICS_TTC_HIGH: 'Current valid TTC is at or below 3 seconds.',
    MODEL_SCORE_PREPARE_THRESHOLD: 'The model score crossed the validation-calibrated prepare-to-slow threshold.',
    PHYSICS_TTC_MODERATE: 'Current valid TTC is at or below 5 seconds.',
    PHYSICS_THW_SHORT: 'Current time headway is at or below 2 seconds.',
    MODEL_SCORE_CAUTION_THRESHOLD: 'The model score crossed the validation-calibrated caution threshold.',
    NO_ACTION_THRESHOLD_MET: 'No current physics or model threshold requires an action.'
  };

  episodes.forEach((ep, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${ep.episodeId} · ${ep.scenarioId}`;
    select.appendChild(option);
  });

  function currentEpisode() { return episodes[episodeIndex]; }
  function currentSample() { return currentEpisode().samples[sampleIndex]; }
  function fmt(value, digits = 2) { return value === null || value === undefined ? 'Undefined' : Number(value).toFixed(digits); }
  function kmh(value) { return `${fmt(value * 3.6, 1)} km/h`; }

  function renderIndicators(sample) {
    const items = [
      ['Ego speed', kmh(sample.egoSpeedMps)], ['Lead speed', kmh(sample.leadSpeedMps)],
      ['Bumper gap', `${fmt(sample.gapM)} m`], ['Closing speed', `${fmt(sample.relativeSpeedMps)} m/s`],
      ['Time to collision', sample.ttcValid ? `${fmt(sample.ttcS)} s` : 'Undefined'],
      ['Time headway', sample.thwValid ? `${fmt(sample.thwS)} s` : 'Undefined'],
      ['Required deceleration', sample.dracValid ? `${fmt(sample.dracMps2)} m/s²` : 'Undefined'],
      ['Current TTC category', sample.currentRisk]
    ];
    $('indicator-grid').replaceChildren(...items.map(([label, value]) => {
      const cell = document.createElement('div'); cell.className = 'indicator';
      const title = document.createElement('span'); title.className = 'indicator-label'; title.textContent = label;
      const content = document.createElement('strong'); content.className = 'indicator-value'; content.textContent = value;
      cell.append(title, content); return cell;
    }));
  }

  function renderScenario(ep) {
    const p = ep.parameters;
    $('scenario-title').textContent = ep.episodeId;
    $('scenario-family').textContent = ep.scenarioId;
    $('scenario-description').textContent = scenarioNames[ep.scenarioId] || 'Longitudinal vehicle interaction';
    const chips = [
      `Initial ego ${fmt(p.ego_initial_speed_mps, 1)} m/s`,
      `Initial lead ${fmt(p.lead_initial_speed_mps, 1)} m/s`,
      `Initial gap ${fmt(p.initial_gap_m, 1)} m`,
      `Lead length ${fmt(p.lead_length_m, 1)} m`
    ];
    if (p.lead_brake_onset_s !== null) chips.push(`Lead brake at ${fmt(p.lead_brake_onset_s, 1)} s`);
    if (p.ego_reaction_delay_s !== null) chips.push(`Ego delay ${fmt(p.ego_reaction_delay_s, 1)} s`);
    $('scenario-parameters').replaceChildren(...chips.map((text) => {
      const chip = document.createElement('span'); chip.className = 'parameter-chip'; chip.textContent = text; return chip;
    }));
  }

  function renderChart(ep, active) {
    const svg = $('gap-chart');
    const width = Math.max(320, svg.clientWidth || 800), height = 190;
    const pad = { left: 48, right: 16, top: 12, bottom: 28 };
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const points = ep.samples;
    const values = points.map((p) => p.gapM);
    const min = Math.min(0, ...values), max = Math.max(10, ...values);
    const yMin = min - Math.max(2, (max - min) * .08), yMax = max + Math.max(2, (max - min) * .08);
    const x = (t) => pad.left + (t / Math.max(ep.samples.at(-1).timeS, .1)) * (width - pad.left - pad.right);
    const y = (v) => pad.top + ((yMax - v) / (yMax - yMin)) * (height - pad.top - pad.bottom);
    const NS = 'http://www.w3.org/2000/svg';
    const el = (name, attrs = {}) => { const node = document.createElementNS(NS, name); Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v)); return node; };
    for (let i = 0; i <= 4; i++) {
      const value = yMin + (yMax - yMin) * i / 4;
      svg.appendChild(el('line', { x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value), class: 'chart-grid' }));
      const label = el('text', { x: pad.left - 8, y: y(value) + 3, 'text-anchor': 'end', class: 'chart-axis' });
      label.textContent = value.toFixed(0); svg.appendChild(label);
    }
    const zeroY = y(0);
    svg.appendChild(el('line', { x1: pad.left, x2: width - pad.right, y1: zeroY, y2: zeroY, class: 'chart-zero' }));
    const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.timeS).toFixed(2)},${y(p.gapM).toFixed(2)}`).join(' ');
    svg.appendChild(el('path', { d: path, class: 'chart-line' }));
    const selected = el('circle', { cx: x(active.timeS), cy: y(active.gapM), r: 5, class: 'chart-current' });
    svg.appendChild(selected);
    const xStart = el('text', { x: pad.left, y: height - 7, class: 'chart-axis' }); xStart.textContent = '0 s'; svg.appendChild(xStart);
    const xEnd = el('text', { x: width - pad.right, y: height - 7, 'text-anchor': 'end', class: 'chart-axis' }); xEnd.textContent = `${ep.samples.at(-1).timeS.toFixed(1)} s`; svg.appendChild(xEnd);
    $('chart-caption').textContent = `Source: synthetic Phase 2 trajectory · ${ep.episodeId} · ${points.length} samples at 10 Hz · x-axis: simulation time (s) · y-axis: bumper gap (m)`;
  }

  function renderContributions(sample) {
    const list = $('contribution-list');
    const items = sample.topContributions;
    const maxAbs = Math.max(.1, ...items.map((item) => Math.abs(item.contribution)));
    list.replaceChildren(...items.map((item) => {
      const row = document.createElement('div'); row.className = 'contribution-row';
      const name = document.createElement('span'); name.className = 'contribution-name'; name.textContent = item.name;
      const track = document.createElement('div'); track.className = 'bar-track';
      const zero = document.createElement('span'); zero.className = 'bar-zero'; track.appendChild(zero);
      const bar = document.createElement('span');
      const magnitude = Math.abs(item.contribution) / maxAbs * 48;
      const positive = item.contribution >= 0;
      bar.className = `bar-value${positive ? '' : ' negative'}`;
      bar.style.width = `${magnitude}%`;
      bar.style.left = positive ? '50%' : `${50 - magnitude}%`;
      track.appendChild(bar);
      const value = document.createElement('span'); value.className = 'contribution-number'; value.textContent = `${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(3)}`;
      row.append(name, track, value); return row;
    }));
    $('base-logit-note').textContent = `Base log-odds ${sample.baseLogOdds.toFixed(3)} · current logit ${sample.logit.toFixed(3)} · contributions shown are the six largest by absolute value.`;
  }

  function render() {
    const ep = currentEpisode();
    const sample = currentSample();
    renderScenario(ep);
    const probability = sample.score;
    $('score-value').textContent = `${(probability * 100).toFixed(1)}%`;
    $('score-meter').style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    $('score-meter').classList.toggle('positive', sample.binaryPrediction === 1);
    const predicted = $('prediction-status');
    predicted.textContent = sample.binaryPrediction ? 'Predicted conflict' : 'No conflict predicted';
    predicted.className = `status-pill ${sample.binaryPrediction ? 'positive' : 'negative'}`;
    $('target-label').textContent = sample.target === null ? 'Censored' : sample.target ? 'Conflict label: positive' : 'Conflict label: negative';
    $('recommendation-title').textContent = actionLabels[sample.recommendation] || sample.recommendation;
    $('recommendation-reason').textContent = reasonLabels[sample.reason] || sample.reason;
    const disagreement = $('disagreement-pill');
    disagreement.textContent = sample.disagreement.replaceAll('_', ' ');
    disagreement.className = `disagreement-pill${sample.disagreement === 'agreement_or_partial' ? '' : ' flagged'}`;
    $('quality-status').textContent = `Input quality: ${sample.dataQuality.replaceAll('_', ' ')}`;
    renderIndicators(sample);
    renderChart(ep, sample);
    renderContributions(sample);
    slider.max = String(ep.samples.length - 1);
    slider.value = String(sampleIndex);
    $('step-readout').textContent = `${sample.timeS.toFixed(1)} s · step ${sample.step + 1} of ${ep.samples.length}`;
    $('data-provenance').textContent = `${data.episodeCount} held-out episodes · score reconciliation Δ ${data.maxProbabilityDeltaFromPhase7.toExponential(1)}`;
  }

  select.addEventListener('change', () => { episodeIndex = Number(select.value); sampleIndex = 0; render(); });
  slider.addEventListener('input', () => { sampleIndex = Number(slider.value); render(); });
  $('previous-button').addEventListener('click', () => { sampleIndex = Math.max(0, sampleIndex - 1); render(); });
  $('next-button').addEventListener('click', () => { sampleIndex = Math.min(currentEpisode().samples.length - 1, sampleIndex + 1); render(); });
  window.addEventListener('resize', () => renderChart(currentEpisode(), currentSample()));
  render();
})();

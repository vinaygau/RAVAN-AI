"""Paired held-out evaluation with trajectory CSV and an offline replay page."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from src.ravan_game.rl.actions import DriverAction
from src.ravan_game.rl.baseline import PhysicsRulePolicy
from src.ravan_game.rl.environment import RavanDrivingEnv
from src.ravan_game.rl.scenarios import TEST_SCENARIOS
from src.ravan_game.rl.shield import PhysicsSafetyShield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="PPO model .zip file")
    parser.add_argument("--episodes-per-scenario", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=TEST_SCENARIOS, default=list(TEST_SCENARIOS))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def _finite_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize(rows: list[dict]) -> dict:
    result = {"episodes": len(rows)}
    for key in (
        "collision", "completed", "near_miss_steps", "high_risk_seconds",
        "distance_m", "mean_speed_kmh", "minimum_ttc_s", "minimum_front_gap_m",
        "total_reward", "mean_action_latency_ms", "lane_changes", "shield_override_steps",
        "fallback_active_steps",
    ):
        values = [float(row[key]) for row in rows if row[key] is not None]
        result[f"{key}_mean"] = _finite_mean(values)
        result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    result["collision_rate"] = _finite_mean([float(row["collision"]) for row in rows])
    result["completion_rate"] = _finite_mean([float(row["completed"]) for row in rows])
    return result


def _write_results_page(path: Path, summary: dict, paired: list[dict], replay: list[dict]) -> None:
    payload = json.dumps({"summary": summary, "paired": paired, "episodes": replay}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ravan AI — PPO Evaluation</title>
<style>
:root{color-scheme:dark;--bg:#06111d;--panel:#0b1c2b;--line:#123d56;--cyan:#16c9ee;--green:#32dc77;--red:#ff5b57;--muted:#9bb1c2}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 50% -20%,#123753,var(--bg) 65%);color:#eff8ff;font:15px Segoe UI,Arial,sans-serif}main{max-width:1200px;margin:auto;padding:26px}h1{margin:0 0 4px;font-size:28px}p{color:var(--muted);margin:6px 0 18px}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}.card{background:linear-gradient(145deg,#0d2536,#071521);border:1px solid var(--line);border-radius:14px;padding:15px}.metric{font-size:25px;font-weight:700;color:var(--cyan)}.label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.section{margin-top:20px}.tablewrap{overflow:auto}.table{border-collapse:collapse;width:100%;min-width:760px}.table th,.table td{text-align:left;padding:10px;border-bottom:1px solid #17344a}.table th{color:#99b2c6;font-size:12px;text-transform:uppercase}.pill{border-radius:99px;padding:4px 9px;background:#0c3140;color:var(--cyan)}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.controls select,.controls button{background:#10283a;border:1px solid #26617d;color:#eafbff;border-radius:8px;padding:9px}.replay{display:grid;grid-template-columns:minmax(270px,380px) 1fr;gap:16px}.roadbox{background:#09111a;border:1px solid var(--line);border-radius:12px;padding:10px}canvas{width:100%;height:540px;display:block;border-radius:7px;background:#111d28}.chart{width:100%;height:245px;background:#091725;border:1px solid var(--line);border-radius:10px}.small{font-size:13px;color:var(--muted)}input[type=range]{width:min(650px,100%)}.good{color:var(--green)}.bad{color:var(--red)}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.replay{grid-template-columns:1fr}main{padding:16px}}
</style><body><main>
<h1>Ravan AI · PPO Evaluation</h1><p>Held-out traffic scenarios · same seeds for PPO and the physics-rule baseline · local simulation only</p>
<div id="summary" class="grid"></div>
<section class="section card tablewrap"><div class="label">Aggregate results</div><table class="table"><thead><tr><th>Policy</th><th>Episodes</th><th>Collision rate</th><th>Completed</th><th>Near-miss time</th><th>High/critical risk</th><th>Mean speed</th><th>Distance</th><th>Action latency</th></tr></thead><tbody id="results"></tbody></table><p class="small">Near miss is a time step with calculated same-lane TTC ≤ 2.5 s. High/critical exposure is total simulation time in either risk category. Rates and averages are descriptive for this sample, not a safety certification.</p></section>
<section class="section card"><div class="label">Scenario comparison</div><div id="paired" class="tablewrap"></div></section>
<section class="section card"><div class="label">Episode replay</div><div class="controls"><label>Session <select id="session"></select></label><button id="play">▶ Play</button><label>Time <input id="time" type="range" min="0" max="0" value="0"></label><span id="frameLabel" class="small"></span></div><div class="replay"><div class="roadbox"><canvas id="road" width="380" height="540"></canvas><div id="readout" class="small"></div></div><div><canvas id="chart" class="chart" width="800" height="245"></canvas><p class="small">The road replay uses recorded simulation positions. The chart shows ego speed and front-vehicle TTC.</p></div></div></section>
<p class="small section">Ravan AI is an educational simulation. PPO and the physics rules are advisory research policies and do not control a real vehicle.</p>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), fmt=(x,d=1)=>x==null?'—':Number(x).toFixed(d), pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
const sum=document.getElementById('summary');sum.innerHTML=[['Policies',Object.keys(D.summary).length],['Paired test episodes',D.paired.length],['Scenarios',new Set(D.paired.map(x=>x.scenario)).size],['Evaluation', 'Held out']].map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="metric">${x[1]}</div></div>`).join('');
const keys=Object.keys(D.summary),tb=document.getElementById('results');tb.innerHTML=keys.map(k=>{let s=D.summary[k];return `<tr><td><span class="pill">${k}</span></td><td>${s.episodes}</td><td>${pct(s.collision_rate)}</td><td>${pct(s.completion_rate)}</td><td>${fmt(s.near_miss_steps_mean*.1)} s</td><td>${fmt(s.high_risk_seconds_mean)} s</td><td>${fmt(s.mean_speed_kmh_mean)} km/h</td><td>${fmt(s.distance_m_mean)} m</td><td>${fmt(s.mean_action_latency_ms_mean,2)} ms</td></tr>`}).join('');
document.getElementById('paired').innerHTML=`<table class="table"><thead><tr><th>Scenario</th><th>Seed</th><th>Rule collision</th><th>Raw PPO</th><th>Shielded PPO</th><th>Rule near-miss steps</th><th>Raw PPO</th><th>Shielded PPO</th><th>Shield overrides</th><th>Rule fallback steps</th></tr></thead><tbody>${D.paired.map(x=>`<tr><td>${x.scenario}</td><td>${x.seed}</td><td>${x.rule_collision?'Yes':'No'}</td><td>${x.ppo_collision?'Yes':'No'}</td><td>${x.shield_collision?'Yes':'No'}</td><td>${x.rule_near_miss_steps}</td><td>${x.ppo_near_miss_steps}</td><td>${x.shield_near_miss_steps}</td><td>${x.shield_overrides}</td><td>${x.fallback_active_steps}</td></tr>`).join('')}</tbody></table>`;
const sel=document.getElementById('session'),slider=document.getElementById('time'),road=document.getElementById('road'),ctx=road.getContext('2d'),chart=document.getElementById('chart'),cx=chart.getContext('2d');let current=null,timer=null;
D.episodes.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${e.policy} · ${e.scenario} · seed ${e.seed}`;sel.appendChild(o)});
function drawChart(ep,idx){let w=chart.width,h=chart.height;cx.clearRect(0,0,w,h);cx.fillStyle='#091725';cx.fillRect(0,0,w,h);let fs=ep.frames, maxS=Math.max(1,...fs.map(f=>f.speed_kmh)), n=fs.length;for(let j=0;j<5;j++){let y=20+j*(h-45)/4;cx.strokeStyle='#1c3548';cx.beginPath();cx.moveTo(40,y);cx.lineTo(w-15,y);cx.stroke()}function line(key,color,max){cx.strokeStyle=color;cx.lineWidth=2;cx.beginPath();fs.forEach((f,i)=>{let x=40+i*(w-60)/Math.max(1,n-1),val=f[key]==null?max:Math.min(max,f[key]),y=h-25-(val/max)*(h-55);if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y)});cx.stroke();let f=fs[idx];if(f){let x=40+idx*(w-60)/Math.max(1,n-1),y=h-25-(Math.min(max,f[key]??max)/max)*(h-55);cx.fillStyle=color;cx.beginPath();cx.arc(x,y,5,0,Math.PI*2);cx.fill()}}line('speed_kmh','#12d0ef',Math.ceil(maxS/10)*10);line('ttc_s','#ffc83d',10);cx.fillStyle='#b7c9d5';cx.font='12px Segoe UI';cx.fillText('Speed (km/h) · cyan',45,15);cx.fillText('TTC (s) · yellow',190,15);cx.fillText('episode time →',w-100,h-5)}
function draw(){if(!current)return;let i=Number(slider.value),f=current.frames[i];slider.max=current.frames.length-1;slider.value=i;document.getElementById('frameLabel').textContent=`${f.time_s.toFixed(1)} s · ${i+1}/${current.frames.length} · ${f.risk}`;let w=road.width,h=road.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#153a24';ctx.fillRect(0,0,w,h);ctx.fillStyle='#24313b';ctx.beginPath();ctx.moveTo(45,0);ctx.lineTo(w-45,0);ctx.lineTo(w-5,h);ctx.lineTo(5,h);ctx.closePath();ctx.fill();ctx.strokeStyle='#dbe5eb';ctx.lineWidth=3;ctx.setLineDash([18,17]);for(let lane=1;lane<3;lane++){let x=w*lane/3;ctx.beginPath();ctx.moveTo(w/2+(x-w/2)*.35,0);ctx.lineTo(x,h);ctx.stroke()}ctx.setLineDash([]);function car(x,y,color,label){x=45+x/10.8*(w-90);y=h/2-floatClamp(y,-65,65)*4;ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=12;ctx.beginPath();ctx.roundRect(x-14,y-22,28,44,7);ctx.fill();ctx.shadowBlur=0;ctx.fillStyle='#fff';ctx.font='10px Segoe UI';ctx.textAlign='center';ctx.fillText(label,x,y+4)}(f.vehicles||[]).forEach(v=>car(v.x,v.dy,'#24a9d8',Math.round(v.speed_kmh)));car(f.ego_x,0,f.risk==='CRITICAL'||f.risk==='HIGH'?'#ff5252':'#37e17c','EGO');document.getElementById('readout').textContent=`${f.speed_kmh.toFixed(1)} km/h · Risk ${f.risk} · Front gap ${f.front_gap_m==null?'—':f.front_gap_m.toFixed(1)+' m'} · TTC ${f.ttc_s==null?'—':f.ttc_s.toFixed(1)+' s'} · ${f.override_reason||'No shield override'}`;drawChart(current,i)}
function floatClamp(x,a,b){return Math.max(a,Math.min(b,x))}sel.onchange=()=>{current=D.episodes[Number(sel.value)];slider.value=0;draw()};slider.oninput=draw;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play'}else{document.getElementById('play').textContent='❚❚ Pause';timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play';return}slider.value=Number(slider.value)+1;draw()},120)}};sel.onchange();
</script></body></html>'''
    path.write_text(page.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.episodes_per_scenario < 1 or args.max_steps < 1:
        raise SystemExit("episodes-per-scenario and max-steps must be positive.")
    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    output_dir = args.output_dir or ROOT / "data" / "rl_evaluations" / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(model_path, device=args.device)
    rule = PhysicsRulePolicy()
    shield_fallback = PhysicsRulePolicy()
    policies = ("physics_rule", "ppo", "ppo_shielded")
    episode_rows: list[dict] = []
    replay_episodes: list[dict] = []
    trajectory_path = output_dir / "trajectories.csv"
    fields = (
        "policy", "scenario", "seed", "step", "time_s", "action", "reward",
        "speed_kmh", "lane", "lateral_m", "longitudinal_m", "front_gap_m",
        "front_speed_kmh", "ttc_s", "thw_s", "drac_mps2", "risk", "risk_score",
        "collision", "action_latency_ms", "proposed_action", "shield_override",
        "shield_reason", "fallback_active", "fallback_reason", "nearby_vehicles_json",
    )

    with trajectory_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for scenario in args.scenarios:
            for repetition in range(args.episodes_per_scenario):
                seed = args.seed + repetition
                for policy_name in policies:
                    env = RavanDrivingEnv(
                        max_episode_steps=args.max_steps,
                        scenario_families=(scenario,),
                    )
                    obs, reset_info = env.reset(seed=seed, options={"scenario_family": scenario})
                    rule.reset()
                    shield_fallback.reset()
                    samples = {
                        "ttc": [], "front_gap": [], "speed": [], "near_miss": 0,
                        "high_risk_steps": 0, "lane_changes": 0,
                        "distance_m": 0.0, "total_reward": 0.0, "latency": [],
                        "shield_overrides": 0,
                        "fallback_active_steps": 0,
                    }
                    frames = []
                    shield = PhysicsSafetyShield()
                    fallback_active = False
                    fallback_reason = None
                    initial_pos = env.simulation.player.longitudinal_m
                    previous_lane = env.simulation.player.lane_index
                    collision = False
                    completed = False
                    for step in range(args.max_steps):
                        start = perf_counter()
                        override_reason = None
                        if policy_name == "physics_rule":
                            proposed_action = rule.act(env)
                        else:
                            predicted, _ = model.predict(obs, deterministic=True)
                            proposed_action = DriverAction(int(predicted))
                        action = proposed_action
                        if policy_name == "ppo_shielded":
                            if fallback_active:
                                action = shield_fallback.act(env)
                                samples["fallback_active_steps"] += 1
                            else:
                                action, override_reason = shield.filter(
                                    env,
                                    proposed_action,
                                    fallback_policy=shield_fallback.act,
                                )
                                if override_reason is not None:
                                    fallback_active = True
                                    fallback_reason = override_reason
                                    samples["shield_overrides"] += 1
                        latency_ms = (perf_counter() - start) * 1000.0
                        obs, reward, terminated, truncated, info = env.step(action)
                        sim = env.simulation
                        risk = env._assessment
                        ego = sim.player
                        risk_dict = risk.to_dict()
                        samples["speed"].append(ego.speed_mps * 3.6)
                        samples["latency"].append(latency_ms)
                        samples["total_reward"] += reward
                        samples["distance_m"] = ego.longitudinal_m - initial_pos
                        samples["near_miss"] += int(risk.ttc_s is not None and risk.ttc_s <= 2.5)
                        samples["high_risk_steps"] += int(risk.risk_level in ("HIGH", "CRITICAL"))
                        samples["ttc"].append(risk.ttc_s) if risk.ttc_s is not None else None
                        samples["front_gap"].append(risk.distance_m) if risk.distance_m is not None else None
                        if ego.lane_index != previous_lane:
                            samples["lane_changes"] += 1
                            previous_lane = ego.lane_index
                        nearby = [
                            {
                                "x": round(agent.vehicle.lateral_m, 2),
                                "dy": round(agent.vehicle.longitudinal_m - ego.longitudinal_m, 2),
                                "speed_kmh": round(agent.vehicle.speed_mps * 3.6, 1),
                            }
                            for agent in sorted(
                                sim.traffic,
                                key=lambda item: abs(item.vehicle.longitudinal_m - ego.longitudinal_m),
                            )[:8]
                        ]
                        trajectory_row = {
                            "policy": policy_name,
                            "scenario": scenario,
                            "seed": seed,
                            "step": step + 1,
                            "time_s": info["simulation_time_s"],
                            "action": DriverAction(int(action)).name,
                            "proposed_action": DriverAction(int(proposed_action)).name,
                            "shield_override": bool(override_reason),
                            "shield_reason": override_reason or "",
                            "fallback_active": fallback_active,
                            "fallback_reason": fallback_reason or "",
                            "reward": round(reward, 5),
                            "speed_kmh": round(ego.speed_mps * 3.6, 2),
                            "lane": ego.lane_index,
                            "lateral_m": round(ego.lateral_m, 3),
                            "longitudinal_m": round(ego.longitudinal_m, 3),
                            "front_gap_m": risk_dict["distance_m"],
                            "front_speed_kmh": (risk.nearest_front or {}).get("speed_kmh"),
                            "ttc_s": risk_dict["ttc_s"],
                            "thw_s": risk_dict["thw_s"],
                            "drac_mps2": risk_dict["drac_mps2"],
                            "risk": risk.risk_level,
                            "risk_score": risk.risk_score,
                            "collision": risk.collision,
                            "action_latency_ms": round(latency_ms, 4),
                            "nearby_vehicles_json": json.dumps(nearby, separators=(",", ":")),
                        }
                        writer.writerow(trajectory_row)
                        if step % 5 == 0 or terminated or truncated:
                            frames.append({
                                "time_s": float(info["simulation_time_s"]),
                                "speed_kmh": float(ego.speed_mps * 3.6),
                                "lane": ego.lane_index,
                                "ego_x": float(ego.lateral_m),
                                "front_gap_m": risk.distance_m,
                                "ttc_s": risk.ttc_s,
                                "risk": risk.risk_level,
                                "override_reason": override_reason or (
                                    f"Physics-rule fallback active after: {fallback_reason}"
                                    if fallback_active and fallback_reason else None
                                ),
                                "fallback_active": fallback_active,
                                "vehicles": nearby,
                            })
                        collision = collision or terminated
                        if terminated or truncated:
                            completed = bool(truncated and not terminated)
                            break
                    env.close()
                    duration_s = (step + 1) * env.fixed_dt_s * env.action_repeat
                    result = {
                        "policy": policy_name,
                        "scenario": scenario,
                        "seed": seed,
                        "steps": step + 1,
                        "duration_s": duration_s,
                        "collision": int(collision),
                        "completed": int(completed),
                        "near_miss_steps": samples["near_miss"],
                        "high_risk_seconds": samples["high_risk_steps"] * env.fixed_dt_s * env.action_repeat,
                        "distance_m": samples["distance_m"],
                        "mean_speed_kmh": _finite_mean(samples["speed"]),
                        "minimum_ttc_s": min(samples["ttc"]) if samples["ttc"] else None,
                        "minimum_front_gap_m": min(samples["front_gap"]) if samples["front_gap"] else None,
                        "total_reward": samples["total_reward"],
                        "mean_action_latency_ms": _finite_mean(samples["latency"]),
                        "lane_changes": samples["lane_changes"],
                        "shield_override_steps": samples["shield_overrides"],
                        "fallback_active_steps": samples["fallback_active_steps"],
                    }
                    episode_rows.append(result)
                    replay_episodes.append({
                        "policy": policy_name,
                        "scenario": scenario,
                        "seed": seed,
                        "collision": bool(collision),
                        "completed": bool(completed),
                        "shield_override_steps": samples["shield_overrides"],
                        "fallback_active_steps": samples["fallback_active_steps"],
                        "frames": frames,
                    })
                    print(
                        f"{policy_name:12s} {scenario:24s} seed={seed} "
                        f"collision={collision} completion={completed} "
                        f"distance={result['distance_m']:.1f}m"
                    )

    grouped = {name: [row for row in episode_rows if row["policy"] == name] for name in policies}
    summary = {name: _summarize(rows) for name, rows in grouped.items()}
    paired = []
    for scenario in args.scenarios:
        for repetition in range(args.episodes_per_scenario):
            seed = args.seed + repetition
            rule_row = next(row for row in episode_rows if row["scenario"] == scenario and row["seed"] == seed and row["policy"] == "physics_rule")
            ppo_row = next(row for row in episode_rows if row["scenario"] == scenario and row["seed"] == seed and row["policy"] == "ppo")
            shield_row = next(row for row in episode_rows if row["scenario"] == scenario and row["seed"] == seed and row["policy"] == "ppo_shielded")
            paired.append({
                "scenario": scenario,
                "seed": seed,
                "rule_collision": rule_row["collision"],
                "ppo_collision": ppo_row["collision"],
                "rule_near_miss_steps": rule_row["near_miss_steps"],
                "ppo_near_miss_steps": ppo_row["near_miss_steps"],
                "shield_collision": shield_row["collision"],
                "shield_near_miss_steps": shield_row["near_miss_steps"],
                "shield_overrides": shield_row["shield_override_steps"],
                "fallback_active_steps": shield_row["fallback_active_steps"],
                "rule_reward": rule_row["total_reward"],
                "ppo_reward": ppo_row["total_reward"],
            })
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "model": str(model_path),
        "seed_base": args.seed,
        "episodes_per_scenario": args.episodes_per_scenario,
        "scenarios": args.scenarios,
        "policies": summary,
        "episodes": episode_rows,
    }, indent=2), encoding="utf-8")
    _write_results_page(output_dir / "results.html", summary, paired, replay_episodes)
    print(f"Summary: {summary_path}")
    print(f"Replay:  {output_dir / 'results.html'}")
    print(f"CSV:     {trajectory_path}")


if __name__ == "__main__":
    main()

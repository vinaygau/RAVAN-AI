# Phase 1 Scenario Specification

## Purpose and boundary

Phase 1 freezes the inputs and expected behavior for the first Ravan AI pilot: a transparent, one-dimensional, same-lane longitudinal simulator. It is intended to validate vehicle-motion generation and basic following-risk calculations before data generation or model training.

The pilot is a research simulation. It does not model steering, lateral movement, perception, road geometry, traffic rules, or vehicle control. Lane changes, cut-ins, intersections, wrong-way driving, and CARLA integration are out of scope for this phase.

## State and coordinate convention

- Time is measured in seconds; position in metres; speed in metres per second; acceleration in metres per second squared.
- Use a one-dimensional road coordinate that increases in the direction of travel.
- Each vehicle state is its **front-bumper** position, speed, and acceleration. The ego vehicle starts behind the lead vehicle.
- The bumper-to-bumper gap is `lead_front_x - lead_length - ego_front_x`. A non-positive gap means contact/overlap under this simplified geometry.
- Both vehicles use the same time step and are updated synchronously. Acceleration is piecewise constant during each step. If braking would take speed below zero within a step, integrate position only to the within-step stop time, then hold position and speed at zero for the remainder of that step.
- The lead vehicle's length is configured, not inferred. The ego vehicle's length is not needed for this front-bumper convention.

## First-stage scenarios

| ID | Scenario | Controlled setup | Expected behavior |
|---|---|---|---|
| S1 | Steady safe following | Positive gap; equal speeds; no braking | Gap stays constant; no closing TTC is reported. |
| S2 | Closing on steady lead | Ego faster than lead; lead maintains speed | Gap decreases; TTC is finite while the gap is positive and closing speed remains positive. |
| S3 | Lead brakes, ego does not respond | Lead applies configured braking after a delay; ego holds speed | Gap can shrink to contact; the event time is recorded from simulated states. |
| S4 | Lead brakes, ego responds after delay | Ego begins configured braking after the reaction delay | Compare gap and event outcome against S3 using the same initial state. |
| S5 | Safe braking control | Larger initial gap and/or earlier ego response | No contact under the selected scenario settings; verify the result rather than assuming it from the label. |
| S6 | Non-closing pair | Lead is faster than or equal to ego | Gap stays constant or increases; longitudinal TTC is undefined/infinite, never zero. |
| S7 | Ego/lead stop | One or both vehicles decelerate to zero | Speed never becomes negative; position remains non-decreasing. |

Each episode has a stable `scenario_id`, a recorded random seed when randomized, initial conditions, event schedule, and a clean state trace. Phase 1 uses deterministic settings for acceptance cases; parameter sampling is reserved for the pilot generation phase.

## Frozen operational assumptions

1. The first prediction task is not defined in Phase 1. This phase defines scenarios and checks simulator behavior only; it does not assign ML hazard labels.
2. A lead-braking event is a scripted input, not itself a hazard label. A braking episode may remain safe if the ego vehicle preserves separation.
3. TTC is calculated only for a same-lane pair with positive gap and positive closing speed `v_ego - v_lead`. Otherwise it is undefined (`null` in stored data), with an explicit validity flag. Do not replace undefined TTC with zero.
4. TTC uses the current gap and relative speed and is a simplified indicator; it does not account for future acceleration.
5. Contact is detected when the bumper gap is less than or equal to zero. Preserve the first contact time and minimum gap for later evaluation.
6. These assumptions are specific to a one-dimensional prototype and are not universal safety criteria.

## Pilot parameter envelope

These are provisional engineering bounds for deterministic synthetic experiments, not standards or empirical claims. The exact test values are in `../config/phase1.json`.

| Parameter | Pilot range | Notes |
|---|---:|---|
| Time step | 0.1 s | Target 10 Hz as proposed in the research document. |
| Episode duration | 8–12 s | Long enough to observe a scripted braking response in the pilot. |
| Ego initial speed | 10–30 m/s | Stored in m/s; display conversion can happen later. |
| Lead initial speed | 5–28 m/s | Must be non-negative and within configured speed cap. |
| Initial bumper gap | 10–60 m | Must be strictly positive. |
| Lead length | 4–5 m | Explicit geometry input for the bumper-gap convention. |
| Lead braking onset | 2–6 s | Must occur before episode end. |
| Lead deceleration magnitude | 2–6 m/s² | Applied as negative acceleration. |
| Ego reaction delay | 0–1.5 s | Zero means immediate response after the defined trigger. |
| Ego braking magnitude | 2–6 m/s² | Applied after reaction delay when the response policy is enabled. |
| Acceleration limits | -8 to +3 m/s² | Clamp commanded acceleration to this configured envelope. |
| Speed cap | 35 m/s | Clamp speed to `[0, speed_cap]`. |

Initial values must satisfy `ego_front_x = 0`, `lead_front_x = initial_gap + lead_length`, non-negative speeds, positive gap, and valid event timing. Scenario sampling must retain the sampled inputs in episode metadata.

## Acceptance checks

Phase 1 is complete when the scenario definitions, parameter configuration, and test cases are committed and internally consistent; the coordinate convention and event timing are unambiguous; and every acceptance case has a measurable expected result. The simulation implementation and execution of these cases belong to later work.

## Explicitly deferred

Risk-label thresholds, future-hazard event definition, prediction horizon labels, dataset size, model selection, train/validation/test splits, noise models, recommendation policy, dashboard, and simulator integration remain open for later phases. Do not treat the provisional parameter envelope as validated until checked in the selected simulator.

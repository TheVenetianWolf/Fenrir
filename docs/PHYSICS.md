# 🧠 FENRIR Physics Explained
## Understanding How the Simulation Works, Without the Equations Overload

FENRIR isn’t just a digital toy.
It’s a live model of pursuit and interception, inspired by how guided missiles, animals, and even sports players chase moving targets.

This document explains the core physics and logic in FENRIR, in plain English.
It covers motion, guidance, notation, and what all those mysterious numbers and symbols mean.

## ⚙️ 1. The World Inside FENRIR

At its heart, FENRIR runs a small physical simulation loop:
it advances time in tiny steps (fractions of a second), updating where the missile and target are, how fast they’re moving, and how their paths curve.

The world contains:
-Missile: the pursuer, controlled by a guidance law.
-Target: the object being chased, which can move straight or take evasive actions.
-Environment: which may include air drag (resistance), wind, and later perhaps gravity or noise.

## 🧩 2. Motion and State
Every object has a state, a snapshot of its motion.

| Symbol | Quantity     | Units          | Description                                                                |
| :----: | :----------- | :------------- | :------------------------------------------------------------------------- |
|  **r** | position     | metres (m)     | The object’s location in 2D space, `[x, y]`.                              |
|  **v** | velocity     | m/s            | The rate of change of position, how fast and in which direction it moves. |
|  **a** | acceleration | m/s²           | The rate of change of velocity, how sharply it turns or speeds up.        |
|  **t** | time         | seconds (s)    | The running clock of the simulation.                                       |
|  **m** | mass         | kilograms (kg) | Used for drag and dynamics, though simplified here.                        |


The missile and target move step-by-step through time using a numerical integrator, a fancy term for “updating positions by small amounts repeatedly.”

## 🎯 3. Proportional Navigation (PN)

This is the brain of the missile.
The Proportional Navigation (PN) law is one of the simplest and most elegant ideas in guidance.

It says:

“Turn at a rate proportional to how fast your line-of-sight angle to the target is changing.”

In essence:

- If the target moves across your view, steer toward it.
- If it stays still in your view, you’re already on a collision course.

This simple rule makes missiles (and animals!) naturally intercept moving targets.

### The key parameter — N

The navigation constant, N, controls how aggressively the missile responds.
Typical values are between 2 and 5:

- Low N: sluggish pursuit, more likely to miss.
- High N: aggressive turns, faster interception (but potentially unstable).

## 📐 4. LOS, Range, and Closing Speed

To navigate, the missile measures three main things, all displayed in the telemetry panel:
| Quantity               | Symbol            | Meaning                             | Units | Description                                                      |
| :--------------------- | :---------------- | :---------------------------------- | :---- | :--------------------------------------------------------------- |
| **Range**              | `R`               | Distance between missile and target | m     | The straight-line gap.                                           |
| **Line-of-sight rate** | `λ̇` (lambda dot) | Change in viewing angle             | rad/s | How quickly the target moves across the missile’s field of view. |
| **Closing speed**      | `Vc`              | Rate at which distance shrinks      | m/s   | The component of relative velocity directly toward the target.   |

In proportional navigation, the commanded lateral acceleration is:
a_lat = N * Vc * λ̇

Don’t worry, you don’t need to calculate it.
The simulation does this automatically at every time step.

## 🌬️ 5. Drag and Dynamics

Air resists motion. FENRIR models this with a quadratic drag law, meaning drag grows with the square of velocity:

F_drag = -½ * ρ * Cd * A * |v| * v

|  Symbol | Meaning                             |
| :-----: | :---------------------------------- |
| ρ (rho) | Air density (kg/m³)                 |
|    Cd   | Drag coefficient (depends on shape) |
|    A    | Cross-sectional area (m²)           |
|    v    | Velocity relative to air            |

In simple terms:
the faster you move, the stronger the drag pulling back.

The missile’s BasicMissileDynamics class then combines this with its lateral acceleration command, updating its velocity and position accordingly.

## 🧭 6. The Radar View
The “Radar — Missile-Centric” panel is a stylised visual inspired by aircraft scopes.

The green sweep rotates continuously, simulating a radar dish.

-The rings represent increasing range (e.g. 1 km, 2 km, 3 km, ...).
-The blip shows the target relative to the missile’s position.
-Fading dots are echoes, remnants of where the sweep recently passed the target.
-No actual radar physics are simulated yet, it’s a visual metaphor for situational awareness.

## 📊 7. The Charts

Below the main plots, FENRIR charts telemetry values over time:
-Range vs. Time: how the distance closes until interception.
-LOS Rate vs. Time: how the line-of-sight rate changes as the missile adjusts course.
-(Optional) Commanded vs. Achieved Acceleration: a measure of performance realism.

All of this data can be exported to CSV for further analysis or plotting elsewhere.

## 🧮 8. Units and Abbreviations
| Symbol | Name                      | Unit  | In Words                        |
| :----: | :------------------------ | :---- | :------------------------------ |
|   `m`  | metre                     | —     | unit of length                  |
|   `s`  | second                    | —     | unit of time                    |
|  `kg`  | kilogram                  | —     | unit of mass                    |
|  `m/s` | metres per second         | —     | speed                           |
| `m/s²` | metres per second squared | —     | acceleration                    |
|  `λ̇`  | lambda dot                | rad/s | line-of-sight rate              |
|   `N`  | navigation constant       | —     | gain in proportional navigation |
|  `Vc`  | closing velocity          | m/s   | approach rate toward target     |
|   `R`  | range                     | m     | distance to target              |

## 🧩 9. Limitations (for now)
To keep things simple, FENRIR does not yet model:
-Three-dimensional motion (2D only)
-Gravity or lift
-Seeker errors or lag
-Control saturation and thrust curves
-Terrain or obstacles

Despite this, the existing model already captures the essence of guidance dynamics used in modern missiles and robotics, and can be extended later.

## 🐺 10. In Essence
FENRIR is not about warfare.
It’s about motion, the physics of pursuit, prediction, and response.
It’s a playground for anyone curious about how guidance systems think and react, rendered as clean visual dynamics on your screen.

## 📚 Further Reading

If you’d like to dive deeper into the ideas behind FENRIR, here are some beginner-friendly resources:
- Missile Guidance for Beginners – Defense Acquisition University (US) – a short overview explaining proportional navigation in plain terms.
-“Proportional Navigation Made Simple” by P. Zarchan – an excellent chapter from Tactical and Strategic Missile Guidance, written clearly even for non-engineers.
-“How Do Missiles Lock On?” – MinutePhysics video on YouTube, showing how geometry and timing drive interception.
-“The Pursuit Curve” – Mathematical article on how pursuers (dogs, insects, robots) naturally trace curved paths when chasing a target.
-NASA Glenn Research Center – Basics of Flight – concise explanations of drag, lift, and acceleration.

You don’t need a background in maths to enjoy these.
They show how the same geometry that guides a missile also governs the flight of a hawk, the path of a robot vacuum, or even the strategy of a football defender.
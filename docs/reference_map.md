# Scientific reference map

This document is the public citation-to-implementation authority for the
fixed-20-kg Loaded CMJ system-identification release. The citation keys shown
throughout are the exact keys in [`references.bib`](references.bib).

The committed MJCF, plant, schema, preprocessing contract, and test evidence
are the authority for this repository's model. No retained external source is
claimed as the source of the human-model topology, masses, inertias, actuator
constants, contact constants, measurement imperfections, parameter bounds, or
the 20 kg condition. The external references below document methods,
measurement interpretation, plausibility checks, and software/assets.

## Reference-to-code map

| Reference | Role | Supported component | Implementation location | Project evidence | Confidence |
| --- | --- | --- | --- | --- | --- |
| `REF-NW` `@nocedal_wright_2006` | `METHODS_BACKGROUND` | Bounded optimization, feasibility, stopping and residual discipline | `src/loaded_cmj/identification.py`; `docs/identification.md` | Explicit methods binding; current bounded least-squares implementation | HIGH |
| `REF-LP` `@lynch_park_2017` | `METHODS_BACKGROUND` | Wrenches, frames, constrained multibody/contact reasoning | `src/loaded_cmj/clean_core.py`; `src/loaded_cmj/plant.py`; `docs/mechanics.md` | Explicit mechanics-method binding; world-frame contact-wrench reconstruction | HIGH |
| `REF-T` `@tedrake_underactuated_2024` | `METHODS_BACKGROUND` | Floating-base underactuation and hybrid contact modes | `src/loaded_cmj/plant.py`; `src/loaded_cmj/validation.py` | Explicit mechanics-method binding; phase/contact guards in the target | HIGH |
| `REF-F` `@featherstone_2008` | `METHODS_BACKGROUND` | Rigid-body/spatial-dynamics and contact numerical reasoning | `assets/loaded_cmj_model.xml`; `src/loaded_cmj/clean_core.py`; `docs/mechanics.md` | Explicit rigid-body-method binding; compiled MuJoCo model remains first-party authority | HIGH |
| `REF-SH` `@shabana_2010` | `METHODS_BACKGROUND` | Constrained multibody dynamics and integration-error separation | `src/loaded_cmj/model.py`; `docs/reproducibility.md` | Explicit numerical-method binding; model-integrity checks | HIGH |
| `REF-HKD` `@hamill_knutzen_derrick_2015` | `MEASUREMENT_METHOD_SOURCE` | GRF sign, impulse-momentum and COM/landing interpretation | `src/loaded_cmj/clean_core.py`; `src/loaded_cmj/plant.py`; `docs/measurements.md` | Explicit measurement-method binding; no numerical plant values taken | HIGH |
| `REF-ZK` `@zatsiorsky_kinematics_1998` | `METHODS_BACKGROUND` | Sampled motion, frames and differential-kinematics interpretation | `src/loaded_cmj/preprocessing.py`; `docs/measurements.md` | Explicit measurement-method binding; bar velocity remains a declared finite difference | HIGH |
| `REF-ZT` `@zatsiorsky_kinetics_2002` | `MEASUREMENT_METHOD_SOURCE` | External/internal force separation, force platforms and impulse | `src/loaded_cmj/plant.py`; `src/loaded_cmj/clean_core.py`; `docs/mechanics.md` | Explicit measurement-method binding; bilateral aggregation is first-party | HIGH |
| `REF-K` `@komi_2003` | `VALIDATION_SOURCE` | CMJ, landing, SSC and force-transient plausibility checks | `src/loaded_cmj/validation.py`; `docs/validation.md` | Explicit validation binding; synthetic mechanics only, not human validation | HIGH |
| `REF-KH` `@khalil_2002` | `METHODS_BACKGROUND` | Equilibrium, stability and perturbation interpretation | `src/loaded_cmj/validation.py`; `docs/validation.md` | Explicit stability-method binding; thresholds remain repository-defined | HIGH |
| `REF-AM` `@astrom_murray_2008` | `METHODS_BACKGROUND` | Separation of plant, controller, actuator and sensor roles | `src/loaded_cmj/plant.py`; `src/loaded_cmj/preprocessing.py`; `docs/methodology.md` | Explicit clock/measurement-method binding; no controller constants derived | HIGH |
| `REF-SS` `@sarkka_svensson_2023` | `METHODS_BACKGROUND` | Continuous dynamics versus sampled measurement routes | `src/loaded_cmj/preprocessing.py`; `docs/measurements.md` | Explicit state/measurement-method binding; no Bayesian filter is claimed | HIGH |
| `REF-LJ` `@ljung_1999` | `METHODS_BACKGROUND` | Model structure, experiment design, residuals and held-out validation | `src/loaded_cmj/identification.py`; `src/loaded_cmj/validation.py`; `docs/identification.md` | Explicit system-identification binding; current solver is project-specific | HIGH |
| `REF-BE` `@beck_1979` | `METHODS_BACKGROUND` | Structural hypotheses and verification before parameter fitting | `src/loaded_cmj/identification.py`; `docs/identification.md` | Explicit model-structure binding; no Beck parameter values used | HIGH |
| `REF-BR` `@brogliato_1996` | `METHODS_BACKGROUND` | Unilateral contact, impact modes and nonsmooth-event hazards | `src/loaded_cmj/plant.py`; `src/loaded_cmj/validation.py`; `docs/mechanics.md` | Explicit contact-method binding; MuJoCo soft-contact constants are first-party | HIGH |
| `REF-MJ` `@todorov_erez_tassa_2012` | `SOFTWARE_ASSET_REFERENCE` | MuJoCo rigid-body dynamics, contacts and stepping | `assets/loaded_cmj_model.xml`; `src/loaded_cmj/plant.py`; `src/loaded_cmj/clean_core.py` | Direct `mujoco` imports, `mj_step`, and `mj_contactForce` use | HIGH |
| `REF-NP` `@harris_numpy_2020` | `SOFTWARE_ASSET_REFERENCE` | Numerical arrays and deterministic numerical utilities | `src/loaded_cmj/plant.py`; `src/loaded_cmj/identification.py`; `tools/` | Direct NumPy imports and array operations | HIGH |
| `REF-SP` `@virtanen_scipy_2020` | `SOFTWARE_ASSET_REFERENCE` | Nonlinear least-squares optimization dependency | `src/loaded_cmj/identification.py` | Direct `scipy.optimize.least_squares` import and use | HIGH |
| `REF-MPL` `@hunter_matplotlib_2007` | `SOFTWARE_ASSET_REFERENCE` | Publication plots and media-suite analysis figures | `examples/plot_*.py`; `examples/generate_media_suite.py` | Direct Matplotlib imports in the checked-in plot/media tooling | HIGH |
| `REF-MH` `@makehuman_community_2026` | `SOFTWARE_ASSET_REFERENCE` | Render-only MakeHuman visual asset and license pathway | `assets/makehuman_cmj_visual/`; `src/loaded_cmj/rendering.py` | `PROVENANCE.md`, `LICENSE_CC0.txt`, `MODIFICATIONS.md`, and renderer asset loading | HIGH |
| `REF-MHP` `@briceno_paul_2019` | `GENERAL_BACKGROUND` | MakeHuman framework acknowledgment only | `assets/makehuman_cmj_visual/LICENSE_CC0.txt`; `docs/provenance.md` | Committed asset-citation record; no physics or anthropometry claim | HIGH |

## Traceability records

Each retained source has one primary role. `IMPLEMENTATION_RELATION` describes
how the source relates to the release; it does not imply that source-derived
quantitative values entered the plant.

### Methods background

- `REF-NW` — **FULL_CITATION:** Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd ed., Springer, New York, 2006. **DOI:** `10.1007/978-0-387-40065-5`. **AUTHORITATIVE_URL:** [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** bounded system identification and feasibility checks. **CLAIM_OR_METHOD_SUPPORTED:** constrained optimization and residual/stopping discipline. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/identification.py`, `docs/identification.md`. **DEVELOPMENT_EVIDENCE:** explicit methods-source binding in the mechanics-first verification record and matching committed optimizer structure. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-LP` — **FULL_CITATION:** Kevin M. Lynch and Frank C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017. **DOI:** `10.1017/9781316661239`. **AUTHORITATIVE_URL:** [Cambridge University Press](https://www.cambridge.org/core/books/modern-robotics/57C3BB1C6D5CB40320FA96E5FA3BCEC6). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** contact frames and wrenches. **CLAIM_OR_METHOD_SUPPORTED:** frame/wrench and constrained-dynamics interpretation. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/clean_core.py`, `src/loaded_cmj/plant.py`. **DEVELOPMENT_EVIDENCE:** explicit contact-mechanics source binding; world-frame rotation is verified in the committed contact collector. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-T` — **FULL_CITATION:** Russ Tedrake, *Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation*, MIT 6.832 course notes, 2024. **DOI:** none. **AUTHORITATIVE_URL:** [MIT course notes](https://underactuated.csail.mit.edu/). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** floating-base and hybrid phase interpretation. **CLAIM_OR_METHOD_SUPPORTED:** underactuation and contact-mode/guard reasoning. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/plant.py`, `src/loaded_cmj/validation.py`. **DEVELOPMENT_EVIDENCE:** explicit hybrid-mechanics source binding; event guards remain code-defined. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-F` — **FULL_CITATION:** Roy Featherstone, *Rigid Body Dynamics Algorithms*, 1st ed., Springer, New York, 2008. **DOI:** `10.1007/978-1-4899-7560-7`. **AUTHORITATIVE_URL:** [Springer](https://link.springer.com/book/10.1007/978-1-4899-7560-7). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** rigid-body and contact-dynamics reasoning. **CLAIM_OR_METHOD_SUPPORTED:** spatial rigid-body dynamics and numerical-error separation. **IMPLEMENTATION_LOCATION:** `assets/loaded_cmj_model.xml`, `src/loaded_cmj/clean_core.py`. **DEVELOPMENT_EVIDENCE:** explicit source binding; current publisher metadata corrected the newer bundle's erroneous `978-0-387-74315-8` DOI. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-SH` — **FULL_CITATION:** Ahmed A. Shabana, *Computational Dynamics*, 3rd ed., John Wiley and Sons, 2010. **DOI:** `10.1002/9780470686850`. **AUTHORITATIVE_URL:** DOI resolver `https://doi.org/10.1002/9780470686850`. **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** constrained multibody and numerical integration checks. **CLAIM_OR_METHOD_SUPPORTED:** distinction between model equations, constraints, and integration error. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/model.py`, `docs/reproducibility.md`. **DEVELOPMENT_EVIDENCE:** explicit numerical-method binding and compiled-model integrity evidence. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-KH` — **FULL_CITATION:** Hassan K. Khalil, *Nonlinear Systems*, 3rd ed., Prentice Hall, 2002. **DOI:** none. **AUTHORITATIVE_URL:** [author's edition page](https://www.egr.msu.edu/~khalil/NonlinearSystems/index.html). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** stability and recovery interpretation. **CLAIM_OR_METHOD_SUPPORTED:** equilibrium/stability and perturbation language. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/validation.py`, `docs/validation.md`. **DEVELOPMENT_EVIDENCE:** explicit stability-method binding; numeric thresholds are not sourced from the book. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-AM` — **FULL_CITATION:** Karl J. Åström and Richard M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, 2008. **DOI:** none used. **AUTHORITATIVE_URL:** [Caltech Authors](https://authors.library.caltech.edu/records/yzs24-xsx88). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** plant/controller/measurement separation. **CLAIM_OR_METHOD_SUPPORTED:** distinct dynamical and sampled-observation roles. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/plant.py`, `src/loaded_cmj/preprocessing.py`. **DEVELOPMENT_EVIDENCE:** explicit clock-route binding; control and sensor constants remain first-party. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-SS` — **FULL_CITATION:** Simo Särkkä and Lennart Svensson, *Bayesian Filtering and Smoothing*, 2nd ed., Cambridge University Press, 2023. **DOI:** `10.1017/9781108917407`. **AUTHORITATIVE_URL:** [Cambridge University Press](https://www.cambridge.org/core/books/bayesian-filtering-and-smoothing/F88740E8D25010CF3119A5CA379FA37A). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** continuous plant and sampled measurement boundary. **CLAIM_OR_METHOD_SUPPORTED:** state/measurement route separation; no Bayesian filter is claimed. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/preprocessing.py`, `docs/measurements.md`. **DEVELOPMENT_EVIDENCE:** explicit state/measurement-method binding. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-LJ` — **FULL_CITATION:** Lennart Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall PTR, Upper Saddle River, NJ, 1999. **DOI:** none. **AUTHORITATIVE_URL:** [Ljung's book page](https://rt.isy.liu.se/en/books/sysid/). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** experiment design, residuals, identifiability, and held-out validation. **CLAIM_OR_METHOD_SUPPORTED:** general system-identification workflow. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/identification.py`, `src/loaded_cmj/validation.py`. **DEVELOPMENT_EVIDENCE:** explicit system-identification source binding; current objective and bounds are repository-specific. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-BE` — **FULL_CITATION:** M. B. Beck, “Model Structure Identification from Experimental Data,” in E. Halfon (ed.), *Theoretical Systems Ecology: Advances and Case Studies*, Academic Press, London, 1979, pp. 259–289. **DOI:** none. **AUTHORITATIVE_URL:** [IIASA record](https://pure.iiasa.ac.at/id/eprint/1046/). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** structural model choice before fitting. **CLAIM_OR_METHOD_SUPPORTED:** competing hypotheses and verification/validation before parameter estimation. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/identification.py`, `docs/identification.md`. **DEVELOPMENT_EVIDENCE:** explicit source binding; the current 27-coordinate schema is first-party. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-BR` — **FULL_CITATION:** Bernard Brogliato, *Nonsmooth Impact Mechanics: Models, Dynamics and Control*, Lecture Notes in Control and Information Sciences 220, 1st ed., Springer, 1996. **DOI:** `10.1007/BFb0027733`. **AUTHORITATIVE_URL:** [Springer](https://link.springer.com/book/10.1007/BFb0027733). **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** unilateral contact, impact and event semantics. **CLAIM_OR_METHOD_SUPPORTED:** contact-mode and nonsmooth numerical reasoning. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/plant.py`, `src/loaded_cmj/validation.py`. **DEVELOPMENT_EVIDENCE:** explicit contact-method binding; MuJoCo's soft-contact law is not presented as a literature-derived law. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.

### Measurement-method sources

- `REF-HKD` — **FULL_CITATION:** Joseph Hamill, Kathleen M. Knutzen, and Timothy R. Derrick, *Biomechanical Basis of Human Movement*, 4th ed., Wolters Kluwer, 2015. **DOI:** none located. **AUTHORITATIVE_URL:** ISBN `978-1-4511-7730-5` publisher record. **ROLE:** `MEASUREMENT_METHOD_SOURCE`. **MODEL_COMPONENT:** force-platform/GRF and impulse interpretation. **CLAIM_OR_METHOD_SUPPORTED:** sign, action-reaction, impulse-momentum, COM and landing interpretation. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/clean_core.py`, `src/loaded_cmj/plant.py`, `docs/measurements.md`. **DEVELOPMENT_EVIDENCE:** explicit measurement-method binding; force scale/bias/noise remain synthetic. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-ZT` — **FULL_CITATION:** Vladimir M. Zatsiorsky, *Kinetics of Human Motion*, Human Kinetics, Champaign, IL, 2002. **DOI:** none located. **AUTHORITATIVE_URL:** ISBN `978-0-7360-3778-5` publisher record. **ROLE:** `MEASUREMENT_METHOD_SOURCE`. **MODEL_COMPONENT:** external/internal force separation, bilateral force and impulse. **CLAIM_OR_METHOD_SUPPORTED:** force-platform and support-mechanics interpretation. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/plant.py`, `src/loaded_cmj/clean_core.py`, `docs/mechanics.md`. **DEVELOPMENT_EVIDENCE:** explicit measurement-method binding; bilateral contact aggregation is first-party. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.
- `REF-ZK` — **FULL_CITATION:** Vladimir M. Zatsiorsky, *Kinematics of Human Motion*, Human Kinetics, Champaign, IL, 1998. **DOI:** none located. **AUTHORITATIVE_URL:** ISBN `978-0-88011-676-3` publisher record. **ROLE:** `METHODS_BACKGROUND`. **MODEL_COMPONENT:** sampled bar kinematics and finite differences. **CLAIM_OR_METHOD_SUPPORTED:** coordinate/frame and sampled-motion interpretation. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/preprocessing.py`, `docs/measurements.md`. **DEVELOPMENT_EVIDENCE:** explicit source binding; the 100 Hz grid and difference convention are repository-defined. **IMPLEMENTATION_RELATION:** `METHOD_INFORMED`. **CONFIDENCE:** HIGH.

### Validation sources

- `REF-K` — **FULL_CITATION:** Paavo V. Komi (ed.), *Strength and Power in Sport*, 2nd ed., Blackwell Science, 2003. **DOI:** `10.1002/9780470757215`. **AUTHORITATIVE_URL:** DOI resolver `https://doi.org/10.1002/9780470757215`. **ROLE:** `VALIDATION_SOURCE`. **MODEL_COMPONENT:** CMJ/landing/SSC plausibility interpretation. **CLAIM_OR_METHOD_SUPPORTED:** literature-informed mechanical plausibility checks for force transients and landing/SSC language. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/validation.py`, `docs/validation.md`. **DEVELOPMENT_EVIDENCE:** explicit validation-source binding; no human dataset was used. **IMPLEMENTATION_RELATION:** `VALIDATION_ONLY`. **CONFIDENCE:** HIGH.

### Software and assets

- `REF-MJ` — **FULL_CITATION:** Emanuel Todorov, Tom Erez, and Yuval Tassa, “MuJoCo: A Physics Engine for Model-Based Control,” *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2012, pp. 5026–5033. **DOI:** `10.1109/IROS.2012.6386109`. **AUTHORITATIVE_URL:** [IEEE DOI](https://doi.org/10.1109/IROS.2012.6386109). **ROLE:** `SOFTWARE_ASSET_REFERENCE`. **MODEL_COMPONENT:** MuJoCo runtime and contact/step API. **CLAIM_OR_METHOD_SUPPORTED:** software dependency only. **IMPLEMENTATION_LOCATION:** `assets/loaded_cmj_model.xml`, `src/loaded_cmj/plant.py`, `src/loaded_cmj/clean_core.py`. **DEVELOPMENT_EVIDENCE:** direct imports and runtime calls in the committed target. **IMPLEMENTATION_RELATION:** `SOFTWARE_DEPENDENCY`. **CONFIDENCE:** HIGH.
- `REF-NP` — **FULL_CITATION:** Charles R. Harris et al., “Array Programming with NumPy,” *Nature*, 585(7825), 357–362, 2020. Full author list is retained in `references.bib`. **DOI:** `10.1038/s41586-020-2649-2`. **AUTHORITATIVE_URL:** [Nature DOI](https://doi.org/10.1038/s41586-020-2649-2). **ROLE:** `SOFTWARE_ASSET_REFERENCE`. **MODEL_COMPONENT:** array and numerical utilities. **CLAIM_OR_METHOD_SUPPORTED:** software dependency only. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/`, `tools/`, `tests/`. **DEVELOPMENT_EVIDENCE:** direct NumPy imports and array operations. **IMPLEMENTATION_RELATION:** `SOFTWARE_DEPENDENCY`. **CONFIDENCE:** HIGH.
- `REF-SP` — **FULL_CITATION:** Pauli Virtanen et al., “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python,” *Nature Methods*, 17(3), 261–272, 2020. The named author list and SciPy contributor group are retained in `references.bib`. **DOI:** `10.1038/s41592-019-0686-2`. **AUTHORITATIVE_URL:** [Nature Methods DOI](https://doi.org/10.1038/s41592-019-0686-2). **ROLE:** `SOFTWARE_ASSET_REFERENCE`. **MODEL_COMPONENT:** bounded least-squares implementation. **CLAIM_OR_METHOD_SUPPORTED:** software dependency only. **IMPLEMENTATION_LOCATION:** `src/loaded_cmj/identification.py`. **DEVELOPMENT_EVIDENCE:** direct `scipy.optimize.least_squares` import and use in the current target. **IMPLEMENTATION_RELATION:** `SOFTWARE_DEPENDENCY`. **CONFIDENCE:** HIGH.
- `REF-MPL` — **FULL_CITATION:** John D. Hunter, “Matplotlib: A 2D Graphics Environment,” *Computing in Science and Engineering*, 9(3), 90–95, 2007. **DOI:** `10.1109/MCSE.2007.55`. **AUTHORITATIVE_URL:** [IEEE DOI](https://doi.org/10.1109/MCSE.2007.55). **ROLE:** `SOFTWARE_ASSET_REFERENCE`. **MODEL_COMPONENT:** checked-in scientific plot and media tooling. **CLAIM_OR_METHOD_SUPPORTED:** plotting software dependency only. **IMPLEMENTATION_LOCATION:** `examples/plot_observables.py`, `examples/plot_bilateral_asymmetry.py`, `examples/generate_media_suite.py`. **DEVELOPMENT_EVIDENCE:** direct Matplotlib imports in the publication-analysis surface. **IMPLEMENTATION_RELATION:** `SOFTWARE_DEPENDENCY`. **CONFIDENCE:** HIGH.
- `REF-MH` — **FULL_CITATION:** MakeHuman Community, *MakeHuman Community: Project, Asset, and License Documentation*, official project/licensing documentation, accessed 2026-08-08. **DOI:** none. **AUTHORITATIVE_URL:** [official license guidance](https://static.makehumancommunity.org/about/license.html) and [asset-pack catalog](https://static.makehumancommunity.org/assets/assetpacks/index.html). **ROLE:** `SOFTWARE_ASSET_REFERENCE`. **MODEL_COMPONENT:** render-only visual asset and license boundary. **CLAIM_OR_METHOD_SUPPORTED:** asset provenance and licensing, not physics or anthropometry. **IMPLEMENTATION_LOCATION:** `assets/makehuman_cmj_visual/`, `src/loaded_cmj/rendering.py`. **DEVELOPMENT_EVIDENCE:** tracked asset inventory, license note, provenance, modifications, and renderer-only loading. **IMPLEMENTATION_RELATION:** `ASSET_DEPENDENCY`. **CONFIDENCE:** HIGH.

### General background

- `REF-MHP` — **FULL_CITATION:** Leyde Briceno and Gunther Paul, “MakeHuman: A Review of the Modelling Framework,” in *Proceedings of the 20th Congress of the International Ergonomics Association (IEA 2018)*, *Advances in Intelligent Systems and Computing*, vol. 822, Springer, Cham, 2019, pp. 224–232. **DOI:** `10.1007/978-3-319-96077-7_23`. **AUTHORITATIVE_URL:** [Springer](https://doi.org/10.1007/978-3-319-96077-7_23). **ROLE:** `GENERAL_BACKGROUND`. **MODEL_COMPONENT:** MakeHuman asset acknowledgment. **CLAIM_OR_METHOD_SUPPORTED:** framework background only; it does not support anthropometric or physics parameters. **IMPLEMENTATION_LOCATION:** `assets/makehuman_cmj_visual/LICENSE_CC0.txt`, `docs/provenance.md`. **DEVELOPMENT_EVIDENCE:** committed asset-citation/licensing record. **IMPLEMENTATION_RELATION:** `BACKGROUND_ONLY`. **CONFIDENCE:** HIGH.

## Direct model basis

The direct model basis is first-party and is not a literature citation:

- `MODEL-LOCAL-01` — `assets/loaded_cmj_model.xml`, `src/loaded_cmj/plant.py`,
  `src/loaded_cmj/clean_core.py`, `src/loaded_cmj/model.py`, and the committed
  model-integrity tests. **ROLE:** `DIRECT_MODEL_BASIS`.
  **IMPLEMENTATION_RELATION:** `DIRECTLY_IMPLEMENTED`. **CONFIDENCE:** HIGH.
  These files define the linked sagittal topology, bilateral lower limbs,
  hindfoot/forefoot/toe structure, actuators, contact pairs, bar/rack, solver,
  timestep, and telemetry semantics.

There is no retained external `DIRECT_MODEL_BASIS` and no retained external
`PARAMETER_SOURCE`. `configs/param_schema.json` and the named configurations
are first-party parameter authorities, not literature-derived subject data.

## Parameter sources

No quantitative literature parameter source was established. Segment geometry,
COMs, inertias, reference mass, joint ranges, actuator gains and time
constants, contact/friction settings, bar/rack compliance, sensor calibration,
noise, event thresholds, and identification bounds are repository-defined.

## Measurement-method sources

Hamill/Knutzen/Derrick and Zatsiorsky's *Kinetics* support the interpretation of
force-platform signals, bilateral force, impulse, and COM-related mechanics.
Zatsiorsky's *Kinematics* supports sampled-motion terminology. They do not
turn the synthetic force scale, bias, noise, or 100 Hz grid into experimental
instrument parameters. The LPT remains a bar-displacement measurement, never a
COM measurement.

## Validation sources

Komi supports literature-informed plausibility language for CMJ/SSC/landing
mechanics. Internal validation is deterministic, numerical, mechanics-based,
and held out within the same synthetic plant. It is not experimental human,
clinical, population, or subject-specific validation.

## Methods background

The remaining methods references inform the vocabulary and verification
discipline for rigid-body dynamics, contact modes, stability, sampled
measurements, optimization, model structure, and system identification. They
are not claims that the implementation reproduces a named textbook algorithm
or that a textbook supplied a numerical value.

## Software/assets

MuJoCo, NumPy, SciPy, and Matplotlib are materially used software dependencies
in the runtime or checked-in publication tooling. The MuJoCo software paper is
the canonical research citation; the [official MuJoCo
documentation](https://mujoco.readthedocs.io/) is the runtime documentation
reference and is not a second bibliography entry. MakeHuman is a separate,
render-only asset chain with its tracked CC0 notice, inventory, provenance, and
modification records. The MakeHuman asset does not affect masses, inertias,
contacts, COM, CoP, force-platform channels, LPT channels, datasets, or scoring.

## First-party engineering assumptions

The following 51-item inventory is the explicit unreferenced-assumption set
carried into this release. Each item is classified as `FIRST_PARTY_ENGINEERING_CHOICE`,
`SYNTHETIC_DATA_DESIGN`, `NUMERICAL_CHOICE`, `CALIBRATED_VALUE`, or
`VALIDATION_CONSTRAINED_CHOICE`; none is presented as a quantitative literature
source. The frozen `alpha=0.02` is the validation-constrained refinement of
item 15.

| Group | Items and classification |
| --- | --- |
| Human/multibody (8) | Linked sagittal simplification; torso/pelvis representation and segmentation; bilateral hip/knee/ankle topology; one-DOF sagittal hinge axes/ranges; hindfoot/forefoot/toe segmentation; passive forefoot rocker/MTP stiffness/damping; geometry/COM/inertia layout; 75 kg reference and mass/inertia scaling — all `FIRST_PARTY_ENGINEERING_CHOICE`. |
| Loaded CMJ (7) | Phase schedule boundaries; loaded-CMJ protocol distinction; countermovement depth scaling; braking/propulsion pose/gain sequencing; 25 ms sustained no-contact takeoff; landing/stabilization/rebound/recovery thresholds — `FIRST_PARTY_ENGINEERING_CHOICE`. Nominal 20 kg load and fixed-load defaults — `SYNTHETIC_DATA_DESIGN`. The selected `alpha=0.02` is a `VALIDATION_CONSTRAINED_CHOICE`, not a physiological or clinical threshold. |
| Actuation (6) | Six torque-actuator abstraction; PD tracking structure; phase feedforward; joint-stiffness bounds; joint-damping bounds; activation-delay/time-constant bounds — `FIRST_PARTY_ENGINEERING_CHOICE`. |
| Foot-ground contact (5) | Six eligible foot contact geoms; friction; contact stiffness/damping bounds; `solref`/`solimp`; contact eligibility and force thresholds — `FIRST_PARTY_ENGINEERING_CHOICE`. |
| External bar/load (4) | Compliant rack; rack translational stiffness/damping; rack pitch stiffness/damping; hand spatial-tendon/grip coupling — `FIRST_PARTY_ENGINEERING_CHOICE`. |
| Force plate (4) | Bilateral aggregation; scale/bias bounds; noise distribution/magnitude; comparison/sample grid — aggregation, scale/bias and grid are `FIRST_PARTY_ENGINEERING_CHOICE`; noise is `SYNTHETIC_DATA_DESIGN`. |
| LPT/bar measurement (5) | Bar displacement only; attachment offset bounds; encoder scale/delay/filter/offset bounds; first-order filter plus integer delay; tether stiffness/damping and zero path — `FIRST_PARTY_ENGINEERING_CHOICE`. |
| Preprocessing (5) | 100 Hz grid/interpolation; 1.5 s weighing and final 1.0 s baseline; onset threshold and 50 ms sustain; velocity differentiation; time alignment/resampling — `NUMERICAL_CHOICE` for grid/differentiation and `FIRST_PARTY_ENGINEERING_CHOICE` for task thresholds/windows. |
| System identification (4) | Current 27-scalar identification structure; parameter-bound envelope; residual composition/scaling/weighting; six-identification plus 32-validation synthetic design — structure, bounds and residuals are `FIRST_PARTY_ENGINEERING_CHOICE`; trial design is `SYNTHETIC_DATA_DESIGN`. |
| Numerical simulation (3) | 0.002 s production timestep and solver settings; settling procedure; synthetic seeds/distributions — timestep/solver and settling are `NUMERICAL_CHOICE`; seeds/distributions are `SYNTHETIC_DATA_DESIGN`. |

The current target is 27 scalar coordinates. A stale 24-parameter statement in
the newer reconstruction bundle was corrected from the committed schema and is
not used as a public claim.

Probabilistic Inference
=======================

This page documents what the inference layers claim, how they represent
uncertainty, and where they deliberately stop short.

Uncertainty semantics
---------------------

Uncertainty is never hidden at an intermediate stage. Every stage exposes:

* the full posterior, not only its argmax;
* the confidence attached to it;
* how much of the sensing apparatus was contributing;
* which sensors supported the conclusion and which contradicted it;
* which sensors supplied nothing at all.

And every stage can decline. ``UNKNOWN`` is an abstention, not an eighth
behaviour, and is excluded from the latent state set by construction.

The load-bearing rule
~~~~~~~~~~~~~~~~~~~~~

    A missing observation is missing evidence. It is never negative evidence
    about the resident.

This is enforced mechanically rather than by convention. Sensor reliability
enters the fusion likelihood as a tempering weight, so a sensor with
reliability zero contributes a *flat* likelihood and leaves the belief
entirely to other modalities. A dead sensor cannot look like a quiet resident,
because its contribution is identically zero rather than "observed nothing".

Sensor health
-------------

:class:`~sensor_modeling.health.SensorHealthMonitor` maintains a verdict per
sensor and emits it as an evidence weight in ``[0, 1]``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Status
     - Meaning
   * - ``HEALTHY``
     - Reporting as declared, with plausible values.
   * - ``DEGRADED``
     - Still reporting, with reduced measurement quality.
   * - ``DROPOUT`` / ``MISSING``
     - Silent long enough to suspect, then to assume, a fault.
   * - ``STUCK``
     - Unchanging value where variation is expected.
   * - ``DRIFTING``
     - Calibration has shifted relative to the sensor's own history.
   * - ``OUT_OF_RANGE``
     - Values outside the declared plausible range.
   * - ``UNKNOWN``
     - Not enough evidence to judge. The honest default, not a failure.

Three rules keep the verdicts conservative:

**Silence only means failure when the sensor promised to speak.** A sensor
that declares an ``expected_interval`` is expected on that cadence, so
prolonged silence is diagnostic. A purely event-driven contact sensor makes no
such promise, and its silence is genuinely ambiguous between "broken" and
"nobody opened the cupboard". For those sensors the monitor declines to call a
failure.

**Repetition is only suspicious where variation is expected.** The three
observation kinds repeat for different reasons. An event sensor reports the
same value on every activation, and a state sensor reports an unchanged level
for as long as the level is unchanged; neither is a fault. Only a *sampled*
sensor, which is supposed to track a varying quantity, is judged stuck by
consecutive identical readings. A state sensor is judged stuck only once its
level has persisted beyond any plausible real duration.

This distinction is load-bearing rather than pedantic. Treating a repeated
level as a fault flagged the bed sensor as stuck through every night of
sleep, discounting the strongest evidence for sleep precisely when it
mattered; correcting it raised clean-record balanced accuracy from 0.805 to
0.858.

**A silent deployment is not a silent resident.** Event sensors make no
promise to report, so their silence cannot normally be called a failure -- but
when every sensor that *did* promise has gone missing at once, the likely
explanation is that the pathway carrying all of them failed. Sensors with a
declared cadence therefore act as canaries: when enough of them fall silent,
event sensors silent over the same period are downgraded too, so their silence
stops being read as observed inactivity. Only silence counts as a canary
signal, since a stuck sensor is still delivering records.

**Drift is reported, never corrected.** Without redundant sensing the monitor
cannot separate sensor drift from genuine environmental change, so it flags
the shift and leaves the judgement to the analyst.

:class:`~sensor_modeling.health.SystemHealthReport` exposes deployment
integrity -- coverage and the faulty set -- entirely separately from any
behavioural conclusion.

Behavioural state ontology
--------------------------

:class:`~sensor_modeling.states.StateOntology` defines the latent states and
their dynamics. The default set is:

``away``, ``home_active``, ``home_inactive``, ``sleeping``, ``bed_awake``,
``bathroom_activity``, ``kitchen_activity``

The states are deliberately weaker than the activities of daily living a
clinician would name. ``kitchen_activity`` says the resident appears to be
active in the kitchen. It does not say they ate, because a contact sensor
cannot supply that evidence.

Transitions are modelled as a **continuous-time Markov chain**. Each state
declares a mean dwell time and a set of permitted destinations; the generator
follows, and the transition operator over any elapsed interval is its matrix
exponential. This is what makes asynchrony a non-issue: observations arriving
at irregular moments need no resampling onto a common grid.

The chain's stationary distribution is the default prior. Before any evidence,
the most defensible belief is what the declared dynamics say about the long
run.

Multimodal fusion
-----------------

:class:`~sensor_modeling.fusion.MultimodalBayesFilter` maintains
``P(Z_t | O_1:t)`` by forward filtering: predict with the transition operator
for the elapsed interval, then multiply in one tempered log-likelihood per
sensor.

Emission models
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Used for
   * - :class:`~sensor_modeling.fusion.PoissonEventEmission`
     - Event streams. Over an interval ``T`` with ``n`` activations, a state
       with rate ``lambda`` scores ``n log(lambda) - lambda T``.
   * - :class:`~sensor_modeling.fusion.GaussianEmission`
     - Sampled scalars such as wearable activity magnitude.
   * - :class:`~sensor_modeling.fusion.BernoulliEmission`
     - Persisting binary states such as bed occupancy. Uses only the most
       recent reading, since repeated reports of an unchanged state are not
       repeated independent measurements of it.
   * - :class:`~sensor_modeling.fusion.BetaEmission`
     - Probability-valued derived features such as a radar presence
       probability, kept on its natural support rather than thresholded.

The Poisson model is what makes silence behave correctly without a special
case. A quiet interval penalises high-rate states in proportion to how long
the silence lasted, so a *working but quiet* sensor is informative while a
*dead* one is not. :class:`~sensor_modeling.fusion.StateEstimate`
distinguishes the two explicitly through its ``silent`` and ``missing``
properties.

Tempering
~~~~~~~~~

Each raw log-likelihood is multiplied by ``weight x reliability x
attribution``. This is a power-likelihood weighting in the sense of
generalised Bayesian updating: it degrades smoothly rather than switching a
sensor abruptly on and off, and reliability zero contributes exactly nothing.

``weight`` additionally lets a deployment correct for sampling-rate imbalance
deliberately, rather than letting a fast-sampling wearable dominate simply by
reporting more often.

Defaults from declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~sensor_modeling.fusion.default_emissions` derives models from the
registry. Its rates factor two independent things: whether a state puts the
resident within range of a sensor, and how much they move while in that state.

Conflating those is a live hazard. In development, treating "in the bedroom"
as "generating bedroom motion" made a bedroom sensor's silence argue against
``sleeping`` exactly as hard as the bed sensor argued for it, collapsing
sleep into ``home_inactive`` 98% of the time. Separating location from
activity level raised end-to-end state accuracy from 0.585 to 0.918 and sleep
recall from 0.00 to 0.96.

Abstention
~~~~~~~~~~

An estimate reports ``UNKNOWN`` when the posterior is too flat
(``min_confidence``) or too little of the apparatus was contributing
(``min_completeness``). Because the prediction step relaxes toward the
stationary distribution when nothing is observed, confidence falls naturally
during an outage and the estimate abstains instead of coasting on a stale
conclusion.

Occupancy and attribution
-------------------------

Ambient sensors observe a home, not a person. Without this layer, a
daughter's Sunday visit reads as a sudden improvement in mobility and a
carer's morning round reads as the resident getting up early.

:class:`~sensor_modeling.context.ResidentContextEstimator` maintains a
posterior over four occupancy contexts -- ``empty``, ``resident_alone``,
``resident_with_visitor``, ``visitor_only`` -- and marginalises it into:

.. code-block:: text

    P(resident_home | O)
    P(visitor_present | O)
    P(multiple_people_present | O)
    P(activity from sensor s was the resident's | O)

The last is the attribution weight the fusion layer consumes. Sensors declared
``attributable`` keep a weight of one; every ambient sensor gets the marginal
probability.

Privacy
~~~~~~~

The evidence is deliberately anonymous. There are **no cameras, no
microphones, and no biometric identification** anywhere in the design. The aim
is uncertainty-aware attribution, not identity. What it uses:

* a personal presence beacon, with in-range probabilities below one, because a
  wearable left on the dresser is not a resident who left the house;
* radar or room-occupancy track counts, via a per-context Poisson mean;
* concurrent activation *events* in distinct rooms -- the strongest anonymous
  evidence of a second person, since nobody is in the kitchen and the
  bathroom at once;
* door crossings, which relax the belief toward a transition rather than
  voting for a context, because a door is the moment occupancy is most likely
  to change.

Presence samples carry an explicit correlation discount
(``sample_weight``). Successive readings mostly re-observe the same unchanged
situation, and treating them as independent would reach certainty within
minutes. The resulting posteriors are deliberately conservative.

Adaptive personal baseline
--------------------------

A baseline frozen at enrolment turns every seasonal shift into a permanent
alarm; one that adapts instantly turns a real decline into the new normal
before anyone notices. :mod:`sensor_modeling.baseline` treats behaviour as a
slowly moving distribution and distinguishes the reasons a day can look
unusual:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Verdict
     - Meaning
   * - ``ORDINARY``
     - Within the personal band. Not a finding.
   * - ``TEMPORARY_DISTURBANCE``
     - Unusual days that have already reverted.
   * - ``PERSISTENT_CHANGE``
     - A shift that has held past ``persistence_days``.
   * - ``GRADUAL_DRIFT``
     - A slow monotone trend rather than a step.
   * - ``ABRUPT_CHANGE``
     - A step located by PELT change-point detection.
   * - ``INSUFFICIENT_DATA``
     - Not enough well-observed days to say anything.

Daily features are aggregated from the **posterior**, as
``sum P(state | t) dt``, not by counting argmax wins, so a day of hesitant
guesses does not look like a day of certainties.

The reference is robust (median and MAD, so one extraordinary day cannot
redefine normal) and **weekday-aware** (Sundays are compared against Sundays,
so ordinary weekly rhythm is not reported as behavioural change). History is
bounded, so the definition of normal moves with the person.

Poorly observed days go through
:meth:`~sensor_modeling.baseline.AdaptiveBaseline.skip`, which records the
verdict but deliberately keeps the day **out** of the history. Letting a
sensor outage enter as a low value would allow the apparatus to rewrite the
resident's normal.

Alert semantics
---------------

Raising an alert is a claim on somebody's attention, and in this domain a
stream of false alarms is the failure mode that gets monitoring switched off
entirely. A change must therefore survive several filters:

* it must be a genuine change, not an ordinary day or a reverted disturbance;
* it must be large enough and have lasted long enough (graded jointly --
  neither alone suffices);
* it must have been observed with adequate sensor coverage;
* it must be attributable to the resident rather than to a visitor.

Partial coverage, partial attribution, and a not-yet-weekday-aware reference
are recorded as explicit **caveats** on the alert rather than left for the
reader to infer.

A gradual drift is graded on the trend that identified it rather than on a
deviation streak, which by construction it does not have. Grading it on the
same axes as a step would leave the slow decline -- the pattern that matters
most in ambient monitoring -- permanently unalertable.

Repeats of the same finding are suppressed for a cooldown unless severity has
escalated, and a burst beyond the configured rate is replaced by a single
notice that further alerts are being withheld, since a flood almost always
means a sensing or configuration problem rather than a sudden change in the
resident.

**System-health alerts are a separate kind** and carry an explicit disclaimer
that they concern the apparatus rather than the resident. A failing sensor
must never surface as a finding about a person.

Alert text is phrased as observation, never diagnosis. The platform is a
research toolkit and is not a medical device.

Interoperability
----------------

:mod:`sensor_modeling.interop` exports the pipeline's output in a FHIR-style
form that keeps the four kinds distinguishable on the way out.

The hazard is specific: once a behavioural conclusion is written into a
clinical record it looks like every other entry there, and a reader has no way
to tell that ``sleeping`` was inferred by a Markov filter rather than
measured. Exporting inferences as observations is how a research prototype
ends up quoted as a clinical fact.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Kind
     - Exported as
   * - Measured observation
     - ``Observation``, ``status: final``, provenance ``measured``.
   * - Derived feature
     - ``Observation``, ``status: final``, provenance ``derived-feature``,
       carrying the upstream device's own confidence.
   * - Inferred state
     - ``Observation``, ``status: preliminary``, provenance ``inferred``, an
       explicit ``method``, the **whole posterior** as components, and
       ``derivedFrom`` listing the contributing sensors.
   * - Algorithmic alert
     - ``DetectedIssue`` -- never an ``Observation``, because an alert is a
       judgement rather than a record of anything observed.

Every resource carries a provenance extension recording how it was produced.
:func:`~sensor_modeling.interop.summarise_provenance` counts them, and
:func:`~sensor_modeling.interop.measured_only` returns just the genuine
measurements, so a consumer can assert in one line that it has not been handed
inferences dressed as measurements.

Two further protections:

* An **abstaining** estimate exports a ``dataAbsentReason`` rather than a
  value, which is the correct idiom for "we do not know" and stops ``unknown``
  from being read as a behavioural finding.
* A **system-health** alert is not attached to a patient at all. A failing
  sensor is equipment maintenance, and attaching it to a person would make it
  look like a clinical finding.

Event-kind observations carry an explicit note that absence of a record is
absence of evidence rather than an observation of zero, so a consumer cannot
reasonably fill the gaps with zeros.

.. warning::

   This is a FHIR-*style* export for interoperability prototyping. It is not a
   validated FHIR profile, has not been conformance-tested against a FHIR
   server, and its codes come from a project-local code system rather than
   LOINC or SNOMED. Its output must not be presented as clinically validated
   data.

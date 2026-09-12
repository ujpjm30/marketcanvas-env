# MarketCanvas-Env

A deterministic 2D design canvas exposing a Gymnasium interface and an MCP
server, built so an agent can be trained to produce marketing assets from a
natural language brief.

## 1. Action and State Design

### Two action layers instead of one

The brief allows either action space. This environment implements both,
because they pull in opposite directions and that tension is the interesting
part.

The deployment target is computer use. An agent operating real design
software has a mouse and a keyboard and nothing else, so the low-level layer
is the distribution that actually matters. Training only on semantic calls
would optimize for an interface the agent will never see.

Learning favours the opposite. Placing one element costs a single
`add_element` call, or three low-level steps, which are selecting a tool,
dragging a box, and typing a label. Tripling the horizon triples the
distance between an action and the terminal reward it contributed to, and
credit assignment degrades with that distance.

Keeping both layers in one environment makes the trade-off measurable rather
than theoretical. Both funnel into the same `Canvas` mutations, so a drag
and a `move_element` cannot disagree about what moving means, and
`ActionHandler` takes an `allow_low_level` flag so either layer can be
ablated on its own.

### The low-level layer was incomplete as specified

Implemented exactly as the brief lists it, with move, click, drag and type,
the low-level layer cannot finish an episode. All four actions worked, but
every one of them operates on an element that already exists and nothing in
the set creates anything. From a blank canvas a drag finds nothing to grab
and typing finds nothing selected, so the layer was only usable alongside a
high-level `add_element`.

The missing piece is modal state. Nobody draws a rectangle in Canva by
dragging on empty space. They pick a tool from the palette first, and the
drag means "draw" only because a tool is active. `select_tool` adds that
mode and carries the current role and colors the way a real toolbar carries
the current style. A drag draws whenever a tool is active and moves when one
is not.

`test_low_level_alone_can_finish_the_task` runs a blank canvas to a scoring
banner without a single semantic action.

One implementation detail was wrong on the first attempt and is worth
recording. Drawing was initially triggered by dragging from an empty point,
which works until a background is laid, after which no empty point exists.
The drag intended to draw a headline moved the background instead. Draw mode
has to be modal, not contextual.

### Semantic state, with pixels available on request

The observation is JSON. The rendered RGB array exists but is not on the
step path.

Raw coordinates are not enough on their own. Questions an agent needs every
step, such as whether two elements overlap, whether something is centered,
or whether an element has fallen off the canvas, are arithmetic the model
would otherwise redo each turn, badly and at token cost. `observation.py`
precomputes them, giving pairwise direction, overlap ratio, which element is
occluded, alignment on the center line and on shared edges, and for anything
carrying text, the color actually behind it.

Each unordered pair is reported once. Reporting both directions doubles the
tokens to say the same thing twice.

### What the observation deliberately leaves out

The current reward and the scoring weights are not in the observation.

The brief reaches the agent as natural language in the prompt, which is the
same information a person would get. Exposing the scalar would replace the
question "what makes a good banner" with "which edit raises the number", and
a policy trained that way optimizes the metric rather than the design. It
also becomes worthless the moment the weights change. `get_current_reward`
exists as an MCP tool for inspection and debugging, not as part of the
state.

### Determinism

`z_index` is assigned by the canvas, not the caller. Letting an agent choose
it permits ties, and a tie has no defined winner, which changes which
element counts as the background behind a piece of text and therefore
changes the contrast score. The brief asks for a deterministically simulated
canvas, so the same action sequence has to produce the same state and the
same reward. Element ids are assigned for the same reason, and `clear()`
resets both counters so a second episode is indistinguishable from the
first.

Rejected actions still consume a step. Free retries would let a policy
brute-force the action schema instead of learning it.

## 2. The Reward Function

### How it works

The reward is terminal and lands in the range -1.0 to 1.0. Design quality is
a property of the finished artifact, so scoring every step would pay an
agent for churn.

Three scorers each answer one question and return a value between 0 and 1.

| Component | Weight | Question |
|---|---|---|
| Constraints | 0.6 | Is what the brief asked for actually there? |
| Contrast | 0.2 | Can the text be read? |
| Layout | 0.2 | Is anything colliding, off-canvas, or unaligned? |

Constraints carry the most weight because a banner missing its CTA has
failed regardless of how clean the layout is. The weighted sum lands between
0 and 1 and is rescaled by `2q - 1`, so a canvas that satisfies nothing
reaches the same floor as an empty one.

Each scorer is a class behind a shared interface and `RewardFunction` takes
the weight list as a constructor argument, so an ablation means passing
different weights rather than editing the scoring code.

Three decisions inside are worth naming.

**Partial credit on constraints.** Two of two required elements scores 1.0
and one of two scores 0.5. Demanding all of them before any reward leaves a
fresh policy with no gradient to follow.

**Contrast is ramped and takes the minimum.** Ratios scale linearly from 1.0
up to the WCAG AA target of 4.5 instead of passing or failing at a
threshold, so a policy gets signal for improving 1.5 toward 4.0. The
component takes the worst element rather than the mean. That was a
correction. Under a mean, a banner with an invisible headline and a legible
button scored 0.5, and a mean also lets an agent dilute one unreadable
element by adding several legible ones. The cost is a sparser signal, since
improving anything but the worst element moves nothing.

**Containment is not a collision.** Text sitting on a backing shape is
normal composition, so only partial overlap is penalized. Without that rule,
laying a background would be punished.

None of this reads pixels. Contrast is computed from the stored hex values,
which keeps scoring a pure function of the semantic state.

### Loopholes that are closed

Each of these has a test in `tests/test_reward.py`, and two of them were
open during development.

**Duplicate roles.** Spraying five headlines so that one of them might
satisfy the requirement. The constraint check demands exactly one element
per role.

**Token-sized elements.** A 4x4 yellow dot is not a call to action, so every
requirement carries a minimum width and height.

**Approximate colors.** `#FFD700` is a reasonable reading of "yellow" and
`#9E9E9E` is not. Named colors match within a flat per-channel RGB tolerance
of 90.

**Text hidden in its own fill.** Contrast originally scored only TEXT
elements, so a button labelled in the same color as its own background was
unreadable and scored nothing against it. Anything carrying content is
scored now, with text measured against whatever sits behind it and a
labelled shape against its own fill.

**Leaving the frame to escape a penalty.** Pushing an element off-canvas
avoids the overlap penalty but is caught by the bounds term.

**Alignment as a degenerate target.** Rewarding center alignment invites
stacking everything on the center line, so alignment carries a quarter of
the layout weight, which is a twentieth of the total. The overlap penalty
dominates that trade.

### Loopholes that are still open

**The thresholds have no justification.** The overlap tolerance of 5%, the
alignment slack of 10px, the color tolerance of 90, and the weights
themselves are all hand-chosen. None was tuned against held-out data,
because there is no labelled data here to tune against. They are starting
points and nothing more.

**Role is self-declared.** An element states what it is, and nothing checks
that a headline's content reads like a headline. A text box containing
"asdf" with `role="headline"` satisfies the constraint.

**Color matching is coarse.** Flat RGB distance accepts colors a person
would name differently.

**Layout quality is barely measured.** Not overlapping, staying on canvas,
and sitting on the center line is a low bar. Nothing scores visual hierarchy
or spacing, which is most of what makes a banner good.

### Does the reward separate good from bad

Two baseline policies run against the same task. A random policy, which
ignores the observation entirely, averages -0.74 over 20 seeds and never
exceeds -0.48. A rules-based policy that reads the observation and repairs
whatever is missing reaches 1.00 in four steps.

The gap matters more than either number. A reward a random policy already
scores well on is not measuring anything, and one that hand-written rules
cannot reach is not a target a learned policy would find either. The spread
here is roughly 1.8 of the available 2.0, which is the room a trained policy
would have to work in.

## 3. Scaling to 10,000 Parallel Rollouts

I have not run this at scale, so what follows is reasoning from what the
environment actually costs today. Measurements are single-threaded on an
M-series laptop, averaged over repeated calls.

| Operation | Cost | Note |
|---|---|---|
| Apply one action | 2.0 us | |
| Build observation, 3 elements | 18.9 us | 1.8 KB of JSON, 3 relations |
| Build observation, 8 elements | 106.3 us | 7.9 KB of JSON, 28 relations |
| Compute reward | 25.9 us | no rendering involved |
| Full episode, no render | 0.14 ms | |
| Render one frame | 5.63 ms | 1.44 MB at 800x600 RGB |

### The simulator is not the bottleneck, and rendering is not part of it

A full episode costs 0.14 ms, so ten thousand of them is under two seconds
of CPU work, which is negligible next to a single VLM forward pass. The
simulator itself will not be what limits throughput.

Rendering is a different story. One frame costs 5.63 ms, roughly 2,800 times
the cost of an action. Rendering every step of every rollout would mean
10,000 environments times 20 steps times 1.44 MB, which is about 288 GB of
pixel data per batch, produced by a Python library that holds the GIL.

The environment is already built so this does not have to happen. Nothing in
the observation or the reward reads pixels. Contrast comes from stored hex
values and layout from stored geometry, so `render()` runs only when a
caller wants an image. With a VLM in the loop that changes, because the
policy wants frames, and the honest answer is that rendering has to leave
the Python path. Rasterize on GPU in batch, at lower resolution than
800x600, and only for the steps the policy actually looks at.

### The observation grows quadratically

Going from 3 elements to 8 grew the observation from 1.8 KB to 7.9 KB and
from 3 relations to 28. Element count rose 2.7 times and cost rose 5.6
times, because every pair is described.

At the current cap of 8 elements this is fine. It stops being fine the
moment the canvas holds realistic designs. Forty elements would be 780
relations and roughly 200 KB of JSON per step, which is more context than a
model can use and more tokens than anyone wants to pay for twenty times an
episode.

The fix is to stop sending everything. Report relations only for pairs close
enough to matter, cap the relation list by configuration, and send deltas
after the first observation rather than the whole state every step.

### MCP is the wrong transport for training

The MCP server runs over stdio, one subprocess per server. Ten thousand
rollouts would mean ten thousand Python processes talking over pipes, which
is not something to attempt.

The interface a model uses interactively and the interface a trainer uses
are simply different. MCP is for a person or a client driving the canvas by
hand, for evaluation, and for debugging. For PPO the environment should be
stepped in process and vectorized, through the policy interface that
`policy.py` already defines. The environment supports both because nothing
about `Canvas` depends on how it is reached.

### What I would change

**Vectorize the engine across environments.** Today each canvas is a list of
dataclass instances and each step is a Python loop, which means ten thousand
environments are ten thousand independent loops. Holding state as arrays
across environments instead would turn a batch of steps into array
arithmetic, so one operation advances every rollout at once rather than one
at a time.

**Keep the reward pixel-free on purpose.** It already is, and at scale that
is worth more than it looks. Scoring a canvas costs 26 us and needs no
frame, so reward computation never forces a render.

**Decouple actors from the learner.** Episodes end at different lengths, so
collecting them synchronously means every worker waits on the slowest one.
Asynchronous rollout collection keeps workers busy instead of idling at a
barrier. Alongside it, a single batched inference server serving all actors
fits better than a model copy per worker, because requests arriving from
thousands of rollouts can be grouped into one forward pass rather than
running thousands of small ones.

**Make the observation budget explicit.** The relation count should be
bounded by configuration, not by the element cap happening to be small.

**Preserve determinism under parallelism.** The canvas assigns element ids
and z-order itself, which is what makes an action sequence reproducible
today. Any vectorized rewrite has to keep that property, because losing it
makes a failed rollout impossible to replay.
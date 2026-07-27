"""
Frozen experiment configuration (decisions D1-D4, locked).

Single source of truth for every experiment module. Values here map
one-to-one to the parameters stated in the paper (Sections III-IV).
Do not change after the first archived run without bumping RUN_SCHEMA
and noting the change in the paper's reproducibility appendix.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARCHIVE = DATA / "archive"           # frozen price/weather inputs
PRICES_DIR = ARCHIVE / "prices"      # one CSV per day: slot,valid_from,price_gbp_per_kwh
WEATHER_FC_DIR = ARCHIVE / "weather_fc"      # forecasts (as issued day-ahead)
WEATHER_ACT_DIR = ARCHIVE / "weather_actual" # realized values for scoring
RUNS = DATA / "runs"                 # JSONL logs, one file per experiment
RUN_SCHEMA = 1

# ---------------------------------------------------------------- D1: models
# Snapshot strings are captured from the provider response at first call
# and written into the run log; the values below are the request aliases.
MODELS = {
    "gpt":    {"provider": "openai",    "model": "gpt-4o-mini"},
    "gemini": {"provider": "google",    "model": "gemini-2.5-flash"},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    # Open-weight backends via DeepInfra (OpenAI-compatible endpoint).
    # Added for the open-vs-closed extension (Exp 1/2 only; scoped in grids).
    # Both are models reported by [17]: Llama-3.3 succeeded at multi-appliance
    # coordination there; Qwen-3 failed it. Re-running them under native
    # function calling tests whether the interface, not the model, drove that
    # gap. Served precision differs from reference weights -- footnoted:
    #   llama-3.3-70b: DeepInfra "Turbo" (optimized/quantized) serve
    #   qwen3-32b:     dense 32B, run in NON-THINKING mode (enable_thinking=
    #                  False) so it does not emit <think> blocks. Qwen3-32B
    #                  defaults to thinking ON, which is documented to plan
    #                  tool calls in the reasoning trace without emitting them
    #                  (~60% of the time) -- i.e. conversational
    #                  short-circuiting. Disabling thinking gives ~100% tool
    #                  execution and keeps the interface comparison clean.
    "llama-3.3": {"provider": "deepinfra",
                  "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "qwen-3": {"provider": "deepinfra",
               "model": "Qwen/Qwen3-32B",
               "enable_thinking": False},
}
TEMPERATURE = 0.0
INTERFACES = ("fc", "text")          # native function calling / parsed ReAct

# ---------------------------------------------------------------- D2: tariff
AGILE_REGION = "C"                   # London distribution region
AGILE_PRODUCT = None                 # resolved once by fetch_archives, then frozen
EXPORT_RATE_GBP = 0.05               # flat SEG export rate. Deliberately a
                                     # standard (non-supplier-bundled) SEG offer
                                     # (~3-7p typical), NOT Octopus Outgoing 15p:
                                     # with F above overnight Agile prices,
                                     # self-consumption is never optimal and
                                     # Exp 3 degenerates (verified in testing).

# ---------------------------------------------------------------- D3: household
LAT, LON = 51.51, -0.13              # London (matches tariff region)
TIMEZONE = "Europe/London"
PV_KWP = 4.0                         # nameplate capacity P_stc
PV_PR = 0.8                          # performance ratio
PV_ALPHA = 0.004                     # temperature coefficient [1/K]
PV_NOCT = 45.0                       # nominal operating cell temperature [degC]
BASE_LOAD_KWH_PER_SLOT = 0.0         # stated simplification: B_t = 0
POWER_CAP_KW = None                  # default off; Exp 2 S2 sets 9.0

# ---------------------------------------------------------------- time grid
SLOTS_PER_DAY = 48                   # half-hourly (Agile settlement)
SLOT_HOURS = 0.5

# ---------------------------------------------------------------- appliances
# Powers/durations follow the anchor paper, re-expressed at 30-min slots.
APPLIANCES = {
    "washing_machine": {"power_kw": 2.0, "slots": 4},   # 120 min
    "dishwasher":      {"power_kw": 1.8, "slots": 3},   # 90 min
    "ev_charger":      {"power_kw": 7.4, "slots": 12},  # 360 min
}
EV_DEADLINE_BUFFER_SLOTS = 1          # 30-min buffer before calendar event

# ---------------------------------------------------------------- D4: archive
# Most recent fully-completed window with both Agile prices and weather
# actuals available. fetch_archives.py pulls this span, computes the
# volatility terciles and clear-sky regimes, and writes day_selection.json;
# the chosen days are then treated as frozen.
ARCHIVE_START = "2026-04-20"
ARCHIVE_END = "2026-06-20"

# Experiment day counts (selection itself lives in day_selection.json)
EXP1_DAYS_PER_TERCILE = 4            # 12 days total
EXP3_DAYS_PER_REGIME = 5             # 15 days total
EXP4A_WEEK_DAYS = 7                  # consecutive, weekdays + weekend
EXP4B_WEEKS = 2

# ---------------------------------------------------------------- run counts
REPS_EXP1 = 3
REPS_EXP2 = 3
REPS_EXP3 = 3
PASSES_EXP4A = 3
REPS_EXP4B = 5
NOISE_LEVELS = (0.10, 0.25, 0.50)    # multiplicative PV-forecast noise (champion only)

# ---------------------------------------------------------------- runner
AGENT_RECURSION_LIMIT = 30           # max ReAct iterations before abort
# DeepInfra output-token cap. Some open-weight serves (e.g. Qwen3-32B with
# max_model_len=40960) reject the provider default max_tokens=65536. 8192 is
# ample for scheduling replies and leaves room for the tool-augmented input.
DEEPINFRA_MAX_TOKENS = 8192
FINAL_ANSWER_TRUNCATE = 2500         # chars of final answer kept in the log
# $/1M tokens (input, output) -- BUDGETING ESTIMATES ONLY; verify against
# provider pricing pages before trusting cost totals.
PRICE_PER_MTOK = {
    "gpt":    (0.15, 0.60),
    "gemini": (0.30, 2.50),
    "claude": (3.00, 15.00),
    # DeepInfra list prices (USD/1M tok) at time of run -- verify before trusting.
    "llama-3.3": (0.13, 0.39),
    "qwen-3":    (0.10, 0.30),
    "mock":   (0.0, 0.0),
}

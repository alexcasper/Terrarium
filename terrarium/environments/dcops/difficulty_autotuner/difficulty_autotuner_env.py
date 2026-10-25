"""InstanceDifficultyAutotuner: closed-loop difficulty calibration for multi-agent DCOPs.

Provides two tightly integrated components:
1. ``DifficultyKnobSpec`` + ``DifficultyAutotuner``: a general, reusable difficulty
   calibration engine that wraps any seeded Terrarium generator (such as
   ``IsThisSeatTakenEnvironment`` or ``InstanceDifficultyAutotunerEnvironment``)
   and searches over instance-generator parameters within a bounded ``probe_budget``
   to hit a target normalized performance ratio (default ``target_score_ratio = 0.70``,
   the "Goldilocks / 70%-solve zone").
2. ``InstanceDifficultyAutotunerEnvironment``: a multi-round constraint-satisfaction /
   resource-coloring DCOP environment where ``num_agents`` solvers select colors
   under private unary preferences and binary conflict edges, and the environment
   auto-calibrates ``difficulty_level in [1, 10]`` after every round using
   Robbins-Monro / bracketed step updates toward ``target_score_ratio``.

Analytic benchmarks (computed from the seeded instance streams, never from the LLM):
- ``exact_optimum_benchmark``: brute-force optimal joint score across all
  ``num_colors^num_agents`` assignments per round;
- ``random_uniform_benchmark``: exact expected joint score under uniform random
  coloring;
- ``static_uncalibrated_benchmark``: counterfactual score ratio and calibration
  error trace had ``difficulty_level`` remained fixed at ``initial_difficulty_level``
  instead of adapting.
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.utils import clear_seed_directories, get_run_timestamp
from .difficulty_autotuner_prompts import InstanceDifficultyAutotunerPrompts

logger = logging.getLogger(__name__)


# ====================================================================== #
# Reusable Difficulty Calibration Engine
# ====================================================================== #


@dataclass(frozen=True)
class DifficultyKnobSpec:
    """Specification of a tunable generator parameter controlling instance hardness."""

    name: str
    min_val: float
    max_val: float
    direction: str = "increasing_harder"
    is_integer: bool = True
    step_size: float = 1.0
    section: str = "environment"

    def __post_init__(self) -> None:
        if self.min_val > self.max_val:
            raise ValueError(f"Knob {self.name}: min_val ({self.min_val}) > max_val ({self.max_val})")
        if self.direction not in ("increasing_harder", "decreasing_harder"):
            raise ValueError(
                f"Knob {self.name}: direction must be 'increasing_harder' or 'decreasing_harder'"
            )

    def _format(self, raw: float) -> Any:
        clamped = min(self.max_val, max(self.min_val, raw))
        if self.is_integer:
            return int(round(clamped))
        return round(float(clamped), 4)

    def difficulty_to_value(self, difficulty_fraction: float) -> Any:
        """Map normalized difficulty d in [0.0, 1.0] to a parameter value."""
        d = min(1.0, max(0.0, float(difficulty_fraction)))
        span = self.max_val - self.min_val
        if self.direction == "increasing_harder":
            raw = self.min_val + d * span
        else:
            raw = self.max_val - d * span
        return self._format(raw)

    def value_to_difficulty(self, value: float) -> float:
        """Map a parameter value back to normalized difficulty d in [0.0, 1.0]."""
        span = self.max_val - self.min_val
        if span <= 0:
            return 0.5
        clamped = min(self.max_val, max(self.min_val, float(value)))
        if self.direction == "increasing_harder":
            return round((clamped - self.min_val) / span, 4)
        return round((self.max_val - clamped) / span, 4)

    def step_value(
        self,
        current_val: float,
        harder: bool,
        step_override: Optional[float] = None,
    ) -> Any:
        """Step the parameter value toward harder or easier instances."""
        delta = float(step_override if step_override is not None else self.step_size)
        sign = 1.0 if harder else -1.0
        if self.direction == "decreasing_harder":
            sign = -sign
        return self._format(float(current_val) + sign * delta)


PRESET_ENVIRONMENT_KNOBS: Dict[str, Tuple[DifficultyKnobSpec, ...]] = {
    "InstanceDifficultyAutotunerEnvironment": (
        DifficultyKnobSpec(
            name="initial_difficulty_level",
            min_val=1,
            max_val=10,
            direction="increasing_harder",
            is_integer=True,
            step_size=1,
        ),
    ),
    "IsThisSeatTakenEnvironment": (
        DifficultyKnobSpec(
            name="empty_seat_buffer",
            min_val=0,
            max_val=8,
            direction="decreasing_harder",
            is_integer=True,
            step_size=1,
        ),
        DifficultyKnobSpec(
            name="group_move_penalty",
            min_val=0.05,
            max_val=0.65,
            direction="increasing_harder",
            is_integer=False,
            step_size=0.10,
        ),
    ),
}


class DifficultyAutotuner:
    """Adaptive difficulty autotuner over seeded DCOP instance generators.

    Supports both:
    - per-round Robbins-Monro stepping (used inside ``InstanceDifficultyAutotunerEnvironment``);
    - bounded probe-budget bracketed bisection search over one or more ``DifficultyKnobSpec``
      knobs against an evaluator callback ``eval_fn(config) -> score_ratio``.
    """

    def __init__(
        self,
        knobs: Optional[Sequence[DifficultyKnobSpec]] = None,
        target_score_ratio: float = 0.70,
        tolerance: float = 0.08,
        probe_budget: int = 5,
        base_config: Optional[Dict[str, Any]] = None,
    ):
        if not (0.0 < float(target_score_ratio) <= 1.0):
            raise ValueError("target_score_ratio must be in (0.0, 1.0]")
        if float(tolerance) < 0.0:
            raise ValueError("tolerance must be non-negative")
        if int(probe_budget) < 1:
            raise ValueError("probe_budget must be at least 1")

        self.knobs: List[DifficultyKnobSpec] = list(
            knobs if knobs is not None else PRESET_ENVIRONMENT_KNOBS["InstanceDifficultyAutotunerEnvironment"]
        )
        if not self.knobs:
            raise ValueError("DifficultyAutotuner requires at least one DifficultyKnobSpec")

        self.target_score_ratio = float(target_score_ratio)
        self.tolerance = float(tolerance)
        self.probe_budget = int(probe_budget)
        self.base_config: Dict[str, Any] = copy.deepcopy(base_config) if base_config else {
            "simulation": {"max_iterations": 5, "seed": 42, "run_timestamp": "autotune"},
            "environment": {"name": "InstanceDifficultyAutotunerEnvironment"},
            "communication_network": {"topology": "complete", "num_agents": 4},
        }

    @classmethod
    def for_environment(
        cls,
        environment_name: str,
        target_score_ratio: float = 0.70,
        tolerance: float = 0.08,
        probe_budget: int = 5,
        base_config: Optional[Dict[str, Any]] = None,
        knobs: Optional[Sequence[DifficultyKnobSpec]] = None,
    ) -> "DifficultyAutotuner":
        resolved_knobs = knobs or PRESET_ENVIRONMENT_KNOBS.get(environment_name)
        if not resolved_knobs:
            raise ValueError(
                f"No preset knobs for {environment_name!r}; pass explicit DifficultyKnobSpec list."
            )
        cfg = copy.deepcopy(base_config) if base_config else {
            "simulation": {"max_iterations": 5, "seed": 42, "run_timestamp": "autotune"},
            "environment": {"name": environment_name},
            "communication_network": {"topology": "complete", "num_agents": 4},
        }
        cfg.setdefault("environment", {})["name"] = environment_name
        return cls(
            knobs=resolved_knobs,
            target_score_ratio=target_score_ratio,
            tolerance=tolerance,
            probe_budget=probe_budget,
            base_config=cfg,
        )

    def step_robbins_monro(
        self,
        current_level: int,
        score_ratio: float,
        step_size: int = 1,
        min_level: int = 1,
        max_level: int = 10,
    ) -> Tuple[int, str, bool]:
        """Single-step closed-loop difficulty update toward target_score_ratio.

        Returns ``(next_level, direction, converged)`` where ``direction`` is
        ``'harder'``, ``'easier'``, or ``'hold'``.
        """
        ratio = float(score_ratio)
        step = max(1, int(step_size))
        upper = self.target_score_ratio + self.tolerance
        lower = self.target_score_ratio - self.tolerance

        if ratio > upper:
            next_level = min(int(max_level), int(current_level) + step)
            return next_level, "harder", False
        if ratio < lower:
            next_level = max(int(min_level), int(current_level) - step)
            return next_level, "easier", False
        return int(current_level), "hold", True

    def build_config_at_difficulty(
        self,
        difficulty_fraction: float,
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return a deep-copied config and knob value dict at normalized difficulty d in [0, 1]."""
        cfg = copy.deepcopy(base_config if base_config is not None else self.base_config)
        knob_values: Dict[str, Any] = {}
        for knob in self.knobs:
            val = knob.difficulty_to_value(difficulty_fraction)
            cfg.setdefault(knob.section, {})[knob.name] = val
            knob_values[knob.name] = val
        return cfg, knob_values

    def calibrate(
        self,
        eval_fn: Callable[[Dict[str, Any]], float],
        initial_difficulty: float = 0.5,
        stop_on_convergence: bool = True,
    ) -> Dict[str, Any]:
        """Run bracketed bisection search over normalized difficulty in [0.0, 1.0].

        At each probe, ``eval_fn(config)`` returns a normalized score ratio in [0, 1].
        If ``score_ratio > target + tolerance`` (instance too easy), the lower bracket
        moves up; if ``score_ratio < target - tolerance`` (instance too hard), the upper
        bracket moves down.
        """
        low_d = 0.0
        high_d = 1.0
        current_d = min(1.0, max(0.0, float(initial_difficulty)))

        probe_history: List[Dict[str, Any]] = []
        best_entry: Optional[Dict[str, Any]] = None
        best_config: Optional[Dict[str, Any]] = None
        converged = False

        for probe_idx in range(1, self.probe_budget + 1):
            cfg, knob_values = self.build_config_at_difficulty(current_d)
            score_ratio = float(eval_fn(cfg))
            abs_error = abs(score_ratio - self.target_score_ratio)
            in_band = abs_error <= self.tolerance

            if score_ratio > self.target_score_ratio + self.tolerance:
                direction = "harder"
                low_d = current_d
            elif score_ratio < self.target_score_ratio - self.tolerance:
                direction = "easier"
                high_d = current_d
            else:
                direction = "hold"
                converged = True

            entry = {
                "probe": probe_idx,
                "difficulty_fraction": round(current_d, 4),
                "knob_values": dict(knob_values),
                "score_ratio": round(score_ratio, 4),
                "abs_error": round(abs_error, 4),
                "in_tolerance_band": in_band,
                "direction": direction,
                "bracket_after": [round(low_d, 4), round(high_d, 4)],
            }
            probe_history.append(entry)

            if best_entry is None or abs_error < best_entry["abs_error"]:
                best_entry = entry
                best_config = cfg

            if in_band and stop_on_convergence:
                break

            current_d = 0.5 * (low_d + high_d)

        assert best_entry is not None and best_config is not None
        return {
            "target_score_ratio": self.target_score_ratio,
            "tolerance": self.tolerance,
            "probe_budget": self.probe_budget,
            "probes_used": len(probe_history),
            "converged": converged,
            "best_difficulty_fraction": best_entry["difficulty_fraction"],
            "best_knob_values": dict(best_entry["knob_values"]),
            "best_score_ratio": best_entry["score_ratio"],
            "best_abs_error": best_entry["abs_error"],
            "best_config": best_config,
            "probe_history": probe_history,
        }


# ====================================================================== #
# InstanceDifficultyAutotunerEnvironment
# ====================================================================== #


class InstanceDifficultyAutotunerEnvironment(AbstractEnvironment):
    """Multi-round constraint satisfaction environment with closed-loop difficulty calibration."""

    def __init__(self, communication_protocol, config, tool_logger):
        self.full_config = config
        self.env_config: Dict[str, Any] = config["environment"]
        self.simulation_config: Dict[str, Any] = config["simulation"]
        self.communication_protocol = communication_protocol
        self.communication_protocol.environment = self
        self.tool_logger = tool_logger

        self.max_iterations = int(self.simulation_config.get("max_iterations", 5))
        self.current_seed = int(self.simulation_config.get("seed", 0))
        self.rng = random.Random(self.current_seed)
        self.run_timestamp = get_run_timestamp(self.full_config)
        clear_seed_directories(self.__class__.__name__, self.current_seed, self.full_config)

        network_cfg = config.get("communication_network") or {}
        num_agents = network_cfg.get("num_agents")
        if num_agents is None or type(num_agents) is not int:
            raise ValueError("communication_network.num_agents must be an integer")
        self.num_agents = int(num_agents)
        if self.num_agents < 2:
            raise ValueError("InstanceDifficultyAutotunerEnvironment needs at least 2 solvers")

        # Calibration and instance-generation parameters.
        self.min_difficulty_level = int(self.env_config.get("min_difficulty_level", 1))
        self.max_difficulty_level = int(self.env_config.get("max_difficulty_level", 10))
        if self.min_difficulty_level < 1 or self.min_difficulty_level > self.max_difficulty_level:
            raise ValueError("Invalid min_difficulty_level / max_difficulty_level range")

        raw_init_level = self.env_config.get(
            "initial_difficulty_level", self.env_config.get("difficulty_level", 5)
        )
        self.initial_difficulty_level = min(
            self.max_difficulty_level,
            max(self.min_difficulty_level, int(raw_init_level)),
        )
        self.current_difficulty_level = self.initial_difficulty_level

        self.target_score_ratio = float(self.env_config.get("target_score_ratio", 0.70))
        self.tolerance = float(self.env_config.get("tolerance", 0.08))
        self.step_size = max(1, int(self.env_config.get("step_size", 2)))
        self.adaptive_enabled = bool(self.env_config.get("adaptive_enabled", True))
        self.base_preference_min = int(self.env_config.get("base_preference_min", 10))
        self.base_preference_max = int(self.env_config.get("base_preference_max", 30))
        if self.base_preference_min <= 0 or self.base_preference_min > self.base_preference_max:
            raise ValueError("Invalid base_preference_min / base_preference_max range")

        self.agent_names = [f"solver_{idx + 1}" for idx in range(self.num_agents)]
        self.all_pairs: List[Tuple[str, str]] = self._canonical_agent_pairs()

        self.autotuner = DifficultyAutotuner(
            knobs=PRESET_ENVIRONMENT_KNOBS["InstanceDifficultyAutotunerEnvironment"],
            target_score_ratio=self.target_score_ratio,
            tolerance=self.tolerance,
            probe_budget=self.max_iterations,
            base_config=self.full_config,
        )

        # Pre-generate seeded instance catalog for every (round_index, difficulty_level)
        # so both adaptive trajectories and counterfactual static benchmarks are exact.
        self.instance_catalog: List[Dict[int, Dict[str, Any]]] = [
            {
                level: self._generate_instance(round_idx, level)
                for level in range(self.min_difficulty_level, self.max_difficulty_level + 1)
            }
            for round_idx in range(self.max_iterations)
        ]

        # Per-round runtime state.
        self.round_num = 0
        self.proposals: Dict[str, List[Dict[str, Any]]] = {name: [] for name in self.agent_names}
        self.pending_actions: Dict[str, Optional[Dict[str, Any]]] = {
            name: None for name in self.agent_names
        }
        self.round_history: List[Dict[str, Any]] = []
        self.difficulty_trajectory: List[int] = [self.initial_difficulty_level]
        self.score_ratio_trajectory: List[float] = []
        self.joint_reward_history: List[int] = []

        self.prompts = InstanceDifficultyAutotunerPrompts(self, self.full_config)
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "%s initialized with %s solvers, initial_difficulty=%s, target_ratio=%.2f, seed=%s",
            self.__class__.__name__,
            self.num_agents,
            self.initial_difficulty_level,
            self.target_score_ratio,
            self.current_seed,
        )

    # ------------------------------------------------------------------ #
    # Seeded Instance Generation & Exact Scoring
    # ------------------------------------------------------------------ #

    def _canonical_agent_pairs(self) -> List[Tuple[str, str]]:
        """Cycle edges first, followed by remaining chords in deterministic order."""
        seen: set[Tuple[int, int]] = set()
        pairs: List[Tuple[str, str]] = []
        for idx in range(self.num_agents):
            u, v = sorted((idx, (idx + 1) % self.num_agents))
            if u != v and (u, v) not in seen:
                seen.add((u, v))
                pairs.append((self.agent_names[u], self.agent_names[v]))
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if (i, j) not in seen:
                    seen.add((i, j))
                    pairs.append((self.agent_names[i], self.agent_names[j]))
        return pairs

    def _level_parameters(self, level: int) -> Dict[str, Any]:
        """Map integer difficulty level in [1, 10] to structural CSP parameters."""
        lvl = min(self.max_difficulty_level, max(self.min_difficulty_level, int(level)))
        if lvl <= 4:
            num_colors = 3
        elif lvl <= 8:
            num_colors = 4
        else:
            num_colors = 5

        max_edges = len(self.all_pairs)
        num_edges = min(max_edges, max(1, 1 + (lvl + 1) // 2))
        conflict_density = round(num_edges / max_edges, 3) if max_edges else 0.0
        per_endpoint_penalty = lvl + 2
        conflict_penalty = 2 * per_endpoint_penalty
        trap_edges = min(num_edges, lvl // 2)
        return {
            "difficulty_level": lvl,
            "num_colors": num_colors,
            "num_edges": num_edges,
            "max_edges": max_edges,
            "conflict_density": conflict_density,
            "per_endpoint_penalty": per_endpoint_penalty,
            "conflict_penalty": conflict_penalty,
            "trap_edges": trap_edges,
        }

    def _evaluate_assignment_on_raw(
        self,
        preferences: Dict[str, Dict[int, int]],
        edges: List[Tuple[str, str]],
        per_endpoint_penalty: int,
        assignment: Mapping[str, int],
    ) -> Dict[str, Any]:
        """Pure integer payoff and conflict evaluation for a color assignment."""
        payoffs: Dict[str, int] = {
            name: int(preferences[name][int(assignment[name])]) for name in self.agent_names
        }
        violated_edges: List[Tuple[str, str]] = []
        conflicts_by_agent: Dict[str, int] = {name: 0 for name in self.agent_names}

        for u, v in edges:
            if int(assignment[u]) == int(assignment[v]):
                violated_edges.append((u, v))
                payoffs[u] -= per_endpoint_penalty
                payoffs[v] -= per_endpoint_penalty
                conflicts_by_agent[u] += 1
                conflicts_by_agent[v] += 1

        raw_joint = sum(payoffs.values())
        realized_joint = max(0, raw_joint)
        return {
            "payoffs": payoffs,
            "raw_joint_score": raw_joint,
            "joint_score": realized_joint,
            "violated_edges": violated_edges,
            "num_conflicts": len(violated_edges),
            "conflicts_by_agent": conflicts_by_agent,
        }

    def _generate_instance(self, round_index: int, level: int) -> Dict[str, Any]:
        """Generate a seeded CSP instance for (round_index, level) and solve its exact optimum."""
        params = self._level_parameters(level)
        num_colors = params["num_colors"]
        num_edges = params["num_edges"]
        per_endpoint_penalty = params["per_endpoint_penalty"]
        trap_edges = params["trap_edges"]

        edges = list(self.all_pairs[:num_edges])
        preferences: Dict[str, Dict[int, int]] = {}
        for name in self.agent_names:
            preferences[name] = {
                c: self.rng.randint(self.base_preference_min, self.base_preference_max)
                for c in range(num_colors)
            }

        # At higher difficulty levels, align top unary preferences across the first
        # `trap_edges` conflict edges so naive greedy picks collide unless solvers coordinate.
        focal_color = (round_index + level) % num_colors
        for edge_idx in range(trap_edges):
            u, v = edges[edge_idx]
            preferences[u][focal_color] = self.base_preference_max
            preferences[v][focal_color] = self.base_preference_max
            for c in range(num_colors):
                if c != focal_color:
                    preferences[u][c] = min(preferences[u][c], self.base_preference_max - 3)
                    preferences[v][c] = min(preferences[v][c], self.base_preference_max - 3)

        # Brute-force all num_colors^num_agents assignments for exact optimum & uniform random mean.
        best_joint = -1
        best_tuple: Tuple[int, ...] = tuple(0 for _ in self.agent_names)
        total_joint_across_space = 0
        total_assignments = 0

        for combo in itertools.product(range(num_colors), repeat=self.num_agents):
            assign_map = {self.agent_names[i]: combo[i] for i in range(self.num_agents)}
            outcome = self._evaluate_assignment_on_raw(
                preferences, edges, per_endpoint_penalty, assign_map
            )
            joint_val = outcome["joint_score"]
            total_joint_across_space += joint_val
            total_assignments += 1
            if joint_val > best_joint:
                best_joint = joint_val
                best_tuple = combo

        optimal_assignment = {self.agent_names[i]: best_tuple[i] for i in range(self.num_agents)}
        random_expected = (
            round(total_joint_across_space / total_assignments, 3) if total_assignments else 0.0
        )

        # Greedy unary assignment (each solver picks its highest unary preference color).
        greedy_assignment = {
            name: max(range(num_colors), key=lambda c: (preferences[name][c], -c))
            for name in self.agent_names
        }
        greedy_outcome = self._evaluate_assignment_on_raw(
            preferences, edges, per_endpoint_penalty, greedy_assignment
        )
        greedy_ratio = (
            round(greedy_outcome["joint_score"] / best_joint, 4) if best_joint > 0 else 0.0
        )

        return {
            "round_index": round_index,
            **params,
            "edges": edges,
            "preferences": preferences,
            "optimal_assignment": optimal_assignment,
            "optimal_joint_score": best_joint,
            "random_expected_score": random_expected,
            "greedy_assignment": greedy_assignment,
            "greedy_joint_score": greedy_outcome["joint_score"],
            "greedy_score_ratio": greedy_ratio,
        }

    def get_current_instance(self, iteration: Optional[int] = None) -> Dict[str, Any]:
        """Return the active CSP instance for the current (or specified) 1-based iteration."""
        if iteration is None:
            round_idx = min(max(self.round_num, 0), self.max_iterations - 1)
        else:
            round_idx = min(max(int(iteration) - 1, 0), self.max_iterations - 1)
        return self.instance_catalog[round_idx][self.current_difficulty_level]

    # ------------------------------------------------------------------ #
    # Action Recording (called by tools)
    # ------------------------------------------------------------------ #

    def _validate_color(self, raw_color: Any, num_colors: int) -> Tuple[Optional[int], Optional[str]]:
        if raw_color is None or isinstance(raw_color, bool):
            return None, "color must be an integer"
        try:
            color_int = int(raw_color)
        except (TypeError, ValueError):
            return None, "color must be an integer"
        if not (0 <= color_int < num_colors):
            return None, f"color must be an integer between 0 and {num_colors - 1}"
        return color_int, None

    def record_proposal(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown solver {agent_name}"}
        instance = self.get_current_instance(iteration)
        color_int, err = self._validate_color(arguments.get("color"), instance["num_colors"])
        if err is not None:
            return {"status": "failed", "reason": err}

        entry: Dict[str, Any] = {"color": color_int}
        message = str(arguments.get("message") or "").strip()
        if message:
            entry["message"] = message
        self.proposals[agent_name].append(entry)
        return {
            "status": "success",
            "proposal": entry,
            "difficulty_level": instance["difficulty_level"],
            "num_colors": instance["num_colors"],
        }

    def commit_assignment(
        self,
        agent_name: str,
        arguments: Dict[str, Any],
        iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        if agent_name not in self.agent_names:
            return {"status": "failed", "reason": f"unknown solver {agent_name}"}
        instance = self.get_current_instance(iteration)
        color_int, err = self._validate_color(arguments.get("color"), instance["num_colors"])
        if err is not None:
            return {"status": "failed", "reason": err}

        self.pending_actions[agent_name] = {"color": color_int}
        return {
            "status": "success",
            "action": "commit_assignment",
            "color": color_int,
            "difficulty_level": instance["difficulty_level"],
            "num_colors": instance["num_colors"],
        }

    # ------------------------------------------------------------------ #
    # Round Settlement & Closed-Loop Autotuning
    # ------------------------------------------------------------------ #

    def _settle_round(self, round_index: int) -> Dict[str, Any]:
        """Settle one completed round and update difficulty_level via Robbins-Monro."""
        difficulty_before = self.current_difficulty_level
        instance = self.instance_catalog[round_index][difficulty_before]
        num_colors = instance["num_colors"]

        actions: Dict[str, Dict[str, Any]] = {}
        assignments: Dict[str, int] = {}
        for name in self.agent_names:
            act = self.pending_actions.get(name)
            if act is None or not (0 <= int(act.get("color", 0)) < num_colors):
                act = {"color": 0, "fallback": True}
            else:
                act = {"color": int(act["color"])}
            actions[name] = act
            assignments[name] = int(act["color"])

        outcome = self._evaluate_assignment_on_raw(
            instance["preferences"],
            instance["edges"],
            instance["per_endpoint_penalty"],
            assignments,
        )
        joint_score = outcome["joint_score"]
        optimal_joint = instance["optimal_joint_score"]
        score_ratio = round(joint_score / optimal_joint, 4) if optimal_joint > 0 else 0.0
        calibration_error = round(abs(score_ratio - self.target_score_ratio), 4)

        if self.adaptive_enabled:
            difficulty_after, tuning_direction, converged_this_round = (
                self.autotuner.step_robbins_monro(
                    current_level=difficulty_before,
                    score_ratio=score_ratio,
                    step_size=self.step_size,
                    min_level=self.min_difficulty_level,
                    max_level=self.max_difficulty_level,
                )
            )
        else:
            difficulty_after = difficulty_before
            tuning_direction = "hold"
            converged_this_round = calibration_error <= self.tolerance

        self.current_difficulty_level = difficulty_after

        return {
            "round": round_index + 1,
            "difficulty_level": difficulty_before,
            "difficulty_before": difficulty_before,
            "difficulty_after": difficulty_after,
            "tuning_direction": tuning_direction,
            "converged_this_round": converged_this_round,
            "num_colors": num_colors,
            "edges": list(instance["edges"]),
            "conflict_density": instance["conflict_density"],
            "per_endpoint_penalty": instance["per_endpoint_penalty"],
            "conflict_penalty": instance["conflict_penalty"],
            "preferences": copy.deepcopy(instance["preferences"]),
            "actions": actions,
            "assignments": assignments,
            "violated_edges": outcome["violated_edges"],
            "num_conflicts": outcome["num_conflicts"],
            "conflicts_by_agent": outcome["conflicts_by_agent"],
            "payoffs": outcome["payoffs"],
            "raw_joint_score": outcome["raw_joint_score"],
            "joint_score": joint_score,
            "optimal_joint_score": optimal_joint,
            "optimal_assignment": dict(instance["optimal_assignment"]),
            "random_expected_score": instance["random_expected_score"],
            "score_ratio": score_ratio,
            "target_score_ratio": self.target_score_ratio,
            "calibration_error": calibration_error,
        }

    # ------------------------------------------------------------------ #
    # Analytic Benchmarks (never from the LLM)
    # ------------------------------------------------------------------ #

    def _active_instances_for_benchmarks(self) -> List[Dict[str, Any]]:
        if self.round_history:
            return [
                self.instance_catalog[idx][rec["difficulty_level"]]
                for idx, rec in enumerate(self.round_history)
            ]
        return [
            self.instance_catalog[idx][self.initial_difficulty_level]
            for idx in range(self.max_iterations)
        ]

    def exact_optimum_benchmark(self) -> Dict[str, Any]:
        """Exact brute-force maximum joint score across the round instances."""
        instances = self._active_instances_for_benchmarks()
        per_round = [int(inst["optimal_joint_score"]) for inst in instances]
        return {
            "joint_score": sum(per_round),
            "per_round_optimal_score": per_round,
            "score_ratio": 1.0,
        }

    def random_uniform_benchmark(self) -> Dict[str, Any]:
        """Exact expected score under uniform random coloring across all rounds."""
        instances = self._active_instances_for_benchmarks()
        per_round = [float(inst["random_expected_score"]) for inst in instances]
        total_opt = sum(int(inst["optimal_joint_score"]) for inst in instances)
        total_rand = round(sum(per_round), 3)
        return {
            "expected_joint_score": total_rand,
            "per_round_expected_score": per_round,
            "score_ratio": round(total_rand / total_opt, 4) if total_opt > 0 else 0.0,
        }

    def static_uncalibrated_benchmark(self) -> Dict[str, Any]:
        """Counterfactual trace if difficulty stayed fixed at initial_difficulty_level."""
        static_level = self.initial_difficulty_level
        per_round_ratios: List[float] = []
        per_round_errors: List[float] = []
        total_joint = 0
        total_opt = 0

        for idx in range(len(self.round_history) if self.round_history else self.max_iterations):
            inst = self.instance_catalog[idx][static_level]
            num_colors = inst["num_colors"]
            if idx < len(self.round_history):
                recorded_assign = self.round_history[idx]["assignments"]
                assign = {
                    name: int(recorded_assign[name]) % num_colors for name in self.agent_names
                }
            else:
                assign = dict(inst["greedy_assignment"])
            outcome = self._evaluate_assignment_on_raw(
                inst["preferences"], inst["edges"], inst["per_endpoint_penalty"], assign
            )
            joint = outcome["joint_score"]
            opt = inst["optimal_joint_score"]
            ratio = round(joint / opt, 4) if opt > 0 else 0.0
            err = round(abs(ratio - self.target_score_ratio), 4)
            per_round_ratios.append(ratio)
            per_round_errors.append(err)
            total_joint += joint
            total_opt += opt

        mae = round(sum(per_round_errors) / len(per_round_errors), 4) if per_round_errors else 0.0
        return {
            "fixed_difficulty_level": static_level,
            "joint_score": total_joint,
            "optimal_joint_score": total_opt,
            "score_ratio_trajectory": per_round_ratios,
            "calibration_error_mae": mae,
        }

    # ------------------------------------------------------------------ #
    # AbstractEnvironment Contract
    # ------------------------------------------------------------------ #

    def get_network_context(self) -> str:
        return (
            f"You are {self.num_agents} resource-assignment solvers across {self.max_iterations} "
            f"rounds. Each round presents a graph-coloring / constraint satisfaction instance "
            f"at difficulty level {self.min_difficulty_level}..{self.max_difficulty_level}. "
            f"Each solver chooses a color index in 0..(num_colors - 1), earning its private "
            f"color preference minus the per-endpoint penalty on any active edge where both "
            f"connected solvers pick the same color. After each round, the environment adjusts "
            f"the difficulty level toward a target score ratio of {self.target_score_ratio:.2f}. "
            f"The blackboard is your shared communication channel."
        )

    def build_agent_context(
        self, agent_name: str, phase: str, iteration: int, **kwargs
    ) -> Dict[str, Any]:
        instance = self.get_current_instance(iteration)
        neighbors = sorted(
            v if u == agent_name else u
            for u, v in instance["edges"]
            if u == agent_name or v == agent_name
        )
        last_record = self.round_history[-1] if self.round_history else None
        context = {
            "agent_name": agent_name,
            "phase": phase,
            "iteration": iteration,
            "round": min(max(iteration, 1), self.max_iterations),
            "max_rounds": self.max_iterations,
            "difficulty_level": instance["difficulty_level"],
            "num_colors": instance["num_colors"],
            "conflict_density": instance["conflict_density"],
            "per_endpoint_penalty": instance["per_endpoint_penalty"],
            "conflict_penalty": instance["conflict_penalty"],
            "own_preferences": dict(instance["preferences"][agent_name]),
            "conflict_neighbors": neighbors,
            "active_edges": list(instance["edges"]),
            "target_score_ratio": self.target_score_ratio,
            "tolerance": self.tolerance,
            "own_payoff_history": [
                int(r["payoffs"].get(agent_name, 0)) for r in self.round_history
            ],
            "difficulty_trajectory": list(self.difficulty_trajectory),
            "last_round": last_record,
            "legal_actions": (
                ["propose_assignment"] if phase == "planning" else ["commit_assignment"]
            ),
        }
        for key, value in kwargs.items():
            context[key] = value
        return context

    def done(self, iteration: int) -> bool:
        return iteration > self.max_iterations

    def compute_max_joint_reward(self) -> float:
        return float(self.exact_optimum_benchmark()["joint_score"])

    def joint_reward(self, actions: Mapping[str, Any]) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["joint_score"])

    def agent_reward(self, agent_name: str, action: Any) -> float:
        if not self.round_history:
            return 0.0
        return float(self.round_history[-1]["payoffs"].get(agent_name, 0))

    def log_iteration(self, iteration: int) -> None:
        round_index = iteration - 1
        record = self._settle_round(round_index)
        self.round_history.append(record)
        self.round_num = iteration
        self.pending_actions = {name: None for name in self.agent_names}
        self.difficulty_trajectory.append(record["difficulty_after"])
        self.score_ratio_trajectory.append(record["score_ratio"])
        self.joint_reward_history.append(record["joint_score"])
        self.max_joint_reward = self.compute_max_joint_reward()

        logger.info(
            "=== %s Round %s === level=%s->%s (%s) score=%s/%s (ratio=%.3f) conflicts=%s",
            self.__class__.__name__,
            record["round"],
            record["difficulty_before"],
            record["difficulty_after"],
            record["tuning_direction"],
            record["joint_score"],
            record["optimal_joint_score"],
            record["score_ratio"],
            record["num_conflicts"],
        )

    def get_final_summary(self) -> Dict[str, Any]:
        joint_total = sum(r["joint_score"] for r in self.round_history)
        exact_opt = self.exact_optimum_benchmark()
        random_bench = self.random_uniform_benchmark()
        static_bench = self.static_uncalibrated_benchmark()
        max_joint = float(exact_opt["joint_score"])
        errors = [r["calibration_error"] for r in self.round_history]
        calibration_mae = round(sum(errors) / len(errors), 4) if errors else 0.0
        round_levels = [r["difficulty_level"] for r in self.round_history]
        converged = any(r["converged_this_round"] for r in self.round_history)

        return {
            "environment": self.__class__.__name__,
            "rounds_simulated": len(self.round_history),
            "initial_difficulty_level": self.initial_difficulty_level,
            "converged_difficulty_level": self.current_difficulty_level,
            "converged": converged,
            "target_score_ratio": self.target_score_ratio,
            "tolerance": self.tolerance,
            "difficulty_trajectory": list(self.difficulty_trajectory),
            "round_difficulty_levels": round_levels,
            "score_ratio_trajectory": list(self.score_ratio_trajectory),
            "calibration_error_mae": calibration_mae,
            "solvers": {
                name: {
                    "total_payoff": sum(
                        int(r["payoffs"].get(name, 0)) for r in self.round_history
                    ),
                    "total_conflicts": sum(
                        int(r["conflicts_by_agent"].get(name, 0)) for r in self.round_history
                    ),
                    "fallbacks_used": sum(
                        1
                        for r in self.round_history
                        if r["actions"].get(name, {}).get("fallback") is True
                    ),
                }
                for name in self.agent_names
            },
            "round_history": self.round_history,
            "benchmarks": {
                "exact_optimum": exact_opt,
                "random_uniform": random_bench,
                "static_uncalibrated": static_bench,
            },
            "joint_reward": joint_total,
            "max_joint_reward": max_joint,
            "joint_reward_ratio": round(joint_total / max_joint, 4) if max_joint > 0 else None,
        }

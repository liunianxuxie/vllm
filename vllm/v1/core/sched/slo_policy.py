# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any

from vllm.v1.request import Request
from vllm.v1.utils import record_function_or_nullcontext

SLO_PREFILL_LENGTH_BONUS = 10.0


def normalize_request_slo_ms(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed == 0:
        return float("inf")
    if math.isnan(parsed) or parsed < 0:
        return default
    return parsed


def set_request_slo_constraints(request: Request, scheduler_config: Any) -> None:
    request.ttft_slo_ms = scheduler_config.default_ttft_slo_ms
    if request.sampling_params is None or request.sampling_params.extra_args is None:
        return

    extra_args = request.sampling_params.extra_args
    ttft_slo_ms = extra_args.get("ttft_slo_ms")
    if ttft_slo_ms is not None:
        request.ttft_slo_ms = normalize_request_slo_ms(
            ttft_slo_ms,
            request.ttft_slo_ms,
        )


def compute_slo_score(
    request: Request,
    now: float,
    max_model_len: int,
    prefill_length_bonus: float,
) -> float:
    with record_function_or_nullcontext("slo_policy: compute_slo_score"):
        score = 0.0

        if request.first_token_ts is None:
            wait_time_ms = (now - request.arrival_time) * 1000
            if request.ttft_slo_ms < float("inf") and request.ttft_slo_ms > 0:
                score += 200.0 * max(0.0, wait_time_ms / request.ttft_slo_ms)

        if (
            request.num_computed_tokens == 0
            and request.num_prompt_tokens > 0
            and prefill_length_bonus > 0
        ):
            prompt_fraction = min(request.num_prompt_tokens / max_model_len, 1.0)
            score += max(0.0, prefill_length_bonus * (1.0 - prompt_fraction))

        return score


def is_waiting_reserve_candidate(
    request: Request,
    score: float,
    now: float,
    prefill_length_bonus: float,
) -> bool:
    if request.first_token_ts is not None or request.ttft_slo_ms <= 0:
        return False
    if request.ttft_slo_ms == float("inf"):
        return False
    wait_time_ms = (now - request.arrival_time) * 1000
    if wait_time_ms < 0.6 * request.ttft_slo_ms:
        return False
    return score >= 0.8 * prefill_length_bonus


def compute_waiting_token_reserve(
    max_num_scheduled_tokens: int,
    waiting_token_reserve_ratio: float,
    running_count: int,
    running_decode_count: int,
    reserve_candidates: list[tuple[float, int]],
) -> int:
    with record_function_or_nullcontext("slo_policy: compute_waiting_token_reserve"):
        if (
            waiting_token_reserve_ratio <= 0
            or not reserve_candidates
            or running_decode_count > max(1, running_count // 2)
        ):
            return 0

        reserve_cap = max(
            1,
            int(max_num_scheduled_tokens * waiting_token_reserve_ratio),
        )
        _score, remaining_prompt = max(reserve_candidates)
        return min(remaining_prompt, reserve_cap)

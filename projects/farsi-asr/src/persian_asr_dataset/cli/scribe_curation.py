from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from superwhisper_api.text.client import SuperwhisperClient, json_schema_response_format
from superwhisper_api.text.models import model_spec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    ClassifyOneFn = Callable[[sqlite3.Row, SuperwhisperClient, str, int], dict[str, Any]]
    _ClassifyResult = tuple[sqlite3.Row, dict[str, Any] | None, str | None]


DEFAULT_DATABASE = Path(
    "data/curation/scribe_jobs/scribe-canonical-all-20260516T192536Z/"
    "scribev2.full-20260523.sqlite"
)

SAME_SPEECH_TASK = "same_speech_script_v1"
ASR_RESCUE_TASK = "asr_rescue_v1"
ASR_RECOVERY_TASK = "asr_recovery_v1"
CONSENSUS_LABEL_TASK = "consensus_label_recovery_v1"
BOUNDARY_EXTRA_OMITTED_ASR_RECOVERY_TASK = "boundary_extra_omitted_asr_recovery_v1"
CONTENT_MISMATCH_ASR_RECOVERY_TASK = "content_mismatch_low_cer_asr_recovery_v1"
CONTENT_MISMATCH_MID_CER_ASR_RECOVERY_TASK = "content_mismatch_mid_cer_asr_recovery_v1"
CONTENT_MISMATCH_HIGH_CER_ASR_RECOVERY_TASK = (
    "content_mismatch_high_cer_20_30_asr_recovery_v1"
)
ORTHOGRAPHY_NEAR_HIGH_WER_ASR_RECOVERY_TASK = (
    "orthography_near_low_cer_high_wer_50_75_asr_recovery_v1"
)
ENTITY_SYMBOL_MID_CER_ASR_RECOVERY_TASK = "entity_symbol_mid_cer_15_25_asr_recovery_v1"
CONTENT_MISMATCH_LOW_CER_HIGH_WER_ASR_RECOVERY_TASK = (
    "content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1"
)
BOUNDARY_LOW_CER_HIGH_WER_ASR_RECOVERY_TASK = (
    "boundary_low_cer_high_wer_50_75_asr_recovery_v1"
)
BOUNDARY_MISMATCH_ASR_RECOVERY_TASK = "boundary_mismatch_asr_recovery_v1"
LARGE_MISMATCH_ASR_RECOVERY_TASK = "large_mismatch_low_cer_asr_recovery_v1"
HELD_STRONG_ASR_RECOVERY_V2_TASKS = (
    "content_mismatch_low_cer_held_strong_asr_recovery_v2",
    "boundary_extra_omitted_held_strong_asr_recovery_v2",
    "boundary_mismatch_held_strong_asr_recovery_v2",
    "content_mismatch_mid_cer_held_strong_asr_recovery_v2",
    "content_mismatch_low_cer_high_wer_50_75_held_strong_asr_recovery_v2",
    "content_mismatch_high_cer_20_30_held_strong_asr_recovery_v2",
    "entity_symbol_mid_cer_15_25_held_strong_asr_recovery_v2",
    "orthography_near_low_cer_high_wer_50_75_held_strong_asr_recovery_v2",
    "boundary_low_cer_high_wer_50_75_held_strong_asr_recovery_v2",
    "large_mismatch_low_cer_held_strong_asr_recovery_v2",
)
HELD_POSSIBLE_ASR_RECOVERY_V3_TASKS = (
    "content_mismatch_low_cer_held_possible_asr_recovery_v3",
    "boundary_extra_omitted_held_possible_asr_recovery_v3",
    "boundary_mismatch_held_possible_asr_recovery_v3",
    "content_mismatch_mid_cer_held_possible_asr_recovery_v3",
    "content_mismatch_low_cer_high_wer_50_75_held_possible_asr_recovery_v3",
    "content_mismatch_high_cer_20_30_held_possible_asr_recovery_v3",
    "entity_symbol_mid_cer_15_25_held_possible_asr_recovery_v3",
    "orthography_near_low_cer_high_wer_50_75_held_possible_asr_recovery_v3",
    "boundary_low_cer_high_wer_50_75_held_possible_asr_recovery_v3",
    "large_mismatch_low_cer_held_possible_asr_recovery_v3",
)
ASR_RECOVERY_EXPORT_TASKS = (
    BOUNDARY_EXTRA_OMITTED_ASR_RECOVERY_TASK,
    CONTENT_MISMATCH_ASR_RECOVERY_TASK,
    CONTENT_MISMATCH_MID_CER_ASR_RECOVERY_TASK,
    CONTENT_MISMATCH_HIGH_CER_ASR_RECOVERY_TASK,
    ORTHOGRAPHY_NEAR_HIGH_WER_ASR_RECOVERY_TASK,
    ENTITY_SYMBOL_MID_CER_ASR_RECOVERY_TASK,
    CONTENT_MISMATCH_LOW_CER_HIGH_WER_ASR_RECOVERY_TASK,
    BOUNDARY_LOW_CER_HIGH_WER_ASR_RECOVERY_TASK,
    BOUNDARY_MISMATCH_ASR_RECOVERY_TASK,
    LARGE_MISMATCH_ASR_RECOVERY_TASK,
    *HELD_STRONG_ASR_RECOVERY_V2_TASKS,
    *HELD_POSSIBLE_ASR_RECOVERY_V3_TASKS,
)
TEXT_RECOVERY_TASK = "text_recovery_v1"
MAX_CLASSIFY_ATTEMPTS = 4

SCRIPT_REVIEW_DECISIONS = {
    "same_speech_wrong_script",
    "same_speech_minor_mixed_script",
    "different_speech",
    "different_language",
    "annotation_or_noise",
    "unclear",
}

SCRIPT_REVIEW_ALIASES = {
    "same_speech_same_script": "same_speech_minor_mixed_script",
    "same_speech": "same_speech_minor_mixed_script",
}

ASR_RESCUE_DECISIONS = {
    "use_reference_asr_rescue",
    "hold_for_audio_review",
    "reject_reference_mismatch",
}

ASR_RECOVERY_DECISIONS = {
    "use_reference_asr_recovery",
    "hold_for_audio_review",
    "reject_reference_mismatch",
}

TEXT_RECOVERY_DECISIONS = {
    "use_reference_text_recovery",
    "hold_for_audio_review",
    "reject_reference_mismatch",
}

CONSENSUS_LABEL_DECISIONS = {
    "use_consensus_label",
    "hold_for_audio_review",
    "reject_consensus_label",
}

CONSENSUS_LABEL_LOW_RISK_CATEGORIES = (
    "punctuation_or_orthography_only",
    "near_match",
    "exact_match",
)

SCRIPT_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "decision": {"type": "string", "enum": sorted(SCRIPT_REVIEW_DECISIONS)},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["sample_id", "job_order", "decision", "reason", "evidence"],
    "additionalProperties": False,
}

ASR_RESCUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "decision": {"type": "string", "enum": sorted(ASR_RESCUE_DECISIONS)},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["sample_id", "job_order", "decision", "reason", "evidence"],
    "additionalProperties": False,
}

ASR_RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "decision": {"type": "string", "enum": sorted(ASR_RECOVERY_DECISIONS)},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["sample_id", "job_order", "decision", "reason", "evidence"],
    "additionalProperties": False,
}

TEXT_RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "decision": {"type": "string", "enum": sorted(TEXT_RECOVERY_DECISIONS)},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["sample_id", "job_order", "decision", "reason", "evidence"],
    "additionalProperties": False,
}

CONSENSUS_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "decision": {"type": "string", "enum": sorted(CONSENSUS_LABEL_DECISIONS)},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["sample_id", "job_order", "decision", "reason", "evidence"],
    "additionalProperties": False,
}

TEXT_RECOVERY_QUEUES = {
    "review_content_safe": """
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'near_match',
            'number_or_symbol_mismatch',
            'named_entity_mismatch',
            'exact_match'
        )
        AND resolved_cer <= 0.15
        AND resolved_wer <= 0.50
    """,
    "large_mismatch_low_cer": """
        row_decision = 'reject_large_mismatch'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'exact_match'
        )
        AND resolved_cer <= 0.10
    """,
}

ASR_RECOVERY_QUEUES = {
    "boundary_extra_omitted": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category IN ('extra_speech', 'omitted_speech')
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
    """,
    "content_mismatch_low_cer": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer <= 0.15
        AND resolved_wer <= 0.50
    """,
    "content_mismatch_mid_cer": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer > 0.15
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
    """,
    "content_mismatch_high_cer_20_30": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer > 0.20
        AND resolved_cer <= 0.30
        AND resolved_wer <= 0.50
    """,
    "orthography_near_low_cer_high_wer_50_75": """
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'near_match'
        )
        AND resolved_cer <= 0.15
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
    """,
    "entity_symbol_mid_cer_15_25": """
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'named_entity_mismatch',
            'number_or_symbol_mismatch'
        )
        AND resolved_cer > 0.15
        AND resolved_cer <= 0.25
        AND resolved_wer <= 0.50
    """,
    "content_mismatch_low_cer_high_wer_50_75": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
    """,
    "boundary_low_cer_high_wer_50_75": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category IN (
            'boundary_mismatch',
            'extra_speech',
            'omitted_speech'
        )
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
    """,
    "boundary_mismatch": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category = 'boundary_mismatch'
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
    """,
    "large_mismatch_low_cer": """
        row_decision = 'reject_large_mismatch'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'exact_match'
        )
        AND resolved_cer <= 0.10
    """,
    "held_boundary_extra_omitted": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category IN ('extra_speech', 'omitted_speech')
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'boundary_extra_omitted_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_content_mismatch_low_cer": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer <= 0.15
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'content_mismatch_low_cer_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_content_mismatch_mid_cer": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer > 0.15
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'content_mismatch_mid_cer_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_content_mismatch_high_cer_20_30": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer > 0.20
        AND resolved_cer <= 0.30
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'content_mismatch_high_cer_20_30_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_orthography_near_low_cer_high_wer_50_75": """
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'near_match'
        )
        AND resolved_cer <= 0.15
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'orthography_near_low_cer_high_wer_50_75_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_entity_symbol_mid_cer_15_25": """
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'named_entity_mismatch',
            'number_or_symbol_mismatch'
        )
        AND resolved_cer > 0.15
        AND resolved_cer <= 0.25
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'entity_symbol_mid_cer_15_25_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_content_mismatch_low_cer_high_wer_50_75": """
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_boundary_low_cer_high_wer_50_75": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category IN (
            'boundary_mismatch',
            'extra_speech',
            'omitted_speech'
        )
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'boundary_low_cer_high_wer_50_75_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_boundary_mismatch": """
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category = 'boundary_mismatch'
        AND resolved_cer <= 0.20
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'boundary_mismatch_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
    "held_large_mismatch_low_cer": """
        row_decision = 'reject_large_mismatch'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'exact_match'
        )
        AND resolved_cer <= 0.10
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.task = 'large_mismatch_low_cer_asr_recovery_v1'
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
        )
    """,
}

ASR_RECOVERY_FLAGS = (
    "asr_exact_reference",
    "asr_strong_reference",
    "asr_possible_reference",
    "asr_mixed_reference",
    "asr_reject_reference",
)

SURFACES = {
    "candidate_medium": "row_decision = 'candidate_medium_text'",
    "candidate_medium_supported": """
        row_decision = 'candidate_medium_text'
        AND COALESCE(audit_difference_category, '') NOT IN (
            'wrong_segment',
            'language_mismatch',
            'script_mismatch',
            'low_confidence_unclear'
        )
    """,
    "e0_exact": "row_decision = 'use_reference_same_normalized'",
    "e1_tiny": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta'
        )
    """,
    "e2_close": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta',
            'candidate_close_text'
        )
    """,
    "e3_broad": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta',
            'candidate_close_text',
            'candidate_medium_text'
        )
    """,
    "max_train_now": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta',
            'candidate_close_text',
            'use_reference_audit_safe_current_gate'
        )
    """,
    "max_train_metric_audit": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
    """,
    "max_train_supported_script": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta',
            'candidate_close_text',
            'use_reference_audit_safe_current_gate'
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
    """,
    "max_train_metric_supported_script": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
    """,
    "max_train_metric_supported_script_rescue": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
    """,
    "max_train_metric_supported_script_rescue_text": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
        OR (
            EXISTS (
                SELECT 1
                FROM scribe_same_speech_review text_recovery
                WHERE text_recovery.sample_id = scribe_training_decisions.sample_id
                    AND text_recovery.job_order = scribe_training_decisions.job_order
                    AND text_recovery.task = 'text_recovery_v1'
                    AND text_recovery.model = 'gpt-5.4-mini'
                    AND text_recovery.decision = 'use_reference_text_recovery'
            )
        )
    """,
    "max_train_metric_supported_script_rescue_text_boundary_asr": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
        OR (
            EXISTS (
                SELECT 1
                FROM scribe_same_speech_review text_recovery
                WHERE text_recovery.sample_id = scribe_training_decisions.sample_id
                    AND text_recovery.job_order = scribe_training_decisions.job_order
                    AND text_recovery.task = 'text_recovery_v1'
                    AND text_recovery.model = 'gpt-5.4-mini'
                    AND text_recovery.decision = 'use_reference_text_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category IN ('extra_speech', 'omitted_speech')
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_extra_omitted_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
    """,
    "max_train_metric_supported_script_rescue_text_asr_recovery": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
        OR (
            EXISTS (
                SELECT 1
                FROM scribe_same_speech_review text_recovery
                WHERE text_recovery.sample_id = scribe_training_decisions.sample_id
                    AND text_recovery.job_order = scribe_training_decisions.job_order
                    AND text_recovery.task = 'text_recovery_v1'
                    AND text_recovery.model = 'gpt-5.4-mini'
                    AND text_recovery.decision = 'use_reference_text_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category IN ('extra_speech', 'omitted_speech')
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_extra_omitted_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category = 'boundary_mismatch'
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_mismatch_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer <= 0.15
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_large_mismatch'
            AND audit_difference_category IN (
                'punctuation_or_orthography_only',
                'exact_match'
            )
            AND resolved_cer <= 0.10
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'large_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
    """,
    "max_train_metric_supported_script_rescue_text_asr_recovery_content20": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
        OR (
            EXISTS (
                SELECT 1
                FROM scribe_same_speech_review text_recovery
                WHERE text_recovery.sample_id = scribe_training_decisions.sample_id
                    AND text_recovery.job_order = scribe_training_decisions.job_order
                    AND text_recovery.task = 'text_recovery_v1'
                    AND text_recovery.model = 'gpt-5.4-mini'
                    AND text_recovery.decision = 'use_reference_text_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category IN ('extra_speech', 'omitted_speech')
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_extra_omitted_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category = 'boundary_mismatch'
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_mismatch_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer <= 0.15
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer > 0.15
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_mid_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_large_mismatch'
            AND audit_difference_category IN (
                'punctuation_or_orthography_only',
                'exact_match'
            )
            AND resolved_cer <= 0.10
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'large_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
    """,
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30": """
        (
            row_decision IN (
                'use_reference_same_normalized',
                'use_reference_tiny_delta',
                'candidate_close_text',
                'use_reference_audit_safe_current_gate'
            )
            OR (
                row_decision = 'candidate_medium_text'
                AND COALESCE(audit_difference_category, '') NOT IN (
                    'wrong_segment',
                    'language_mismatch',
                    'script_mismatch',
                    'low_confidence_unclear'
                )
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_asr_review_flags flag
                WHERE flag.sample_id = scribe_training_decisions.sample_id
                    AND flag.job_order = scribe_training_decisions.job_order
                    AND flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                    AND flag.asr_review_flag = 'llm_accept_supported_by_300m'
            )
        )
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review rescue
                WHERE rescue.sample_id = scribe_training_decisions.sample_id
                    AND rescue.job_order = scribe_training_decisions.job_order
                    AND rescue.task = 'asr_rescue_v1'
                    AND rescue.model = 'gpt-5.4-mini'
                    AND rescue.decision = 'use_reference_asr_rescue'
            )
        )
        OR (
            EXISTS (
                SELECT 1
                FROM scribe_same_speech_review text_recovery
                WHERE text_recovery.sample_id = scribe_training_decisions.sample_id
                    AND text_recovery.job_order = scribe_training_decisions.job_order
                    AND text_recovery.task = 'text_recovery_v1'
                    AND text_recovery.model = 'gpt-5.4-mini'
                    AND text_recovery.decision = 'use_reference_text_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category IN ('extra_speech', 'omitted_speech')
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_extra_omitted_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_boundary_or_length'
            AND audit_difference_category = 'boundary_mismatch'
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'boundary_mismatch_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer <= 0.15
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer > 0.15
            AND resolved_cer <= 0.20
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_mid_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'review_content'
            AND audit_difference_category = 'content_mismatch'
            AND resolved_cer > 0.20
            AND resolved_cer <= 0.30
            AND resolved_wer <= 0.50
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'content_mismatch_high_cer_20_30_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
        OR (
            row_decision = 'reject_large_mismatch'
            AND audit_difference_category IN (
                'punctuation_or_orthography_only',
                'exact_match'
            )
            AND resolved_cer <= 0.10
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review asr_recovery
                WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                    AND asr_recovery.job_order = scribe_training_decisions.job_order
                    AND asr_recovery.task = 'large_mismatch_low_cer_asr_recovery_v1'
                    AND asr_recovery.model = 'gpt-5.4-mini'
                    AND asr_recovery.decision = 'use_reference_asr_recovery'
            )
        )
    """,
    "script_queue": "row_decision = 'needs_same_speech_script_check'",
    "max_reviewable": """
        row_decision IN (
            'use_reference_same_normalized',
            'use_reference_tiny_delta',
            'candidate_close_text',
            'candidate_medium_text',
            'use_reference_audit_safe_current_gate',
            'needs_same_speech_script_check',
            'review_content'
        )
    """,
}

SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer"] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30"]})
    OR (
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'punctuation_or_orthography_only',
            'near_match'
        )
        AND resolved_cer <= 0.15
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review asr_recovery
            WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                AND asr_recovery.job_order = scribe_training_decisions.job_order
                AND asr_recovery.task = 'orthography_near_low_cer_high_wer_50_75_asr_recovery_v1'
                AND asr_recovery.model = 'gpt-5.4-mini'
                AND asr_recovery.decision = 'use_reference_asr_recovery'
        )
    )
    OR (
        row_decision = 'review_content'
        AND audit_difference_category IN (
            'named_entity_mismatch',
            'number_or_symbol_mismatch'
        )
        AND resolved_cer > 0.15
        AND resolved_cer <= 0.25
        AND resolved_wer <= 0.50
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review asr_recovery
            WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                AND asr_recovery.job_order = scribe_training_decisions.job_order
                AND asr_recovery.task = 'entity_symbol_mid_cer_15_25_asr_recovery_v1'
                AND asr_recovery.model = 'gpt-5.4-mini'
                AND asr_recovery.decision = 'use_reference_asr_recovery'
        )
    )
    OR (
        row_decision = 'review_content'
        AND audit_difference_category = 'content_mismatch'
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review asr_recovery
            WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                AND asr_recovery.job_order = scribe_training_decisions.job_order
                AND asr_recovery.task = 'content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1'
                AND asr_recovery.model = 'gpt-5.4-mini'
                AND asr_recovery.decision = 'use_reference_asr_recovery'
        )
    )
    OR (
        row_decision = 'reject_boundary_or_length'
        AND audit_difference_category IN (
            'boundary_mismatch',
            'extra_speech',
            'omitted_speech'
        )
        AND resolved_cer <= 0.20
        AND resolved_wer > 0.50
        AND resolved_wer <= 0.75
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review asr_recovery
            WHERE asr_recovery.sample_id = scribe_training_decisions.sample_id
                AND asr_recovery.job_order = scribe_training_decisions.job_order
                AND asr_recovery.task = 'boundary_low_cer_high_wer_50_75_asr_recovery_v1'
                AND asr_recovery.model = 'gpt-5.4-mini'
                AND asr_recovery.decision = 'use_reference_asr_recovery'
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer"]})
    OR (
        row_decision = 'needs_same_speech_script_check'
        AND EXISTS (
            SELECT 1
            FROM scribe_same_speech_review script_review
            WHERE script_review.sample_id = scribe_training_decisions.sample_id
                AND script_review.job_order = scribe_training_decisions.job_order
                AND script_review.task = 'same_speech_script_v1'
                AND script_review.model = 'gpt-5.4-mini'
                AND script_review.decision IN (
                    'same_speech_wrong_script',
                    'same_speech_minor_mixed_script'
                )
        )
        AND EXISTS (
            SELECT 1
            FROM scribe_asr_review_flags script_flag
            WHERE script_flag.sample_id = scribe_training_decisions.sample_id
                AND script_flag.job_order = scribe_training_decisions.job_order
                AND script_flag.asr_model_label = 'omni300_exact_best_script_accepted_full'
                AND script_flag.asr_review_flag IN (
                    'llm_accept_unconfirmed_by_300m',
                    'accepted_needs_audio_review'
                )
        )
        AND EXISTS (
            SELECT 1
            FROM scribe_asr_crosscheck script_asr
            WHERE script_asr.sample_id = scribe_training_decisions.sample_id
                AND script_asr.job_order = scribe_training_decisions.job_order
                AND script_asr.model_label = 'omni300_exact_best_script_accepted_full'
                AND script_asr.cer <= 0.10
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review held_review
            JOIN scribe_asr_crosscheck held_asr
                ON held_asr.sample_id = held_review.sample_id
                AND held_asr.job_order = held_review.job_order
            WHERE held_review.sample_id = scribe_training_decisions.sample_id
                AND held_review.job_order = scribe_training_decisions.job_order
                AND held_review.model = 'gpt-5.4-mini'
                AND held_review.decision = 'hold_for_audio_review'
                AND held_asr.wer = 0
                AND held_asr.cer = 0
                AND (
                    (
                        held_review.task = 'boundary_extra_omitted_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_boundary_extra_omitted_full'
                    )
                    OR (
                        held_review.task = 'boundary_mismatch_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_boundary_mismatch_full'
                    )
                    OR (
                        held_review.task = 'content_mismatch_low_cer_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_content_mismatch_low_cer_lte40'
                    )
                    OR (
                        held_review.task = 'content_mismatch_mid_cer_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_content_mismatch_mid_cer_full'
                    )
                    OR (
                        held_review.task = 'content_mismatch_high_cer_20_30_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_content_mismatch_high_cer_20_30_full'
                    )
                    OR (
                        held_review.task = 'orthography_near_low_cer_high_wer_50_75_asr_recovery_v1'
                        AND held_asr.model_label =
                            'omni300_orthography_near_low_cer_high_wer_50_75_full'
                    )
                    OR (
                        held_review.task = 'entity_symbol_mid_cer_15_25_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_entity_symbol_mid_cer_15_25_full'
                    )
                    OR (
                        held_review.task = 'content_mismatch_low_cer_high_wer_50_75_asr_recovery_v1'
                        AND held_asr.model_label =
                            'omni300_content_mismatch_low_cer_high_wer_50_75_full'
                    )
                    OR (
                        held_review.task = 'boundary_low_cer_high_wer_50_75_asr_recovery_v1'
                        AND held_asr.model_label =
                            'omni300_boundary_low_cer_high_wer_50_75_full'
                    )
                    OR (
                        held_review.task = 'large_mismatch_low_cer_asr_recovery_v1'
                        AND held_asr.model_label = 'omni300_large_mismatch_low_cer_full'
                    )
                )
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_v2"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review second_pass
            WHERE second_pass.sample_id = scribe_training_decisions.sample_id
                AND second_pass.job_order = scribe_training_decisions.job_order
                AND second_pass.task = 'content_mismatch_low_cer_held_strong_asr_recovery_v2'
                AND second_pass.model = 'gpt-5.4-mini'
                AND second_pass.decision = 'use_reference_asr_recovery'
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_boundary_v2"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_v2"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review second_pass
            WHERE second_pass.sample_id = scribe_training_decisions.sample_id
                AND second_pass.job_order = scribe_training_decisions.job_order
                AND second_pass.task = 'boundary_extra_omitted_held_strong_asr_recovery_v2'
                AND second_pass.model = 'gpt-5.4-mini'
                AND second_pass.decision = 'use_reference_asr_recovery'
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_strong_v2"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_boundary_v2"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review second_pass
            WHERE second_pass.sample_id = scribe_training_decisions.sample_id
                AND second_pass.job_order = scribe_training_decisions.job_order
                AND second_pass.task IN (
                    'boundary_mismatch_held_strong_asr_recovery_v2',
                    'content_mismatch_mid_cer_held_strong_asr_recovery_v2',
                    'content_mismatch_low_cer_high_wer_50_75_held_strong_asr_recovery_v2',
                    'content_mismatch_high_cer_20_30_held_strong_asr_recovery_v2',
                    'entity_symbol_mid_cer_15_25_held_strong_asr_recovery_v2',
                    'orthography_near_low_cer_high_wer_50_75_held_strong_asr_recovery_v2',
                    'boundary_low_cer_high_wer_50_75_held_strong_asr_recovery_v2',
                    'large_mismatch_low_cer_held_strong_asr_recovery_v2'
                )
                AND second_pass.model = 'gpt-5.4-mini'
                AND second_pass.decision = 'use_reference_asr_recovery'
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_held_strong_v2"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review possible_pass
            WHERE possible_pass.sample_id = scribe_training_decisions.sample_id
                AND possible_pass.job_order = scribe_training_decisions.job_order
                AND possible_pass.task IN (
                    'content_mismatch_low_cer_held_possible_asr_recovery_v3',
                    'boundary_extra_omitted_held_possible_asr_recovery_v3',
                    'boundary_mismatch_held_possible_asr_recovery_v3',
                    'content_mismatch_mid_cer_held_possible_asr_recovery_v3',
                    'content_mismatch_low_cer_high_wer_50_75_held_possible_asr_recovery_v3',
                    'content_mismatch_high_cer_20_30_held_possible_asr_recovery_v3',
                    'entity_symbol_mid_cer_15_25_held_possible_asr_recovery_v3',
                    'orthography_near_low_cer_high_wer_50_75_held_possible_asr_recovery_v3',
                    'boundary_low_cer_high_wer_50_75_held_possible_asr_recovery_v3',
                    'large_mismatch_low_cer_held_possible_asr_recovery_v3'
                )
                AND possible_pass.model = 'gpt-5.4-mini'
                AND possible_pass.decision = 'use_reference_asr_recovery'
        )
    )
"""

SURFACES[
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_consensus_lowrisk_v1"
] = f"""
    ({SURFACES["max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3"]})
    OR (
        EXISTS (
            SELECT 1
            FROM scribe_same_speech_review consensus_review
            WHERE consensus_review.sample_id = scribe_training_decisions.sample_id
                AND consensus_review.job_order = scribe_training_decisions.job_order
                AND consensus_review.task = 'consensus_label_recovery_v1'
                AND consensus_review.model = 'gpt-5.4-mini'
                AND consensus_review.decision = 'use_consensus_label'
                AND scribe_training_decisions.resolved_cer < 0.15
                AND scribe_training_decisions.audit_difference_category IN (
                    'punctuation_or_orthography_only',
                    'near_match',
                    'exact_match'
                )
                AND TRIM(COALESCE(scribe_training_decisions.resolved_normalized_scribe, '')) <> ''
                AND scribe_training_decisions.normalized_reference
                    <> scribe_training_decisions.resolved_normalized_scribe
                AND EXISTS (
                    SELECT 1
                    FROM scribe_asr_crosscheck consensus_asr
                    WHERE consensus_asr.sample_id = scribe_training_decisions.sample_id
                        AND consensus_asr.job_order = scribe_training_decisions.job_order
                        AND consensus_asr.normalized_hypothesis
                            = scribe_training_decisions.resolved_normalized_scribe
                )
        )
    )
"""

SURFACE_LABELS = {
    "candidate_medium": "Candidate medium, CER <= 10% and WER <= 35%",
    "candidate_medium_supported": "Candidate medium without stale high-risk old labels",
    "e0_exact": "E0 exact current",
    "e1_tiny": "E1 exact plus CER <= 2%",
    "e2_close": "E2 close, CER <= 5% and WER <= 20%",
    "e3_broad": "E3 broad, CER <= 10% and WER <= 35%",
    "max_train_now": "Maximalist train-now, current metric plus audit-safe gate",
    "max_train_metric_audit": "Maximalist metric plus audit-safe gate",
    "max_train_supported_script": "Maximalist plus script rows supported by 300M",
    "max_train_metric_supported_script": "Maximalist metric plus 300M-supported script rows",
    "max_train_metric_supported_script_rescue": (
        "Maximalist metric plus 300M-supported script and rescue rows"
    ),
    "max_train_metric_supported_script_rescue_text": (
        "Maximalist metric plus script, rescue, and text-recovery rows"
    ),
    "max_train_metric_supported_script_rescue_text_boundary_asr": (
        "Maximalist metric plus script, rescue, text-recovery, and boundary ASR rows"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery": (
        "Maximalist metric plus script, text, and ASR recovery rows"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content20": (
        "Maximalist metric plus script, text, ASR recovery, and content CER <= 20%"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30": (
        "Maximalist metric plus script, text, ASR recovery, and content CER <= 30%"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer": (
        "Content CER <= 30% plus ASR-supported high-WER rescue slices"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script": (
        "High-WER rescue plus strict script rows accepted by LLM with 300M CER <= 10%"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold": (  # noqa: E501
        "Strict script rescue plus held rows where 300M exact-matches the reference"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_v2": (  # noqa: E501
        "Exact-held surface plus accepted low-CER content held-row second pass"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_content30_highwer_strict_script_exact_hold_content_low_boundary_v2": (  # noqa: E501
        "Content-low v2 surface plus accepted boundary held-row second pass"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_strong_v2": (
        "Boundary-v2 surface plus all accepted held strong ASR recovery rows"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_v3": (
        "Held-strong-v2 surface plus accepted possible ASR recovery rows"
    ),
    "max_train_metric_supported_script_rescue_text_asr_recovery_held_possible_consensus_lowrisk_v1": (  # noqa: E501
        "Held-possible-v3 surface plus low-risk Scribe and 300M consensus labels"
    ),
    "script_queue": "E5 Scribe Latin/script same-speech queue",
    "max_reviewable": "Max reviewable surface",
}


@dataclass(frozen=True)
class ViewStats:
    database: str
    created_views: list[str]
    created_tables: list[str]


@dataclass(frozen=True)
class ExportStats:
    surface: str
    output_root: str
    rows_written: int
    hours_written: float
    split_counts: dict[str, int]
    split_hours: dict[str, float]


@dataclass(frozen=True)
class ReviewStats:
    database: str
    task: str
    model: str
    selected: int
    reviewed: int
    failed: int
    skipped_existing: int


@dataclass(frozen=True)
class ReviewManifestStats:
    output: str
    task: str
    model: str
    decisions: list[str]
    rows_written: int
    hours_written: float


@dataclass(frozen=True)
class AsrQueueManifestStats:
    output: str
    queue: str
    rows_written: int
    hours_written: float
    max_duration: float | None
    skipped_existing_asr_rows: int


@dataclass(frozen=True)
class AsrCheckImportStats:
    database: str
    model_label: str
    rows_seen: int
    rows_imported: int
    missing_manifest_rows: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.expanduser(), timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _quote_sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def require_surface(name: str) -> str:
    try:
        return SURFACES[name]
    except KeyError as exc:
        known = ", ".join(sorted(SURFACES))
        raise SystemExit(f"unknown surface {name!r}; known surfaces: {known}") from exc


def resolved_pairs_view_sql() -> str:
    return """
    CREATE VIEW IF NOT EXISTS scribe_resolved_pairs AS
    WITH latest_rerun AS (
        SELECT
            result_id,
            batch_id,
            sample_id,
            job_order,
            transcript,
            normalized_transcript,
            wer,
            cer,
            raw_response_json,
            created_at,
            ROW_NUMBER() OVER (
                PARTITION BY sample_id, job_order
                ORDER BY created_at DESC, result_id DESC
            ) AS rn
        FROM scribe_rerun_results
    )
    SELECT
        curation.sample_id,
        curation.job_order,
        curation.job_id,
        curation.source,
        curation.source_config,
        curation.canonical_dataset,
        curation.project_split,
        curation.original_split,
        curation.audio_filepath,
        curation.audio_sha256,
        curation.duration_seconds,
        curation.raw_reference_text,
        curation.raw_scribe_text AS original_raw_scribe_text,
        CASE
            WHEN latest_rerun.rn = 1 THEN latest_rerun.transcript
            ELSE curation.raw_scribe_text
        END AS resolved_raw_scribe_text,
        curation.normalized_reference,
        curation.normalized_scribe AS original_normalized_scribe,
        CASE
            WHEN latest_rerun.rn = 1 THEN COALESCE(latest_rerun.normalized_transcript, '')
            ELSE COALESCE(curation.normalized_scribe, '')
        END AS resolved_normalized_scribe,
        curation.wer AS original_wer,
        curation.cer AS original_cer,
        CASE
            WHEN latest_rerun.rn = 1 THEN latest_rerun.wer
            ELSE curation.wer
        END AS resolved_wer,
        CASE
            WHEN latest_rerun.rn = 1 THEN latest_rerun.cer
            ELSE curation.cer
        END AS resolved_cer,
        curation.exact_match AS original_exact_match,
        curation.empty_normalized_reference,
        curation.missing_scribe,
        curation.empty_scribe,
        curation.api_eligible,
        curation.difference_category AS curation_difference_category,
        audit.difference_category AS audit_difference_category,
        audit.difference_description AS audit_difference_description,
        audit.likely_cause AS audit_likely_cause,
        audit.suggested_action AS audit_suggested_action,
        CASE WHEN latest_rerun.rn = 1 THEN 1 ELSE 0 END AS used_rerun,
        latest_rerun.result_id AS rerun_result_id,
        latest_rerun.batch_id AS rerun_batch_id,
        latest_rerun.raw_response_json AS rerun_response_json,
        CASE WHEN curation.raw_reference_text GLOB '*[A-Za-z]*' THEN 1 ELSE 0 END
            AS latin_in_reference,
        CASE
            WHEN (
                CASE
                    WHEN latest_rerun.rn = 1 THEN latest_rerun.transcript
                    ELSE curation.raw_scribe_text
                END
            ) GLOB '*[A-Za-z]*'
            THEN 1
            ELSE 0
        END AS latin_in_scribe,
        CASE
            WHEN instr(COALESCE(curation.raw_reference_text, ''), '[') > 0
                OR instr(COALESCE(curation.raw_reference_text, ''), ']') > 0
                OR instr(COALESCE(curation.raw_reference_text, ''), '(') > 0
                OR instr(COALESCE(curation.raw_reference_text, ''), ')') > 0
                OR instr(COALESCE(curation.raw_reference_text, ''), '{') > 0
                OR instr(COALESCE(curation.raw_reference_text, ''), '}') > 0
            THEN 1
            ELSE 0
        END AS bracket_in_reference,
        CASE
            WHEN instr(
                COALESCE(
                    CASE
                        WHEN latest_rerun.rn = 1 THEN latest_rerun.transcript
                        ELSE curation.raw_scribe_text
                    END,
                    ''
                ),
                '['
            ) > 0
                OR instr(
                    COALESCE(
                        CASE
                            WHEN latest_rerun.rn = 1 THEN latest_rerun.transcript
                            ELSE curation.raw_scribe_text
                        END,
                        ''
                    ),
                    ']'
                ) > 0
            THEN 1
            ELSE 0
        END AS bracket_in_scribe
    FROM scribe_curation curation
    LEFT JOIN latest_rerun
        ON latest_rerun.sample_id = curation.sample_id
        AND latest_rerun.job_order = curation.job_order
        AND latest_rerun.rn = 1
    LEFT JOIN scribe_audit audit
        ON audit.sample_id = curation.sample_id
        AND audit.job_order = curation.job_order
    """


def decisions_view_sql() -> str:
    return """
    CREATE VIEW IF NOT EXISTS scribe_training_decisions AS
    SELECT
        pairs.*,
        CASE
            WHEN TRIM(COALESCE(normalized_reference, '')) = ''
                THEN 'drop_empty_reference'
            WHEN TRIM(COALESCE(resolved_normalized_scribe, '')) = ''
                AND latin_in_scribe = 1
                THEN 'needs_same_speech_script_check'
            WHEN TRIM(COALESCE(resolved_normalized_scribe, '')) = ''
                THEN 'drop_scribe_empty'
            WHEN normalized_reference = resolved_normalized_scribe
                THEN 'use_reference_same_normalized'
            WHEN resolved_cer <= 0.02
                THEN 'use_reference_tiny_delta'
            WHEN resolved_cer <= 0.05 AND resolved_wer <= 0.20
                THEN 'candidate_close_text'
            WHEN audit_difference_category IN (
                    'punctuation_or_orthography_only',
                    'near_match',
                    'number_or_symbol_mismatch',
                    'named_entity_mismatch'
                )
                AND resolved_cer <= 0.10
                AND resolved_wer <= 0.35
                THEN 'use_reference_audit_safe_current_gate'
            WHEN resolved_cer <= 0.10 AND resolved_wer <= 0.35
                THEN 'candidate_medium_text'
            WHEN audit_difference_category IN (
                'boundary_mismatch',
                'extra_speech',
                'omitted_speech'
            )
                THEN 'reject_boundary_or_length'
            WHEN audit_difference_category IN ('wrong_segment', 'language_mismatch')
                THEN 'reject_wrong_audio_or_language'
            WHEN audit_difference_category IN (
                'non_speech_annotation',
                'speaker_label_or_annotation'
            )
                THEN 'reject_annotation'
            WHEN resolved_cer > 0.30 OR resolved_wer > 0.75
                THEN 'reject_large_mismatch'
            ELSE 'review_content'
        END AS row_decision,
        CASE
            WHEN TRIM(COALESCE(normalized_reference, '')) = '' THEN 0
            WHEN TRIM(COALESCE(audio_filepath, '')) = '' THEN 0
            WHEN normalized_reference = resolved_normalized_scribe THEN 1
            WHEN resolved_cer <= 0.02 THEN 1
            WHEN resolved_cer <= 0.05 AND resolved_wer <= 0.20 THEN 1
            WHEN audit_difference_category IN (
                    'punctuation_or_orthography_only',
                    'near_match',
                    'number_or_symbol_mismatch',
                    'named_entity_mismatch'
                )
                AND resolved_cer <= 0.10
                AND resolved_wer <= 0.35
                THEN 1
            WHEN resolved_cer <= 0.10 AND resolved_wer <= 0.35 THEN 1
            ELSE 0
        END AS metric_manifest_eligible,
        CASE
            WHEN TRIM(COALESCE(normalized_reference, '')) = '' THEN NULL
            ELSE normalized_reference
        END AS training_text,
        CASE
            WHEN normalized_reference = resolved_normalized_scribe
                THEN 'normalized_reference_exact'
            WHEN resolved_cer <= 0.02
                THEN 'normalized_reference_tiny_delta'
            WHEN resolved_cer <= 0.05 AND resolved_wer <= 0.20
                THEN 'normalized_reference_close_delta'
            WHEN audit_difference_category IN (
                    'punctuation_or_orthography_only',
                    'near_match',
                    'number_or_symbol_mismatch',
                    'named_entity_mismatch'
                )
                AND resolved_cer <= 0.10
                AND resolved_wer <= 0.35
                THEN 'normalized_reference_audit_safe_current_gate'
            WHEN resolved_cer <= 0.10 AND resolved_wer <= 0.35
                THEN 'normalized_reference_medium_delta'
            ELSE NULL
        END AS training_text_source
    FROM scribe_resolved_pairs pairs
    """


def asr_review_flags_view_sql() -> str:
    return f"""
    CREATE VIEW IF NOT EXISTS scribe_asr_review_flags AS
    SELECT
        decisions.sample_id,
        decisions.job_order,
        decisions.audio_filepath,
        decisions.duration_seconds,
        decisions.normalized_reference,
        decisions.resolved_raw_scribe_text,
        decisions.resolved_normalized_scribe,
        decisions.audit_difference_category,
        review.task AS review_task,
        review.model AS review_model,
        review.decision AS script_review_decision,
        review.reason AS script_review_reason,
        review.evidence AS script_review_evidence,
        asr.model_label AS asr_model_label,
        asr.normalized_hypothesis AS asr_normalized_hypothesis,
        asr.wer AS asr_wer,
        asr.cer AS asr_cer,
        CASE
            WHEN review.decision IN (
                    'same_speech_wrong_script',
                    'same_speech_minor_mixed_script'
                )
                THEN 'llm_accept'
            ELSE 'llm_reject'
        END AS script_review_group,
        CASE
            WHEN review.decision IN (
                    'same_speech_wrong_script',
                    'same_speech_minor_mixed_script'
                )
                AND asr.wer <= 0.50
                AND asr.cer <= 0.15
                THEN 'llm_accept_supported_by_300m'
            WHEN review.decision IN (
                    'same_speech_wrong_script',
                    'same_speech_minor_mixed_script'
                )
                AND (asr.wer >= 0.75 OR asr.cer >= 0.25)
                THEN 'accepted_needs_audio_review'
            WHEN review.decision IN (
                    'same_speech_wrong_script',
                    'same_speech_minor_mixed_script'
                )
                THEN 'llm_accept_unconfirmed_by_300m'
            WHEN asr.wer = 0 AND asr.cer = 0
                THEN 'strong_rescue_review'
            WHEN asr.wer <= 0.35 AND asr.cer <= 0.10
                THEN 'weak_rescue_review'
            WHEN asr.wer >= 0.75 OR asr.cer >= 0.25
                THEN 'reject_supported_by_300m'
            ELSE 'reject_unconfirmed_by_300m'
        END AS asr_review_flag
    FROM scribe_training_decisions decisions
    JOIN scribe_same_speech_review review
        ON review.sample_id = decisions.sample_id
        AND review.job_order = decisions.job_order
        AND review.task = '{SAME_SPEECH_TASK}'
    JOIN scribe_asr_crosscheck asr
        ON asr.sample_id = decisions.sample_id
        AND asr.job_order = decisions.job_order
    """


def ensure_review_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scribe_same_speech_review (
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            task TEXT NOT NULL,
            model TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            evidence TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, job_order, task, model),
            FOREIGN KEY (sample_id, job_order) REFERENCES scribe_curation(sample_id, job_order)
        );

        CREATE INDEX IF NOT EXISTS idx_scribe_same_speech_review_decision
            ON scribe_same_speech_review(task, model, decision);
        """
    )


def ensure_asr_crosscheck_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scribe_asr_crosscheck (
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            model_label TEXT NOT NULL,
            source_manifest TEXT NOT NULL,
            benchmark_samples TEXT NOT NULL,
            reference_text TEXT NOT NULL,
            hypothesis_text TEXT NOT NULL,
            normalized_reference TEXT NOT NULL,
            normalized_hypothesis TEXT NOT NULL,
            wer REAL NOT NULL,
            cer REAL NOT NULL,
            duration_seconds REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, job_order, model_label),
            FOREIGN KEY (sample_id, job_order) REFERENCES scribe_curation(sample_id, job_order)
        );

        CREATE INDEX IF NOT EXISTS idx_scribe_asr_crosscheck_model
            ON scribe_asr_crosscheck(model_label, wer, cer);
        """
    )


def install_views(connection: sqlite3.Connection) -> ViewStats:
    connection.execute("DROP VIEW IF EXISTS scribe_asr_review_flags")
    connection.execute("DROP VIEW IF EXISTS scribe_training_decisions")
    connection.execute("DROP VIEW IF EXISTS scribe_resolved_pairs")
    connection.executescript(resolved_pairs_view_sql())
    connection.executescript(decisions_view_sql())
    ensure_review_table(connection)
    ensure_asr_crosscheck_table(connection)
    connection.executescript(asr_review_flags_view_sql())
    database = connection.execute("PRAGMA database_list").fetchone()["file"]
    return ViewStats(
        database=str(database),
        created_views=[
            "scribe_resolved_pairs",
            "scribe_training_decisions",
            "scribe_asr_review_flags",
        ],
        created_tables=["scribe_same_speech_review", "scribe_asr_crosscheck"],
    )


def fetch_grouped(
    connection: sqlite3.Connection,
    query: str,
    params: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, tuple(params)).fetchall()]


def decision_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return fetch_grouped(
        connection,
        """
        SELECT
            row_decision AS decision,
            COUNT(*) AS rows,
            ROUND(SUM(duration_seconds) / 3600.0, 2) AS hours,
            ROUND(AVG(resolved_wer), 4) AS avg_wer,
            ROUND(AVG(resolved_cer), 4) AS avg_cer
        FROM scribe_training_decisions
        GROUP BY row_decision
        ORDER BY
            CASE row_decision
                WHEN 'use_reference_same_normalized' THEN 1
                WHEN 'use_reference_tiny_delta' THEN 2
                WHEN 'candidate_close_text' THEN 3
                WHEN 'use_reference_audit_safe_current_gate' THEN 4
                WHEN 'candidate_medium_text' THEN 5
                WHEN 'needs_same_speech_script_check' THEN 6
                WHEN 'review_content' THEN 7
                WHEN 'reject_boundary_or_length' THEN 8
                WHEN 'reject_wrong_audio_or_language' THEN 9
                WHEN 'reject_annotation' THEN 10
                WHEN 'reject_large_mismatch' THEN 11
                WHEN 'drop_scribe_empty' THEN 12
                WHEN 'drop_empty_reference' THEN 13
                ELSE 99
            END
        """,
    )


def surface_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = []
    for surface, where_sql in SURFACES.items():
        payload = connection.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                ROUND(SUM(duration_seconds) / 3600.0, 2) AS hours,
                ROUND(AVG(resolved_wer), 4) AS avg_wer,
                ROUND(AVG(resolved_cer), 4) AS avg_cer
            FROM scribe_training_decisions
            WHERE {where_sql}
            """
        ).fetchone()
        rows.append(
            {
                "surface": surface,
                "label": SURFACE_LABELS[surface],
                "rows": int(payload["rows"] or 0),
                "hours": payload["hours"],
                "avg_wer": payload["avg_wer"],
                "avg_cer": payload["avg_cer"],
            }
        )
    return rows


def script_queue_breakdown(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return fetch_grouped(
        connection,
        """
        SELECT
            COALESCE(audit_difference_category, 'missing_audit') AS audit_category,
            COUNT(*) AS rows
        FROM scribe_training_decisions
        WHERE row_decision = 'needs_same_speech_script_check'
        GROUP BY audit_category
        ORDER BY rows DESC
        """,
    )


def review_counts(
    connection: sqlite3.Connection,
    task: str,
    model: str | None,
) -> list[dict[str, Any]]:
    if model:
        return fetch_grouped(
            connection,
            """
            SELECT task, model, decision, COUNT(*) AS rows
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            GROUP BY task, model, decision
            ORDER BY rows DESC
            """,
            (task, model),
        )
    return fetch_grouped(
        connection,
        """
        SELECT task, model, decision, COUNT(*) AS rows
        FROM scribe_same_speech_review
        WHERE task = ?
        GROUP BY task, model, decision
        ORDER BY model, rows DESC
        """,
        (task,),
    )


def asr_crosscheck_counts(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return fetch_grouped(
        connection,
        """
        SELECT
            model_label,
            COUNT(*) AS rows,
            ROUND(AVG(wer), 4) AS avg_wer,
            ROUND(AVG(cer), 4) AS avg_cer,
            SUM(CASE WHEN wer <= 0.20 AND cer <= 0.05 THEN 1 ELSE 0 END)
                AS close_rows,
            SUM(CASE WHEN cer <= 0.10 THEN 1 ELSE 0 END)
                AS low_cer_rows
        FROM scribe_asr_crosscheck
        GROUP BY model_label
        ORDER BY model_label
        """,
    )


def report_payload(connection: sqlite3.Connection, task: str, model: str | None) -> dict[str, Any]:
    return {
        "database": connection.execute("PRAGMA database_list").fetchone()["file"],
        "decisions": decision_counts(connection),
        "surfaces": surface_counts(connection),
        "script_queue_by_old_category": script_queue_breakdown(connection),
        "same_speech_reviews": review_counts(connection, task, model),
        "asr_crosschecks": asr_crosscheck_counts(connection),
    }


def print_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = ("" if row.get(column) is None else str(row[column]) for column in columns)
        print("| " + " | ".join(values) + " |")


def print_report_markdown(payload: dict[str, Any]) -> None:
    print(f"# Scribe curation report\n\nDatabase: `{payload['database']}`\n")
    print("## Decisions\n")
    print_markdown_table(payload["decisions"], ["decision", "rows", "hours", "avg_wer", "avg_cer"])
    print("\n## Surfaces\n")
    print_markdown_table(
        payload["surfaces"],
        ["surface", "label", "rows", "hours", "avg_wer", "avg_cer"],
    )
    print("\n## Script Queue By Old Category\n")
    print_markdown_table(payload["script_queue_by_old_category"], ["audit_category", "rows"])
    print("\n## Same Speech Reviews\n")
    print_markdown_table(payload["same_speech_reviews"], ["task", "model", "decision", "rows"])
    print("\n## ASR Crosschecks\n")
    print_markdown_table(
        payload["asr_crosschecks"],
        ["model_label", "rows", "avg_wer", "avg_cer", "close_rows", "low_cer_rows"],
    )


def sample_rows(
    connection: sqlite3.Connection,
    where_sql: str,
    limit: int,
) -> list[dict[str, Any]]:
    return fetch_grouped(
        connection,
        f"""
        SELECT
            sample_id,
            job_order,
            row_decision,
            audit_difference_category,
            ROUND(resolved_wer, 4) AS wer,
            ROUND(resolved_cer, 4) AS cer,
            raw_reference_text,
            resolved_raw_scribe_text,
            normalized_reference,
            resolved_normalized_scribe,
            duration_seconds,
            audio_filepath
        FROM scribe_training_decisions
        WHERE {where_sql}
        ORDER BY job_order
        LIMIT ?
        """,
        (limit,),
    )


def print_rows_jsonl(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))


def manifest_where_sql(
    surface: str,
    task: str,
    model: str,
    *,
    include_script_reviews: bool,
) -> str:
    base = require_surface(surface)
    if not include_script_reviews:
        return f"({base})"
    model_sql = _quote_sql_text(model)
    task_sql = _quote_sql_text(task)
    return f"""
    (
        ({base})
        OR (
            row_decision = 'needs_same_speech_script_check'
            AND EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = scribe_training_decisions.sample_id
                    AND review.job_order = scribe_training_decisions.job_order
                    AND review.task = {task_sql}
                    AND review.model = {model_sql}
                    AND review.decision IN (
                        'same_speech_wrong_script',
                        'same_speech_minor_mixed_script'
                    )
            )
        )
    )
    """


def manifest_rows(
    connection: sqlite3.Connection,
    where_sql: str,
    split: str,
    task: str,
    model: str,
) -> Iterable[sqlite3.Row]:
    asr_recovery_task_placeholders = ", ".join("?" for _ in ASR_RECOVERY_EXPORT_TASKS)
    return connection.execute(
        f"""
        SELECT
            scribe_training_decisions.*,
            script_review.decision AS script_review_decision,
            script_review.reason AS script_review_reason,
            script_review.evidence AS script_review_evidence,
            rescue_review.decision AS rescue_review_decision,
            rescue_review.reason AS rescue_review_reason,
            rescue_review.evidence AS rescue_review_evidence,
            text_recovery_review.decision AS text_recovery_decision,
            text_recovery_review.reason AS text_recovery_reason,
            text_recovery_review.evidence AS text_recovery_evidence,
            consensus_label_review.decision AS consensus_label_decision,
            consensus_label_review.reason AS consensus_label_reason,
            consensus_label_review.evidence AS consensus_label_evidence,
            asr_recovery_review.task AS asr_recovery_task,
            asr_recovery_review.decision AS asr_recovery_decision,
            asr_recovery_review.reason AS asr_recovery_reason,
            asr_recovery_review.evidence AS asr_recovery_evidence,
            CASE
                WHEN consensus_label_review.decision = 'use_consensus_label'
                    AND scribe_training_decisions.resolved_cer < 0.15
                    AND scribe_training_decisions.audit_difference_category IN (
                        'punctuation_or_orthography_only',
                        'near_match',
                        'exact_match'
                    )
                    AND TRIM(COALESCE(
                        scribe_training_decisions.resolved_normalized_scribe,
                        ''
                    )) <> ''
                    AND scribe_training_decisions.normalized_reference
                        <> scribe_training_decisions.resolved_normalized_scribe
                    AND EXISTS (
                        SELECT 1
                        FROM scribe_asr_crosscheck consensus_asr
                        WHERE consensus_asr.sample_id = scribe_training_decisions.sample_id
                            AND consensus_asr.job_order = scribe_training_decisions.job_order
                            AND consensus_asr.normalized_hypothesis
                                = scribe_training_decisions.resolved_normalized_scribe
                    )
                    THEN scribe_training_decisions.resolved_normalized_scribe
                ELSE scribe_training_decisions.training_text
            END AS manifest_text,
            CASE
                WHEN consensus_label_review.decision = 'use_consensus_label'
                    AND scribe_training_decisions.resolved_cer < 0.15
                    AND scribe_training_decisions.audit_difference_category IN (
                        'punctuation_or_orthography_only',
                        'near_match',
                        'exact_match'
                    )
                    AND TRIM(COALESCE(
                        scribe_training_decisions.resolved_normalized_scribe,
                        ''
                    )) <> ''
                    AND scribe_training_decisions.normalized_reference
                        <> scribe_training_decisions.resolved_normalized_scribe
                    AND EXISTS (
                        SELECT 1
                        FROM scribe_asr_crosscheck consensus_asr
                        WHERE consensus_asr.sample_id = scribe_training_decisions.sample_id
                            AND consensus_asr.job_order = scribe_training_decisions.job_order
                            AND consensus_asr.normalized_hypothesis
                                = scribe_training_decisions.resolved_normalized_scribe
                    )
                    THEN 'resolved_scribe_300m_consensus_label'
                ELSE scribe_training_decisions.training_text_source
            END AS manifest_text_source
        FROM scribe_training_decisions
        LEFT JOIN scribe_same_speech_review script_review
            ON script_review.sample_id = scribe_training_decisions.sample_id
            AND script_review.job_order = scribe_training_decisions.job_order
            AND script_review.task = ?
            AND script_review.model = ?
        LEFT JOIN scribe_same_speech_review rescue_review
            ON rescue_review.sample_id = scribe_training_decisions.sample_id
            AND rescue_review.job_order = scribe_training_decisions.job_order
            AND rescue_review.task = ?
            AND rescue_review.model = ?
        LEFT JOIN scribe_same_speech_review text_recovery_review
            ON text_recovery_review.sample_id = scribe_training_decisions.sample_id
            AND text_recovery_review.job_order = scribe_training_decisions.job_order
            AND text_recovery_review.task = ?
            AND text_recovery_review.model = ?
        LEFT JOIN scribe_same_speech_review consensus_label_review
            ON consensus_label_review.sample_id = scribe_training_decisions.sample_id
            AND consensus_label_review.job_order = scribe_training_decisions.job_order
            AND consensus_label_review.task = ?
            AND consensus_label_review.model = ?
        LEFT JOIN (
            SELECT *
            FROM (
                SELECT
                    review.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY review.sample_id, review.job_order
                        ORDER BY
                            CASE
                                WHEN review.task LIKE '%held_strong_asr_recovery_v2'
                                    AND review.decision = 'use_reference_asr_recovery'
                                    THEN 1
                                WHEN review.decision = 'use_reference_asr_recovery' THEN 2
                                WHEN review.decision = 'hold_for_audio_review' THEN 3
                                ELSE 4
                            END,
                            review.task
                    ) AS review_rank
                FROM scribe_same_speech_review review
                WHERE review.task IN ({asr_recovery_task_placeholders})
                    AND review.model = ?
            ) ranked_asr_recovery_review
            WHERE review_rank = 1
        ) asr_recovery_review
            ON asr_recovery_review.sample_id = scribe_training_decisions.sample_id
            AND asr_recovery_review.job_order = scribe_training_decisions.job_order
        WHERE {where_sql}
            AND scribe_training_decisions.project_split = ?
            AND TRIM(COALESCE(
                CASE
                    WHEN consensus_label_review.decision = 'use_consensus_label'
                        AND scribe_training_decisions.resolved_cer < 0.15
                        AND scribe_training_decisions.audit_difference_category IN (
                            'punctuation_or_orthography_only',
                            'near_match',
                            'exact_match'
                        )
                        AND TRIM(COALESCE(
                            scribe_training_decisions.resolved_normalized_scribe,
                            ''
                        )) <> ''
                        AND scribe_training_decisions.normalized_reference
                            <> scribe_training_decisions.resolved_normalized_scribe
                        AND EXISTS (
                            SELECT 1
                            FROM scribe_asr_crosscheck consensus_asr
                            WHERE consensus_asr.sample_id = scribe_training_decisions.sample_id
                                AND consensus_asr.job_order = scribe_training_decisions.job_order
                                AND consensus_asr.normalized_hypothesis
                                    = scribe_training_decisions.resolved_normalized_scribe
                        )
                        THEN scribe_training_decisions.resolved_normalized_scribe
                    ELSE scribe_training_decisions.training_text
                END,
                ''
            )) <> ''
            AND TRIM(COALESCE(scribe_training_decisions.audio_filepath, '')) <> ''
        ORDER BY scribe_training_decisions.job_order
        """,
        (
            task,
            model,
            ASR_RESCUE_TASK,
            model,
            TEXT_RECOVERY_TASK,
            model,
            CONSENSUS_LABEL_TASK,
            model,
            *ASR_RECOVERY_EXPORT_TASKS,
            model,
            split,
        ),
    )


def manifest_payload(row: sqlite3.Row, surface: str) -> dict[str, Any]:
    payload = {
        "audio_filepath": row["audio_filepath"],
        "canonical_dataset": row["canonical_dataset"],
        "duration": row["duration_seconds"],
        "job_order": row["job_order"],
        "old_difference_category": row["audit_difference_category"],
        "project_split": row["project_split"],
        "row_decision": row["row_decision"],
        "sample_id": row["sample_id"],
        "scribe_cer": row["resolved_cer"],
        "scribe_text": row["resolved_normalized_scribe"],
        "scribe_wer": row["resolved_wer"],
        "source": row["source"],
        "source_config": row["source_config"],
        "split": row["project_split"],
        "surface": surface,
        "text": row["manifest_text"],
        "training_text_source": row["manifest_text_source"] or row["row_decision"],
        "used_rerun": row["used_rerun"],
    }
    if row["script_review_decision"]:
        payload.update(
            {
                "script_review_decision": row["script_review_decision"],
                "script_review_reason": row["script_review_reason"],
                "script_review_evidence": row["script_review_evidence"],
            }
        )
    if row["rescue_review_decision"]:
        payload.update(
            {
                "rescue_review_decision": row["rescue_review_decision"],
                "rescue_review_reason": row["rescue_review_reason"],
                "rescue_review_evidence": row["rescue_review_evidence"],
            }
        )
    if row["text_recovery_decision"]:
        payload.update(
            {
                "text_recovery_decision": row["text_recovery_decision"],
                "text_recovery_reason": row["text_recovery_reason"],
                "text_recovery_evidence": row["text_recovery_evidence"],
            }
        )
    if row["consensus_label_decision"]:
        payload.update(
            {
                "consensus_label_decision": row["consensus_label_decision"],
                "consensus_label_reason": row["consensus_label_reason"],
                "consensus_label_evidence": row["consensus_label_evidence"],
            }
        )
    if row["asr_recovery_decision"]:
        payload.update(
            {
                "asr_recovery_task": row["asr_recovery_task"],
                "asr_recovery_decision": row["asr_recovery_decision"],
                "asr_recovery_reason": row["asr_recovery_reason"],
                "asr_recovery_evidence": row["asr_recovery_evidence"],
            }
        )
    return payload


def write_summary_md(output_root: Path, stats: ExportStats) -> None:
    lines = [
        f"# Scribe Curation Export {stats.surface}",
        "",
        f"- Rows: `{stats.rows_written}`",
        f"- Hours: `{stats.hours_written:.2f}`",
        "",
        "| split | rows | hours |",
        "|---|---:|---:|",
    ]
    for split, count in sorted(stats.split_counts.items()):
        lines.append(f"| {split} | {count} | {stats.split_hours[split]:.2f} |")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_manifest(
    connection: sqlite3.Connection,
    surface: str,
    output_root: Path,
    splits: list[str],
    task: str,
    model: str,
    *,
    include_script_reviews: bool,
) -> ExportStats:
    where_sql = manifest_where_sql(
        surface, task, model, include_script_reviews=include_script_reviews
    )
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    hours_written = 0.0
    split_counts: dict[str, int] = {}
    split_hours: dict[str, float] = {}

    for split in splits:
        count = 0
        hours = 0.0
        manifest_path = manifest_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows(connection, where_sql, split, task, model):
                payload = manifest_payload(row, surface)
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
                hours += float(row["duration_seconds"] or 0.0) / 3600.0
        split_counts[split] = count
        split_hours[split] = hours
        rows_written += count
        hours_written += hours

    stats = ExportStats(
        surface=surface,
        output_root=str(output_root),
        rows_written=rows_written,
        hours_written=hours_written,
        split_counts=split_counts,
        split_hours=split_hours,
    )
    (output_root / "summary.json").write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_root, stats)
    return stats


def review_manifest_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    decisions: list[str],
    limit: int,
    max_duration: float | None,
) -> Iterable[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in decisions)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    duration_sql = ""
    params: list[Any] = [task, model, *decisions]
    if max_duration is not None:
        duration_sql = "AND decisions_table.duration_seconds <= ?"
        params.append(max_duration)
    return connection.execute(
        f"""
        SELECT
            decisions_table.*,
            review.decision AS review_decision,
            review.reason AS review_reason,
            review.evidence AS review_evidence
        FROM scribe_training_decisions decisions_table
        JOIN scribe_same_speech_review review
            ON review.sample_id = decisions_table.sample_id
            AND review.job_order = decisions_table.job_order
        WHERE review.task = ?
            AND review.model = ?
            AND review.decision IN ({placeholders})
            AND TRIM(COALESCE(decisions_table.training_text, '')) <> ''
            AND TRIM(COALESCE(decisions_table.audio_filepath, '')) <> ''
            {duration_sql}
        ORDER BY decisions_table.job_order
        {limit_sql}
        """,
        params,
    )


def review_manifest_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = manifest_payload(row, "same_speech_script_review")
    payload.update(
        {
            "review_decision": row["review_decision"],
            "review_reason": row["review_reason"],
            "review_evidence": row["review_evidence"],
        }
    )
    return payload


def export_review_manifest(
    connection: sqlite3.Connection,
    output: Path,
    task: str,
    model: str,
    decisions: list[str],
    limit: int,
    max_duration: float | None,
) -> ReviewManifestStats:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    hours_written = 0.0
    with output.open("w", encoding="utf-8") as handle:
        for row in review_manifest_rows(
            connection,
            task,
            model,
            decisions,
            limit,
            max_duration,
        ):
            handle.write(json.dumps(review_manifest_payload(row), ensure_ascii=False) + "\n")
            rows_written += 1
            hours_written += float(row["duration_seconds"] or 0.0) / 3600.0
    return ReviewManifestStats(
        output=str(output),
        task=task,
        model=model,
        decisions=decisions,
        rows_written=rows_written,
        hours_written=hours_written,
    )


def asr_queue_rows(
    connection: sqlite3.Connection,
    queue: str,
    limit: int,
    max_duration: float | None,
    exclude_existing_model_label: str | None,
) -> tuple[list[sqlite3.Row], int]:
    try:
        where_sql = ASR_RECOVERY_QUEUES[queue]
    except KeyError as exc:
        known = ", ".join(sorted(ASR_RECOVERY_QUEUES))
        raise SystemExit(f"unknown ASR recovery queue {queue!r}; known queues: {known}") from exc

    conditions = [
        f"({where_sql})",
        "TRIM(COALESCE(training_text, '')) <> ''",
        "TRIM(COALESCE(audio_filepath, '')) <> ''",
    ]
    params: list[Any] = []
    if max_duration is not None:
        conditions.append("duration_seconds <= ?")
        params.append(max_duration)
    if exclude_existing_model_label:
        conditions.append(
            """
            NOT EXISTS (
                SELECT 1
                FROM scribe_asr_crosscheck asr
                WHERE asr.sample_id = scribe_training_decisions.sample_id
                    AND asr.job_order = scribe_training_decisions.job_order
                    AND asr.model_label = ?
            )
            """
        )
        params.append(exclude_existing_model_label)

    skipped_existing = 0
    if exclude_existing_model_label:
        skipped_existing = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM scribe_training_decisions
                WHERE ({where_sql})
                    AND EXISTS (
                        SELECT 1
                        FROM scribe_asr_crosscheck asr
                        WHERE asr.sample_id = scribe_training_decisions.sample_id
                            AND asr.job_order = scribe_training_decisions.job_order
                            AND asr.model_label = ?
                    )
                """,
                (exclude_existing_model_label,),
            ).fetchone()[0]
        )

    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        SELECT *
        FROM scribe_training_decisions
        WHERE {" AND ".join(conditions)}
        ORDER BY
            CASE audit_difference_category
                WHEN 'extra_speech' THEN 1
                WHEN 'omitted_speech' THEN 2
                WHEN 'content_mismatch' THEN 3
                WHEN 'boundary_mismatch' THEN 4
                ELSE 9
            END,
            resolved_cer,
            resolved_wer,
            job_order
        {limit_sql}
        """,
        params,
    ).fetchall()
    return rows, skipped_existing


def asr_queue_manifest_payload(row: sqlite3.Row, queue: str) -> dict[str, Any]:
    return {
        "audio_filepath": row["audio_filepath"],
        "audit_difference_category": row["audit_difference_category"],
        "canonical_dataset": row["canonical_dataset"],
        "duration": row["duration_seconds"],
        "job_order": row["job_order"],
        "project_split": row["project_split"],
        "queue": queue,
        "raw_reference_text": row["raw_reference_text"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "row_decision": row["row_decision"],
        "sample_id": row["sample_id"],
        "scribe_cer": row["resolved_cer"],
        "scribe_text": row["resolved_normalized_scribe"],
        "scribe_wer": row["resolved_wer"],
        "source": row["source"],
        "source_config": row["source_config"],
        "text": row["training_text"],
        "training_text_source": "normalized_reference_asr_recovery_candidate",
    }


def export_asr_queue_manifest(
    connection: sqlite3.Connection,
    output: Path,
    queue: str,
    limit: int,
    max_duration: float | None,
    exclude_existing_model_label: str | None,
) -> AsrQueueManifestStats:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped_existing = asr_queue_rows(
        connection,
        queue,
        limit,
        max_duration,
        exclude_existing_model_label,
    )
    rows_written = 0
    hours_written = 0.0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(asr_queue_manifest_payload(row, queue), ensure_ascii=False) + "\n"
            )
            rows_written += 1
            hours_written += float(row["duration_seconds"] or 0.0) / 3600.0
    return AsrQueueManifestStats(
        output=str(output),
        queue=queue,
        rows_written=rows_written,
        hours_written=hours_written,
        max_duration=max_duration,
        skipped_existing_asr_rows=skipped_existing,
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def manifest_lookup(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, row in enumerate(read_jsonl(path)):
        rows_by_index[index] = row
        sample_id = str(row["sample_id"])
        if sample_id in rows_by_id:
            duplicate_ids.add(sample_id)
        else:
            rows_by_id[sample_id] = row
    for sample_id in duplicate_ids:
        del rows_by_id[sample_id]
    return rows_by_index, rows_by_id


def import_asr_check(
    connection: sqlite3.Connection,
    manifest: Path,
    samples: Path,
    model_label: str,
) -> AsrCheckImportStats:
    manifest_rows_by_index, manifest_rows_by_id = manifest_lookup(manifest)
    now = utc_now()
    rows_seen = 0
    rows_imported = 0
    missing_manifest_rows = 0
    with connection:
        for sample in read_jsonl(samples):
            rows_seen += 1
            sample_id = str(sample["id"])
            manifest_row = None
            sample_index = sample.get("index")
            if sample_index is not None:
                manifest_row = manifest_rows_by_index.get(int(sample_index))
                if manifest_row and str(manifest_row["sample_id"]) != sample_id:
                    manifest_row = None
            if manifest_row is None:
                manifest_row = manifest_rows_by_id.get(sample_id)
            if manifest_row is None:
                missing_manifest_rows += 1
                continue
            connection.execute(
                """
                INSERT INTO scribe_asr_crosscheck (
                    sample_id,
                    job_order,
                    model_label,
                    source_manifest,
                    benchmark_samples,
                    reference_text,
                    hypothesis_text,
                    normalized_reference,
                    normalized_hypothesis,
                    wer,
                    cer,
                    duration_seconds,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id, job_order, model_label) DO UPDATE SET
                    source_manifest = excluded.source_manifest,
                    benchmark_samples = excluded.benchmark_samples,
                    reference_text = excluded.reference_text,
                    hypothesis_text = excluded.hypothesis_text,
                    normalized_reference = excluded.normalized_reference,
                    normalized_hypothesis = excluded.normalized_hypothesis,
                    wer = excluded.wer,
                    cer = excluded.cer,
                    duration_seconds = excluded.duration_seconds,
                    updated_at = excluded.updated_at
                """,
                (
                    sample_id,
                    int(manifest_row["job_order"]),
                    model_label,
                    str(manifest),
                    str(samples),
                    str(sample["reference"]),
                    str(sample["hypothesis"]),
                    str(sample["normalized_reference"]),
                    str(sample["normalized_hypothesis"]),
                    float(sample["wer"]),
                    float(sample["cer"]),
                    float(sample["duration_seconds"]),
                    now,
                    now,
                ),
            )
            rows_imported += 1
    return AsrCheckImportStats(
        database=connection.execute("PRAGMA database_list").fetchone()["file"],
        model_label=model_label,
        rows_seen=rows_seen,
        rows_imported=rows_imported,
        missing_manifest_rows=missing_manifest_rows,
    )


def build_script_prompt(row: sqlite3.Row) -> str:
    payload = {
        "sample_id": row["sample_id"],
        "job_order": row["job_order"],
        "normalized_reference": row["normalized_reference"],
        "raw_reference_text": row["raw_reference_text"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "old_difference_category": row["audit_difference_category"],
    }
    return (
        "You compare Persian ASR dataset rows for curation.\n\n"
        "The training label, if accepted, will be normalized_reference.\n"
        "Your job is only to decide whether raw_scribe_text appears to represent "
        "the same spoken utterance as normalized_reference.\n\n"
        "Return JSON with exactly: sample_id, job_order, decision, reason, evidence.\n\n"
        "decision must be one of:\n"
        "same_speech_wrong_script,\n"
        "same_speech_minor_mixed_script,\n"
        "different_speech,\n"
        "different_language,\n"
        "annotation_or_noise,\n"
        "unclear.\n\n"
        "Rules:\n"
        "- Choose same_speech_wrong_script when Latin words are names, acronyms, "
        "products, places, or loanwords matching the Persian-script reference.\n"
        "- Choose same_speech_minor_mixed_script when the sentence mostly matches but "
        "Scribe used an English spelling or translation for a small part.\n"
        "- Choose different_speech when Scribe adds or removes a content word beyond "
        "script, spelling, spacing, or a loanword/acronym rendering.\n"
        "- Choose different_speech when the words refer to a different utterance or meaning.\n"
        "- Choose different_language when Scribe output is mainly another spoken language.\n"
        "- Choose annotation_or_noise for bracketed non-speech labels, speaker labels, "
        "or endpoint artifacts.\n"
        "- Choose unclear when text alone cannot decide.\n"
        "- Do not choose a training label.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_asr_rescue_prompt(row: sqlite3.Row) -> str:
    payload = {
        "sample_id": row["sample_id"],
        "job_order": row["job_order"],
        "normalized_reference": row["normalized_reference"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "script_review_decision": row["script_review_decision"],
        "script_review_reason": row["script_review_reason"],
        "script_review_evidence": row["script_review_evidence"],
        "old_difference_category": row["audit_difference_category"],
        "omni300_normalized_hypothesis": row["asr_normalized_hypothesis"],
        "omni300_wer": row["asr_wer"],
        "omni300_cer": row["asr_cer"],
        "omni300_flag": row["asr_review_flag"],
    }
    return (
        "You review Persian ASR dataset rows for possible recovery.\n\n"
        "The training label, if accepted, will be normalized_reference.\n"
        "Scribe previously failed or disagreed. A local 300M Persian ASR model "
        "also transcribed the same audio. Decide whether the reference text is "
        "safe enough for training from the text evidence.\n\n"
        "Return JSON with exactly: sample_id, job_order, decision, reason, evidence.\n\n"
        "decision must be one of:\n"
        "use_reference_asr_rescue,\n"
        "hold_for_audio_review,\n"
        "reject_reference_mismatch.\n\n"
        "Rules:\n"
        "- Choose use_reference_asr_rescue when the 300M hypothesis preserves the "
        "same Persian words as normalized_reference and Scribe appears to be an "
        "endpoint error, translation, script failure, or unrelated transcription.\n"
        "- Choose reject_reference_mismatch when the 300M hypothesis and Scribe "
        "both indicate the audio likely says a different utterance or another "
        "spoken language than normalized_reference.\n"
        "- Choose hold_for_audio_review when text evidence is mixed, the 300M "
        "hypothesis drops/adds meaningful content, or the row may be translated "
        "speech rather than Persian speech.\n"
        "- Use normalized_reference as the only possible training text.\n"
        "- Do not invent a corrected label.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_asr_recovery_prompt(row: sqlite3.Row) -> str:
    payload = {
        "sample_id": row["sample_id"],
        "job_order": row["job_order"],
        "normalized_reference": row["normalized_reference"],
        "normalized_scribe": row["resolved_normalized_scribe"],
        "raw_reference_text": row["raw_reference_text"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "row_decision": row["row_decision"],
        "old_difference_category": row["audit_difference_category"],
        "current_scribe_wer": row["resolved_wer"],
        "current_scribe_cer": row["resolved_cer"],
        "asr_model_label": row["asr_model_label"],
        "asr_normalized_hypothesis": row["asr_normalized_hypothesis"],
        "asr_wer": row["asr_wer"],
        "asr_cer": row["asr_cer"],
        "asr_recovery_flag": row["asr_recovery_flag"],
    }
    return (
        "You review Persian ASR dataset rows for ASR-backed recovery.\n\n"
        "The training label, if accepted, will be normalized_reference.\n"
        "Scribe and a local 300M Persian ASR model both transcribed the same audio. "
        "Decide whether the reference text is safe enough for ASR training.\n\n"
        "Return JSON with exactly: sample_id, job_order, decision, reason, evidence. "
        "reason and evidence must be plain strings.\n\n"
        "decision must be one of:\n"
        "use_reference_asr_recovery,\n"
        "hold_for_audio_review,\n"
        "reject_reference_mismatch.\n\n"
        "Rules:\n"
        "- Choose use_reference_asr_recovery when the 300M hypothesis preserves the "
        "same Persian content as normalized_reference, and Scribe differences look "
        "like boundary drift, filler, formatting, short tail words, or endpoint error.\n"
        "- Choose reject_reference_mismatch when Scribe and 300M both indicate the "
        "audio says different content, a different speaker segment, or another language.\n"
        "- Choose hold_for_audio_review when content words, names, numbers, polarity, "
        "speaker intent, or segment boundaries remain ambiguous from text evidence.\n"
        "- Use normalized_reference as the only possible training text.\n"
        "- Do not invent a corrected label.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def coerce_review_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_review_text_fields(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        # Intentionally ValueError, not TypeError: this runs inside the classify retry
        # loop / worker (both catch `except Exception`); changing the type would be a
        # silent deviation from the pre-refactor contract.
        raise ValueError("classifier response must be a JSON object")  # noqa: TRY004
    normalized = dict(response)
    normalized["reason"] = coerce_review_text(normalized["reason"])
    normalized["evidence"] = coerce_review_text(normalized["evidence"])
    return normalized


def build_text_recovery_prompt(row: sqlite3.Row) -> str:
    payload = {
        "sample_id": row["sample_id"],
        "job_order": row["job_order"],
        "normalized_reference": row["normalized_reference"],
        "normalized_scribe": row["resolved_normalized_scribe"],
        "raw_reference_text": row["raw_reference_text"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "old_difference_category": row["audit_difference_category"],
        "current_wer": row["resolved_wer"],
        "current_cer": row["resolved_cer"],
    }
    return (
        "You review Persian ASR dataset rows for text recovery.\n\n"
        "The training label, if accepted, will be normalized_reference.\n"
        "Scribe output is evidence about whether the reference/audio pair is usable. "
        "Decide whether the reference text is safe enough for ASR training.\n\n"
        "Return JSON with exactly: sample_id, job_order, decision, reason, evidence.\n\n"
        "decision must be one of:\n"
        "use_reference_text_recovery,\n"
        "hold_for_audio_review,\n"
        "reject_reference_mismatch.\n\n"
        "Rules:\n"
        "- Choose use_reference_text_recovery when differences are bounded spelling, "
        "spacing, normalization, number/name rendering, filler, short boundary drift, "
        "or obvious Scribe error while the same Persian utterance remains clear.\n"
        "- Choose reject_reference_mismatch when Scribe evidence indicates the audio "
        "likely says a different utterance or another spoken language than the reference.\n"
        "- Choose hold_for_audio_review when a content word changes, meaningful speech "
        "is added/omitted, or text evidence cannot prove the reference is safe.\n"
        "- Use normalized_reference as the only possible training text.\n"
        "- Do not invent a corrected label.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_consensus_label_prompt(row: sqlite3.Row) -> str:
    payload = {
        "sample_id": row["sample_id"],
        "job_order": row["job_order"],
        "raw_reference_text": row["raw_reference_text"],
        "raw_scribe_text": row["resolved_raw_scribe_text"],
        "normalized_reference": row["normalized_reference"],
        "consensus_label": row["consensus_label"],
        "normalized_scribe": row["resolved_normalized_scribe"],
        "omni300_normalized_hypothesis": row["asr_normalized_hypothesis"],
        "old_difference_category": row["audit_difference_category"],
        "row_decision": row["row_decision"],
        "scribe_wer": row["resolved_wer"],
        "scribe_cer": row["resolved_cer"],
        "omni300_wer": row["asr_wer"],
        "omni300_cer": row["asr_cer"],
        "asr_model_label": row["asr_model_label"],
    }
    return (
        "You review Persian ASR dataset rows for consensus-label recovery.\n\n"
        "Scribe and a local 300M Persian ASR model produced the same normalized "
        "transcript for this audio, and that consensus_label differs from "
        "normalized_reference. Decide whether consensus_label is safe enough as "
        "the ASR training label for the audio.\n\n"
        "Return JSON with exactly: sample_id, job_order, decision, reason, evidence. "
        "reason and evidence must be plain strings.\n\n"
        "decision must be one of:\n"
        "use_consensus_label,\n"
        "hold_for_audio_review,\n"
        "reject_consensus_label.\n\n"
        "Rules:\n"
        "- Choose use_consensus_label when Scribe and 300M agreement makes "
        "consensus_label the likely transcript for the audio, including apparent "
        "reference typos, normalized spelling, word-form corrections, or boundary "
        "words that likely exist in the audio.\n"
        "- Keep use_consensus_label conservative: the consensus must read like a "
        "transcript correction, not a paraphrase or a rewritten sentence.\n"
        "- Choose hold_for_audio_review when text evidence cannot decide between "
        "normalized_reference and consensus_label, or when names, numbers, polarity, "
        "speaker intent, content-word substitutions, person or number agreement, "
        "dropped clauses, added clauses, or segment boundaries remain ambiguous.\n"
        "- Choose reject_consensus_label when consensus_label contains non-speech "
        "tags, endpoint artifacts, another language, incoherent text, or an "
        "obvious hallucination.\n"
        "- Use consensus_label as the only replacement label when accepted.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def pending_script_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
) -> tuple[list[sqlite3.Row], int]:
    existing = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            """,
            (task, model),
        ).fetchone()[0]
    )
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        SELECT decisions.*
        FROM scribe_training_decisions decisions
        WHERE row_decision = 'needs_same_speech_script_check'
            AND NOT EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = decisions.sample_id
                    AND review.job_order = decisions.job_order
                    AND review.task = ?
                    AND review.model = ?
            )
        ORDER BY decisions.job_order
        {limit_sql}
        """,
        (task, model),
    ).fetchall()
    return rows, existing


def pending_asr_rescue_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    flags: list[str],
) -> tuple[list[sqlite3.Row], int]:
    existing = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            """,
            (task, model),
        ).fetchone()[0]
    )
    placeholders = ", ".join("?" for _ in flags)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        SELECT flags_table.*
        FROM scribe_asr_review_flags flags_table
        WHERE flags_table.asr_model_label = 'omni300_exact_best_script_rejected_full'
            AND flags_table.asr_review_flag IN ({placeholders})
            AND NOT EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = flags_table.sample_id
                    AND review.job_order = flags_table.job_order
                    AND review.task = ?
                    AND review.model = ?
            )
        ORDER BY
            CASE flags_table.asr_review_flag
                WHEN 'strong_rescue_review' THEN 1
                WHEN 'weak_rescue_review' THEN 2
                ELSE 9
            END,
            flags_table.asr_cer,
            flags_table.job_order
        {limit_sql}
        """,
        (*flags, task, model),
    ).fetchall()
    return rows, existing


def asr_recovery_flag_case(alias: str = "asr") -> str:
    return f"""
        CASE
            WHEN {alias}.wer = 0 AND {alias}.cer = 0
                THEN 'asr_exact_reference'
            WHEN {alias}.wer <= 0.20 AND {alias}.cer <= 0.05
                THEN 'asr_strong_reference'
            WHEN {alias}.wer <= 0.35 AND {alias}.cer <= 0.10
                THEN 'asr_possible_reference'
            WHEN {alias}.wer >= 0.75 OR {alias}.cer >= 0.25
                THEN 'asr_reject_reference'
            ELSE 'asr_mixed_reference'
        END
    """


def pending_asr_recovery_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    queue: str,
    asr_model_label: str,
    flags: list[str],
) -> tuple[list[sqlite3.Row], int]:
    try:
        where_sql = ASR_RECOVERY_QUEUES[queue]
    except KeyError as exc:
        known = ", ".join(sorted(ASR_RECOVERY_QUEUES))
        raise SystemExit(f"unknown ASR recovery queue {queue!r}; known queues: {known}") from exc
    existing = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            """,
            (task, model),
        ).fetchone()[0]
    )
    placeholders = ", ".join("?" for _ in flags)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        WITH candidates AS (
            SELECT
                scribe_training_decisions.*,
                asr.model_label AS asr_model_label,
                asr.normalized_hypothesis AS asr_normalized_hypothesis,
                asr.wer AS asr_wer,
                asr.cer AS asr_cer,
                {asr_recovery_flag_case()} AS asr_recovery_flag
            FROM scribe_training_decisions
            JOIN scribe_asr_crosscheck asr
                ON asr.sample_id = scribe_training_decisions.sample_id
                AND asr.job_order = scribe_training_decisions.job_order
                AND asr.model_label = ?
            WHERE ({where_sql})
        )
        SELECT *
        FROM candidates
        WHERE asr_recovery_flag IN ({placeholders})
            AND NOT EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = candidates.sample_id
                    AND review.job_order = candidates.job_order
                    AND review.task = ?
                    AND review.model = ?
            )
        ORDER BY
            CASE asr_recovery_flag
                WHEN 'asr_exact_reference' THEN 1
                WHEN 'asr_strong_reference' THEN 2
                WHEN 'asr_possible_reference' THEN 3
                WHEN 'asr_mixed_reference' THEN 4
                ELSE 9
            END,
            asr_cer,
            asr_wer,
            job_order
        {limit_sql}
        """,
        (asr_model_label, *flags, task, model),
    ).fetchall()
    return rows, existing


def pending_text_recovery_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    queue: str,
) -> tuple[list[sqlite3.Row], int]:
    try:
        where_sql = TEXT_RECOVERY_QUEUES[queue]
    except KeyError as exc:
        known = ", ".join(sorted(TEXT_RECOVERY_QUEUES))
        raise SystemExit(f"unknown text recovery queue {queue!r}; known queues: {known}") from exc
    existing = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            """,
            (task, model),
        ).fetchone()[0]
    )
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        SELECT decisions.*
        FROM scribe_training_decisions decisions
        WHERE ({where_sql})
            AND NOT EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = decisions.sample_id
                    AND review.job_order = decisions.job_order
                    AND review.task = ?
                    AND review.model = ?
            )
        ORDER BY decisions.resolved_cer, decisions.resolved_wer, decisions.job_order
        {limit_sql}
        """,
        (task, model),
    ).fetchall()
    return rows, existing


def pending_consensus_label_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
) -> tuple[list[sqlite3.Row], int]:
    existing = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM scribe_same_speech_review
            WHERE task = ? AND model = ?
            """,
            (task, model),
        ).fetchone()[0]
    )
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        WITH candidates AS (
            SELECT
                decisions.*,
                asr.model_label AS asr_model_label,
                asr.normalized_hypothesis AS asr_normalized_hypothesis,
                asr.normalized_hypothesis AS consensus_label,
                asr.wer AS asr_wer,
                asr.cer AS asr_cer,
                ROW_NUMBER() OVER (
                    PARTITION BY decisions.sample_id, decisions.job_order
                    ORDER BY asr.cer, asr.wer, asr.model_label
                ) AS consensus_rank
            FROM scribe_training_decisions decisions
            JOIN scribe_asr_crosscheck asr
                ON asr.sample_id = decisions.sample_id
                AND asr.job_order = decisions.job_order
            WHERE TRIM(COALESCE(decisions.resolved_normalized_scribe, '')) <> ''
                AND decisions.resolved_normalized_scribe = asr.normalized_hypothesis
                AND decisions.normalized_reference <> decisions.resolved_normalized_scribe
                AND TRIM(COALESCE(decisions.audio_filepath, '')) <> ''
        )
        SELECT *
        FROM candidates
        WHERE consensus_rank = 1
            AND NOT EXISTS (
                SELECT 1
                FROM scribe_same_speech_review review
                WHERE review.sample_id = candidates.sample_id
                    AND review.job_order = candidates.job_order
                    AND review.task = ?
                    AND review.model = ?
            )
        ORDER BY asr_cer, asr_wer, job_order
        {limit_sql}
        """,
        (task, model),
    ).fetchall()
    return rows, existing


def store_review(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    task: str,
    model: str,
    classification: dict[str, Any],
) -> None:
    now = utc_now()
    with connection:
        connection.execute(
            """
            INSERT INTO scribe_same_speech_review (
                sample_id,
                job_order,
                task,
                model,
                decision,
                reason,
                evidence,
                response_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id, job_order, task, model) DO UPDATE SET
                decision = excluded.decision,
                reason = excluded.reason,
                evidence = excluded.evidence,
                response_json = excluded.response_json,
                updated_at = excluded.updated_at
            """,
            (
                row["sample_id"],
                row["job_order"],
                task,
                model,
                classification["decision"],
                classification["reason"],
                classification["evidence"],
                json.dumps(classification, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )


def classify_one(
    row: sqlite3.Row,
    client: SuperwhisperClient,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_CLASSIFY_ATTEMPTS):
        try:
            response = client.generate_json(
                model,
                [{"role": "user", "content": build_script_prompt(row)}],
                schema=SCRIPT_REVIEW_SCHEMA,
                response_format_name="same_speech_script_review",
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_CLASSIFY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError("unreachable classifier retry state") from last_error
    if str(response["sample_id"]) != str(row["sample_id"]):
        raise ValueError("sample_id mismatch in response")
    if int(response["job_order"]) != int(row["job_order"]):
        raise ValueError("job_order mismatch in response")
    decision = SCRIPT_REVIEW_ALIASES.get(str(response["decision"]), str(response["decision"]))
    response["decision"] = decision
    if decision not in SCRIPT_REVIEW_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    return dict(response)


def classify_asr_rescue_one(
    row: sqlite3.Row,
    client: SuperwhisperClient,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_CLASSIFY_ATTEMPTS):
        try:
            response = client.generate_json(
                model,
                [{"role": "user", "content": build_asr_rescue_prompt(row)}],
                schema=ASR_RESCUE_SCHEMA,
                response_format_name="asr_rescue_review",
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_CLASSIFY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError("unreachable classifier retry state") from last_error
    if str(response["sample_id"]) != str(row["sample_id"]):
        raise ValueError("sample_id mismatch in response")
    if int(response["job_order"]) != int(row["job_order"]):
        raise ValueError("job_order mismatch in response")
    decision = str(response["decision"])
    if decision not in ASR_RESCUE_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    return dict(response)


def classify_asr_recovery_one(
    row: sqlite3.Row,
    client: SuperwhisperClient,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_CLASSIFY_ATTEMPTS):
        try:
            model_response = client.generate(
                model,
                [{"role": "user", "content": build_asr_recovery_prompt(row)}],
                response_format=json_schema_response_format(
                    "asr_recovery_review",
                    ASR_RECOVERY_SCHEMA,
                ),
                max_tokens=max_tokens,
            )
            response = normalize_review_text_fields(model_response.json())
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_CLASSIFY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError("unreachable classifier retry state") from last_error
    if str(response["sample_id"]) != str(row["sample_id"]):
        raise ValueError("sample_id mismatch in response")
    if int(response["job_order"]) != int(row["job_order"]):
        raise ValueError("job_order mismatch in response")
    decision = str(response["decision"])
    response["decision"] = decision
    if decision not in ASR_RECOVERY_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    return dict(response)


def classify_text_recovery_one(
    row: sqlite3.Row,
    client: SuperwhisperClient,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_CLASSIFY_ATTEMPTS):
        try:
            response = client.generate_json(
                model,
                [{"role": "user", "content": build_text_recovery_prompt(row)}],
                schema=TEXT_RECOVERY_SCHEMA,
                response_format_name="text_recovery_review",
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_CLASSIFY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError("unreachable classifier retry state") from last_error
    if str(response["sample_id"]) != str(row["sample_id"]):
        raise ValueError("sample_id mismatch in response")
    if int(response["job_order"]) != int(row["job_order"]):
        raise ValueError("job_order mismatch in response")
    decision = str(response["decision"])
    if decision not in TEXT_RECOVERY_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    return dict(response)


def classify_consensus_label_one(
    row: sqlite3.Row,
    client: SuperwhisperClient,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_CLASSIFY_ATTEMPTS):
        try:
            model_response = client.generate(
                model,
                [{"role": "user", "content": build_consensus_label_prompt(row)}],
                response_format=json_schema_response_format(
                    "consensus_label_review",
                    CONSENSUS_LABEL_SCHEMA,
                ),
                max_tokens=max_tokens,
            )
            response = normalize_review_text_fields(model_response.json())
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_CLASSIFY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError("unreachable classifier retry state") from last_error
    if str(response["sample_id"]) != str(row["sample_id"]):
        raise ValueError("sample_id mismatch in response")
    if int(response["job_order"]) != int(row["job_order"]):
        raise ValueError("job_order mismatch in response")
    decision = str(response["decision"])
    response["decision"] = decision
    if decision not in CONSENSUS_LABEL_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    return dict(response)


def _run_bounded(
    rows: list[sqlite3.Row],
    worker: Callable[[sqlite3.Row], _ClassifyResult],
    on_result: Callable[[_ClassifyResult], None],
    max_workers: int,
) -> None:
    """Map ``worker`` over ``rows`` with at most ~``max_workers`` in flight, streaming
    each result to ``on_result`` as it completes (order-independent)."""
    max_in_flight = max(max_workers * 4, max_workers)
    index = 0
    futures: dict[Future[_ClassifyResult], sqlite3.Row] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while index < len(rows) and len(futures) < max_in_flight:
            futures[pool.submit(worker, rows[index])] = rows[index]
            index += 1
        while futures:
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                on_result(future.result())
                while index < len(rows) and len(futures) < max_in_flight:
                    futures[pool.submit(worker, rows[index])] = rows[index]
                    index += 1


def _run_classification(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    rows: list[sqlite3.Row],
    skipped_existing: int,
    classify_one_fn: ClassifyOneFn,
    max_workers: int,
    max_tokens: int,
) -> ReviewStats:
    """Run ``classify_one_fn`` over ``rows`` with bounded parallelism and persist results.

    Shared engine for the ``classify_*_rows`` entry points: they differ only in which
    pending-row query feeds ``rows`` and which per-row classifier is passed here.
    """
    db_path = connection.execute("PRAGMA database_list").fetchone()["file"]
    if not rows:
        return ReviewStats(
            database=db_path,
            task=task,
            model=model,
            selected=0,
            reviewed=0,
            failed=0,
            skipped_existing=skipped_existing,
        )

    client = SuperwhisperClient()
    model_spec(model)
    write_lock = threading.Lock()
    counters = {"reviewed": 0, "failed": 0}
    failures: list[str] = []

    def worker(item: sqlite3.Row) -> _ClassifyResult:
        try:
            return item, classify_one_fn(item, client, model, max_tokens), None
        except Exception as exc:  # noqa: BLE001 - batch review should continue and report failures.
            return item, None, f"{type(exc).__name__}: {exc}"

    def on_result(result: _ClassifyResult) -> None:
        row, classification, error = result
        if classification is None:
            counters["failed"] += 1
            failures.append(f"{row['job_order']}: {error}")
            print(f"FAIL {row['job_order']}: {error}", file=sys.stderr)
            return
        with write_lock:
            store_review(connection, row, task, model, classification)
        counters["reviewed"] += 1
        print(f"OK {row['job_order']}: {classification['decision']}", file=sys.stderr)

    _run_bounded(rows, worker, on_result, max_workers)

    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures[:20]:
            print(failure, file=sys.stderr)

    return ReviewStats(
        database=db_path,
        task=task,
        model=model,
        selected=len(rows),
        reviewed=counters["reviewed"],
        failed=counters["failed"],
        skipped_existing=skipped_existing,
    )


def classify_script_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    max_workers: int,
    max_tokens: int,
) -> ReviewStats:
    rows, skipped_existing = pending_script_rows(connection, task, model, limit)
    return _run_classification(
        connection,
        task,
        model,
        rows,
        skipped_existing,
        classify_one,
        max_workers,
        max_tokens,
    )


def classify_asr_rescue_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    max_workers: int,
    max_tokens: int,
    flags: list[str],
) -> ReviewStats:
    rows, skipped_existing = pending_asr_rescue_rows(connection, task, model, limit, flags)
    return _run_classification(
        connection,
        task,
        model,
        rows,
        skipped_existing,
        classify_asr_rescue_one,
        max_workers,
        max_tokens,
    )


def classify_asr_recovery_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    max_workers: int,
    max_tokens: int,
    queue: str,
    asr_model_label: str,
    flags: list[str],
) -> ReviewStats:
    rows, skipped_existing = pending_asr_recovery_rows(
        connection,
        task,
        model,
        limit,
        queue,
        asr_model_label,
        flags,
    )
    return _run_classification(
        connection,
        task,
        model,
        rows,
        skipped_existing,
        classify_asr_recovery_one,
        max_workers,
        max_tokens,
    )


def classify_text_recovery_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    max_workers: int,
    max_tokens: int,
    queue: str,
) -> ReviewStats:
    rows, skipped_existing = pending_text_recovery_rows(
        connection,
        task,
        model,
        limit,
        queue,
    )
    return _run_classification(
        connection,
        task,
        model,
        rows,
        skipped_existing,
        classify_text_recovery_one,
        max_workers,
        max_tokens,
    )


def classify_consensus_label_rows(
    connection: sqlite3.Connection,
    task: str,
    model: str,
    limit: int,
    max_workers: int,
    max_tokens: int,
) -> ReviewStats:
    rows, skipped_existing = pending_consensus_label_rows(connection, task, model, limit)
    return _run_classification(
        connection,
        task,
        model,
        rows,
        skipped_existing,
        classify_consensus_label_one,
        max_workers,
        max_tokens,
    )


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Inspect, review, and export SQLite-backed Scribe curation rows."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("install-views")

    report = subparsers.add_parser("report")
    report.add_argument("--format", choices=("json", "markdown"), default="json")
    report.add_argument("--review-task", default=SAME_SPEECH_TASK)
    report.add_argument("--review-model", default=None)

    sample = subparsers.add_parser("sample")
    sample_filter = sample.add_mutually_exclusive_group(required=True)
    sample_filter.add_argument("--decision", default=None)
    sample_filter.add_argument("--surface", choices=sorted(SURFACES), default=None)
    sample.add_argument("--limit", type=int, default=20)
    sample.add_argument("--format", choices=("jsonl", "markdown"), default="jsonl")

    export = subparsers.add_parser("export-manifest")
    export.add_argument("--surface", choices=sorted(SURFACES), required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--split", action="append", default=[])
    export.add_argument("--include-script-reviews", action="store_true")
    export.add_argument("--review-task", default=SAME_SPEECH_TASK)
    export.add_argument("--review-model", default="gpt-5.4-mini")

    review_manifest = subparsers.add_parser("export-review-manifest")
    review_manifest.add_argument("--output", type=Path, required=True)
    review_manifest.add_argument("--task", default=SAME_SPEECH_TASK)
    review_manifest.add_argument("--review-model", default="gpt-5.4-mini")
    review_manifest.add_argument(
        "--decision",
        action="append",
        choices=sorted(SCRIPT_REVIEW_DECISIONS),
        default=[],
    )
    review_manifest.add_argument("--limit", type=int, default=0)
    review_manifest.add_argument("--max-duration", type=float, default=None)

    asr_manifest = subparsers.add_parser("export-asr-queue-manifest")
    asr_manifest.add_argument("--queue", choices=sorted(ASR_RECOVERY_QUEUES), required=True)
    asr_manifest.add_argument("--output", type=Path, required=True)
    asr_manifest.add_argument("--limit", type=int, default=0)
    asr_manifest.add_argument("--max-duration", type=float, default=None)
    asr_manifest.add_argument("--exclude-existing-model-label", default=None)

    asr_check = subparsers.add_parser("import-asr-check")
    asr_check.add_argument("--manifest", type=Path, required=True)
    asr_check.add_argument("--samples", type=Path, required=True)
    asr_check.add_argument("--model-label", required=True)

    review = subparsers.add_parser("classify-script")
    review.add_argument("--task", default=SAME_SPEECH_TASK)
    review.add_argument("--model", default="gpt-5.4-mini")
    review.add_argument("--limit", type=int, default=100)
    review.add_argument("--max-workers", type=int, default=16)
    review.add_argument("--max-tokens", type=int, default=512)

    rescue = subparsers.add_parser("classify-asr-rescue")
    rescue.add_argument("--task", default=ASR_RESCUE_TASK)
    rescue.add_argument("--model", default="gpt-5.4-mini")
    rescue.add_argument("--limit", type=int, default=100)
    rescue.add_argument("--max-workers", type=int, default=16)
    rescue.add_argument("--max-tokens", type=int, default=512)
    rescue.add_argument(
        "--flag",
        action="append",
        choices=(
            "strong_rescue_review",
            "weak_rescue_review",
        ),
        default=[],
    )

    asr_recovery = subparsers.add_parser("classify-asr-recovery")
    asr_recovery.add_argument("--task", default=ASR_RECOVERY_TASK)
    asr_recovery.add_argument("--model", default="gpt-5.4-mini")
    asr_recovery.add_argument("--queue", choices=sorted(ASR_RECOVERY_QUEUES), required=True)
    asr_recovery.add_argument("--asr-model-label", required=True)
    asr_recovery.add_argument("--limit", type=int, default=100)
    asr_recovery.add_argument("--max-workers", type=int, default=16)
    asr_recovery.add_argument("--max-tokens", type=int, default=512)
    asr_recovery.add_argument(
        "--flag",
        action="append",
        choices=ASR_RECOVERY_FLAGS,
        default=[],
    )

    text_recovery = subparsers.add_parser("classify-text-recovery")
    text_recovery.add_argument("--task", default=TEXT_RECOVERY_TASK)
    text_recovery.add_argument("--model", default="gpt-5.4-mini")
    text_recovery.add_argument("--queue", choices=sorted(TEXT_RECOVERY_QUEUES), required=True)
    text_recovery.add_argument("--limit", type=int, default=100)
    text_recovery.add_argument("--max-workers", type=int, default=16)
    text_recovery.add_argument("--max-tokens", type=int, default=512)

    consensus_label = subparsers.add_parser("classify-consensus-label")
    consensus_label.add_argument("--task", default=CONSENSUS_LABEL_TASK)
    consensus_label.add_argument("--model", default="gpt-5.4-mini")
    consensus_label.add_argument("--limit", type=int, default=100)
    consensus_label.add_argument("--max-workers", type=int, default=16)
    consensus_label.add_argument("--max-tokens", type=int, default=512)

    subparsers.add_parser("review-summary")
    return parser


def _cmd_report(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    payload = report_payload(connection, args.review_task, args.review_model)
    if args.format == "markdown":
        print_report_markdown(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_sample(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    where_sql = (
        require_surface(args.surface)
        if args.surface
        else f"row_decision = {_quote_sql_text(args.decision)}"
    )
    rows = sample_rows(connection, where_sql, args.limit)
    if args.format == "markdown":
        print_markdown_table(
            rows,
            [
                "job_order",
                "row_decision",
                "audit_difference_category",
                "wer",
                "cer",
                "raw_reference_text",
                "resolved_raw_scribe_text",
            ],
        )
    else:
        print_rows_jsonl(rows)
    return 0


def _cmd_export_manifest(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    splits = args.split or ["train", "dev", "test"]
    export_stats = export_manifest(
        connection=connection,
        surface=args.surface,
        output_root=args.output_root.expanduser(),
        splits=splits,
        include_script_reviews=args.include_script_reviews,
        task=args.review_task,
        model=args.review_model,
    )
    print(json.dumps(asdict(export_stats), ensure_ascii=False, indent=2))
    return 0


def _cmd_export_review_manifest(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    decisions = args.decision or [
        "same_speech_wrong_script",
        "same_speech_minor_mixed_script",
    ]
    review_manifest_stats = export_review_manifest(
        connection=connection,
        output=args.output,
        task=args.task,
        model=args.review_model,
        decisions=decisions,
        limit=args.limit,
        max_duration=args.max_duration,
    )
    print(json.dumps(asdict(review_manifest_stats), ensure_ascii=False, indent=2))
    return 0


def _cmd_export_asr_queue_manifest(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> int:
    asr_manifest_stats = export_asr_queue_manifest(
        connection=connection,
        output=args.output,
        queue=args.queue,
        limit=args.limit,
        max_duration=args.max_duration,
        exclude_existing_model_label=args.exclude_existing_model_label,
    )
    print(json.dumps(asdict(asr_manifest_stats), ensure_ascii=False, indent=2))
    return 0


def _cmd_import_asr_check(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    asr_check_stats = import_asr_check(
        connection=connection,
        manifest=args.manifest,
        samples=args.samples,
        model_label=args.model_label,
    )
    print(json.dumps(asdict(asr_check_stats), ensure_ascii=False, indent=2))
    return 0


def _cmd_classify_script(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    review_stats = classify_script_rows(
        connection=connection,
        task=args.task,
        model=args.model,
        limit=args.limit,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(asdict(review_stats), ensure_ascii=False, indent=2))
    return 0 if review_stats.failed == 0 else 1


def _cmd_classify_asr_rescue(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    flags = args.flag or ["strong_rescue_review", "weak_rescue_review"]
    review_stats = classify_asr_rescue_rows(
        connection=connection,
        task=args.task,
        model=args.model,
        limit=args.limit,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        flags=flags,
    )
    print(json.dumps(asdict(review_stats), ensure_ascii=False, indent=2))
    return 0 if review_stats.failed == 0 else 1


def _cmd_classify_asr_recovery(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    flags = args.flag or [
        "asr_exact_reference",
        "asr_strong_reference",
        "asr_possible_reference",
    ]
    review_stats = classify_asr_recovery_rows(
        connection=connection,
        task=args.task,
        model=args.model,
        limit=args.limit,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        queue=args.queue,
        asr_model_label=args.asr_model_label,
        flags=flags,
    )
    print(json.dumps(asdict(review_stats), ensure_ascii=False, indent=2))
    return 0 if review_stats.failed == 0 else 1


def _cmd_classify_text_recovery(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    review_stats = classify_text_recovery_rows(
        connection=connection,
        task=args.task,
        model=args.model,
        limit=args.limit,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        queue=args.queue,
    )
    print(json.dumps(asdict(review_stats), ensure_ascii=False, indent=2))
    return 0 if review_stats.failed == 0 else 1


def _cmd_classify_consensus_label(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> int:
    review_stats = classify_consensus_label_rows(
        connection=connection,
        task=args.task,
        model=args.model,
        limit=args.limit,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(asdict(review_stats), ensure_ascii=False, indent=2))
    return 0 if review_stats.failed == 0 else 1


def _cmd_review_summary(connection: sqlite3.Connection, _args: argparse.Namespace) -> int:
    payload = review_counts(connection, SAME_SPEECH_TASK, None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# Commands that only need (connection, args). `install-views` is handled separately
# because it is the one command that consumes the ViewStats returned by the shared setup.
_COMMANDS: dict[str, Callable[[sqlite3.Connection, argparse.Namespace], int]] = {
    "report": _cmd_report,
    "sample": _cmd_sample,
    "export-manifest": _cmd_export_manifest,
    "export-review-manifest": _cmd_export_review_manifest,
    "export-asr-queue-manifest": _cmd_export_asr_queue_manifest,
    "import-asr-check": _cmd_import_asr_check,
    "classify-script": _cmd_classify_script,
    "classify-asr-rescue": _cmd_classify_asr_rescue,
    "classify-asr-recovery": _cmd_classify_asr_recovery,
    "classify-text-recovery": _cmd_classify_text_recovery,
    "classify-consensus-label": _cmd_classify_consensus_label,
    "review-summary": _cmd_review_summary,
}


def run_with_connection(args: argparse.Namespace) -> int:
    connection = connect(args.database)
    try:
        stats = install_views(connection)
        if args.command == "install-views":
            print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
            return 0
        handler = _COMMANDS.get(args.command)
        if handler is None:
            raise SystemExit(f"unknown command: {args.command}")
        return handler(connection, args)
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_with_connection(args)


if __name__ == "__main__":
    raise SystemExit(main())

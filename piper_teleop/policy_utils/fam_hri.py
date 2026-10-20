"""Local adapter for the external FAM-HRI bundle_pipeline used by the source.

This repository does not include SAM3 weights or the fam_hri package. This
adapter consumes already synchronized RGB/gaze and text, never TCP camera data.
"""

import importlib
from pathlib import Path
import sys
import numpy as np


class FamHRIPlanner:
    def __init__(self, config):
        self.config = config
        for key, default in (("min_dwell_s", 0.5), ("max_disp_px", 25)):
            value = float(config.get(key, default))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        root = config.get("fam_hri_root")
        if root:
            path = Path(root).expanduser().resolve()
            if not (path / "fam_hri/server/bundle_pipeline.py").is_file():
                raise FileNotFoundError(
                    "fam_hri_root must contain fam_hri/server/bundle_pipeline.py"
                )
            sys.path.insert(0, str(path))
        try:
            self.pipeline = importlib.import_module("fam_hri.server.bundle_pipeline")
        except ImportError as error:
            raise ImportError(
                "Install the external FAM-HRI package/SAM3 dependencies or set fam_hri_root; "
                "they were not included in aria_eval_scripts_refactored"
            ) from error
        self.labeler = self.planner = None
        self.closed = False

    def plan(self, frames, queue_state, transcript, initial):
        del queue_state, initial
        if self.closed:
            raise RuntimeError("FAM-HRI planner is closed")
        if not transcript.strip():
            return {"command": "NoOp"}
        selected, run = [], []
        dwell = float(self.config.get("min_dwell_s", 0.5))
        dispersion = float(self.config.get("max_disp_px", 25))
        fired = False
        for frame in frames:
            gaze = np.asarray(frame.gaze_xy, dtype=float)
            if (
                gaze.shape != (2,)
                or not np.isfinite(gaze).all()
                or frame.full_rgb is None
            ):
                run = []
                fired = False
                continue
            if (
                run
                and np.linalg.norm(gaze - np.mean([f.gaze_xy for f in run], axis=0))
                > dispersion
            ):
                run = []
                fired = False
            run.append(frame)
            duration = (frame.timestamp_ns - run[0].timestamp_ns) / 1e9
            if duration >= dwell and not fired:
                selected.append((frame, duration))
                fired = True
        if not selected:
            return {"command": "NoOp"}
        p = self.pipeline
        text_prompt = self.config.get(
            "object_prompt", "cup . bottle . muffin . basket . tool"
        )
        if self.labeler is None:
            self.labeler = p.build_labeler(
                self.config.get("device", "cuda"),
                text_prompt,
                float(self.config.get("score_threshold", 0.3)),
            )
        if self.planner is None:
            model = self.config.get("llm_model")
            if not model:
                raise ValueError("Set llm_model for the external FAM-HRI planner")
            self.planner = p.build_planner(model)
        inputs = [
            p.KFInput(
                timestamp_ns=f.timestamp_ns,
                gaze_xy=tuple(f.gaze_xy),
                dwell_s=d,
                image_rgb=f.full_rgb,
            )
            for f, d in selected
        ]
        plan, _ = p.process_utterance(
            transcript,
            inputs,
            self.labeler,
            self.planner,
            text_prompt,
            opts=p.PipelineOpts(
                no_llm=bool(self.config.get("no_llm", False)),
                planner_context=self.config.get("planner_context", ""),
            ),
            return_annotated=False,
        )
        if self.closed:
            return {"command": "NoOp"}
        labels = plan.get("keyframe_labels", [])
        actions = plan.get("actions", [])
        elements = []
        for i, instruction in enumerate(plan.get("subtasks", [])):
            action = actions[i] if i < len(actions) else {}
            referent = action.get("object_label") if isinstance(action, dict) else None
            candidates = [
                j
                for j, label in enumerate(labels)
                if referent
                and label.get("gaze_object_label") == referent
                and j < len(selected)
            ]
            frame = selected[
                candidates[0] if candidates else min(i, len(selected) - 1)
            ][0]
            seconds = (frame.timestamp_ns - frames[0].timestamp_ns) / 1e9
            elements.append(
                {
                    "subtask_instruction": instruction,
                    "keyframe_interval": {"start": seconds, "end": seconds},
                }
            )
        return {"command": "Append", "new_elements": elements}

    def close(self):
        self.closed = True

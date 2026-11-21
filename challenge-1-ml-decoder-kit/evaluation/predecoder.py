"""Frozen tier-0 commits composed in the Gross error-mechanism frame."""
import hashlib
import inspect
import json
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.adapters import KitSession, ShotResult
from evaluation.predecoder_layout import GrossLayout, matrix_identity
from evaluation.router import TieredSession


def model_factory_sha256(spec):
    module, name = spec.split(":", 1)
    factory = getattr(import_module(module), name)
    source = inspect.getsourcefile(factory)
    if source is None:
        raise ValueError("CNN model factory must have a readable Python source file")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def binary_vector(value, width, name):
    value = np.asarray(value)
    if value.shape != (width,) or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must be binary shape ({width},)")
    return value.astype(np.uint8, copy=False)


def residual_syndrome(H, syndrome, commits):
    syndrome = binary_vector(syndrome, H.shape[0], "syndrome")
    commits = binary_vector(commits, H.shape[1], "commits")
    return syndrome ^ (np.asarray(H @ commits, dtype=np.uint8).reshape(-1) % 2)


def combine_correction(commits, residual):
    commits = binary_vector(commits, len(commits), "commits")
    residual = binary_vector(residual, len(commits), "residual correction")
    return commits ^ residual


def combine_observables(L, commits, residual_prediction):
    commits = binary_vector(commits, L.shape[1], "commits")
    residual_prediction = binary_vector(residual_prediction, L.shape[0], "residual prediction")
    frame = np.asarray((L @ commits) % 2, dtype=np.uint8).reshape(-1)
    return frame ^ residual_prediction


def bp_present_probability(posterior_ratios):
    """relay-bp posterior_ratios are log P(absent)/P(present); positive means absent.

    Verified against the installed 0.2.2 tiny-matrix decoder: prior 0.01 with
    zero syndrome yields positive ratios, while a decoded present fault has
    a negative ratio after convergence.
    """
    from scipy.special import expit

    ratios = np.asarray(posterior_ratios, dtype=np.float64)
    if np.isnan(ratios).any():
        raise ValueError("BP posterior contains NaN")
    return expit(-ratios)


class FrozenCommitter:
    def __init__(self, problem, manifest_path=None, *, validation_candidate=None):
        self.problem = problem
        if (manifest_path is None) == (validation_candidate is None):
            raise ValueError("provide exactly one frozen manifest or validation candidate")
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.manifest = (json.loads(self.manifest_path.read_text()) if self.manifest_path is not None
                         else dict(validation_candidate))
        manifest = self.manifest
        if self.manifest_path is not None and manifest.get("selection_stage") != "validation":
            raise ValueError("commit threshold must be frozen from validation data")
        if self.manifest_path is None and manifest.get("selection_stage") != "validation_candidate":
            raise ValueError("unfrozen commits are allowed only for validation selection")
        if manifest.get("basis") != problem.basis or manifest.get("p") != problem.p:
            raise ValueError("commit manifest basis or p mismatch")
        if manifest.get("matrix_sha256") != matrix_identity(problem.H, problem.L, problem.priors):
            raise ValueError("commit manifest matrix identity mismatch")
        self.mode = manifest["mode"]
        self.threshold = float(manifest["threshold"])
        self.settings = {"mode": self.mode, "threshold": self.threshold,
                         "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
                         if self.manifest_path is not None else None}
        self.layout = GrossLayout(problem.H, problem.L, problem.priors) if self.mode == "cnn" else None
        self.bp = None
        if self.mode == "bp" or manifest.get("use_bp_context"):
            import relay_bp

            # The approved BP control is eight iterations WITH memory.
            self.bp = relay_bp.MinSumBPDecoderF64(problem.H.tocsr(), error_priors=problem.priors,
                                                  max_iter=8, alpha=1.0, gamma0=0.15)
            self.settings["bp_context"] = {"max_iter": 8, "alpha": 1.0, "gamma0": 0.15}
        self.model = None
        if self.mode == "cnn":
            spec = manifest["model_factory"]
            module, name = spec.split(":", 1)
            source_hash = model_factory_sha256(spec)
            if manifest.get("model_factory_sha256") and manifest["model_factory_sha256"] != source_hash:
                raise ValueError("CNN model factory source changed after validation")
            self.model = getattr(import_module(module), name)(problem, manifest["model_path"], manifest.get("model_config", {}))
            self.settings["model_factory"] = spec
            self.settings["model_path"] = manifest["model_path"]
            self.settings["device"] = str(getattr(self.model, "device", "unknown"))
            self.settings["model_factory_sha256"] = source_hash
            expected = manifest.get("model_sha256")
            if expected and hashlib.sha256(Path(manifest["model_path"]).read_bytes()).hexdigest() != expected:
                raise ValueError("CNN model file differs from frozen manifest")
        elif self.mode != "bp":
            raise ValueError("commit mode must be bp or cnn")

    def commit(self, syndrome, full_detectors):
        syndrome = binary_vector(syndrome, 936, "syndrome")
        context = {"basis": self.problem.basis, "full_detectors": np.asarray(full_detectors, dtype=np.uint8)}
        bp_iterations = None
        if self.bp is not None:
            result = self.bp.decode_detailed(syndrome)
            ratios = np.asarray(result.posterior_ratios, dtype=np.float64)
            if ratios.shape != (8784,):
                raise ValueError("memory-BP posterior does not have 8784 fault entries")
            context["bp_posterior_ratios"] = ratios
            bp_iterations = int(result.iterations)
        if self.mode == "bp":
            if self.threshold < 0:
                raise ValueError("BP presence log-odds threshold must be nonnegative")
            # Positive log P(0)/P(1) is evidence of absence, never a commit.
            commits = (context["bp_posterior_ratios"] <= -self.threshold).astype(np.uint8)
        else:
            grid = self.layout.vector_to_syndrome_grid(syndrome)
            values = np.asarray(self.model.predict_faults(grid, self.layout.prior_grid, context))
            if values.shape == self.layout.fault_shape:
                values = self.layout.fault_grid_to_vector(values)
            if values.shape != (8784,) or not np.isfinite(values).all():
                raise ValueError("CNN must return finite probabilities/logits for 8784 mechanisms")
            kind = self.manifest["output_kind"]
            if kind == "probability":
                if not 0 <= self.threshold <= 1 or (values < 0).any() or (values > 1).any():
                    raise ValueError("CNN probability outputs/threshold must lie in [0,1]")
            elif kind != "logit":
                raise ValueError("CNN output_kind must be probability or logit")
            commits = (values >= self.threshold).astype(np.uint8)
        return commits, {"commits": int(commits.sum()), "bp_context_iterations": bp_iterations}


class ComposedSession:
    def __init__(self, problem, options, *, validation_committer=None):
        start = perf_counter()
        manifest_paths = options.get("commit_manifests", {})
        path = manifest_paths.get(problem.basis)
        if not path and validation_committer is None:
            raise ValueError(f"missing frozen {problem.basis} commit manifest")
        self.problem = problem
        self.committer = validation_committer or FrozenCommitter(problem, path)
        downstream = options.get("downstream", "tiered")
        if downstream == "tiered":
            models = options.get("router_models", {})
            if problem.basis not in models:
                raise ValueError(f"missing frozen {problem.basis} router model")
            self.downstream = TieredSession(problem, models[problem.basis])
        elif downstream == "bposd":
            self.downstream = KitSession(problem, "bposd")
        else:
            raise ValueError("downstream must be tiered or bposd")
        self.setup_seconds = perf_counter() - start
        if downstream == "tiered":
            expected = self.committer.manifest.get("router_model_sha256")
            actual = hashlib.sha256(Path(models[problem.basis]).read_bytes()).hexdigest()
            if expected and expected != actual:
                raise ValueError("downstream router differs from validation-frozen router")
            trained_commit_hash = self.downstream.model.get("commit_manifest_sha256")
            if trained_commit_hash and trained_commit_hash != self.committer.settings["manifest_sha256"]:
                raise ValueError("residual router was trained with a different commit manifest")
        self.settings = {"committer": self.committer.settings, "downstream": downstream,
                         "downstream_settings": self.downstream.settings,
                         "preprocessor_is_ml": self.committer.mode == "cnn", "router_is_ml": downstream == "tiered",
                         "composition": "residual = s XOR Hc; logical = Lc XOR downstream(residual)"}

    def decode_shot(self, full_detectors, basis_detectors):
        start = perf_counter()
        commits, diagnostics = self.committer.commit(basis_detectors, full_detectors)
        tier0_seconds = perf_counter() - start
        residual = residual_syndrome(self.problem.H, basis_detectors, commits)
        downstream = self.downstream.decode_shot(full_detectors, residual)
        prediction = combine_observables(self.problem.L, commits, downstream.predictions)
        metrics = dict(downstream.metrics)
        metrics.update(diagnostics)
        metrics.update({"tier0_seconds": tier0_seconds, "commit_weight": int(commits.sum()),
                        "residual_weight": int(residual.sum()),
                        "kit_decode_seconds": downstream.decode_seconds if downstream.decode_seconds is not None
                        else downstream.metrics.get("kit_decode_seconds")})
        return ShotResult(prediction, metrics)


def make_composed_decoder(problem, options):
    return ComposedSession(problem, options)


def oracle_commits_for_diagnostic(mechanism_truth):
    """Synthetic-label ceiling only. Never wire this into make_composed_decoder."""
    return binary_vector(mechanism_truth, 8784, "synthetic mechanism truth").copy()

"""Conditional Wald prediction and first-stage diagnostics."""

import numpy as np


WALD_UNDEFINED_TOLERANCE = 1e-12


def wald_diagnostics(dt, true_delta_t=None):
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    absolute = np.abs(dt)
    result = {
        'min_abs_delta_t': float(np.min(absolute)),
        'q01_abs_delta_t': float(np.quantile(absolute, .01)),
        'q05_abs_delta_t': float(np.quantile(absolute, .05)),
        'q10_abs_delta_t': float(np.quantile(absolute, .10)),
        'median_abs_delta_t': float(np.median(absolute)),
        'exact_zero_count': int(np.sum(dt == 0.0)),
        'negative_delta_t_rate': float(np.mean(dt < 0.0)),
        'weak_rate_0.01': float(np.mean(absolute <= .01)),
        'weak_rate_0.05': float(np.mean(absolute <= .05)),
        'weak_rate_0.10': float(np.mean(absolute <= .10)),
        'undefined_count_1e-12': int(np.sum(absolute <= WALD_UNDEFINED_TOLERANCE)),
    }
    if true_delta_t is not None:
        truth = np.asarray(true_delta_t, dtype=np.float64).reshape(-1)
        if truth.shape != dt.shape:
            raise ValueError('estimated and true delta_T are not aligned')
        result.update({
            'first_stage_rmse_vs_true': float(np.sqrt(np.mean((dt-truth)**2))),
            'first_stage_correlation_vs_true': float(np.corrcoef(dt, truth)[0, 1]),
            'first_stage_sign_mismatch_rate': float(np.mean(np.sign(dt) != np.sign(truth))),
        })
    return result


def ratio(dy, dt):
    """Literal Wald ratio; weak thresholds are diagnostics, never clipping."""
    dy = np.asarray(dy, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    if dy.shape != dt.shape:
        raise ValueError('delta_Y and delta_T are not aligned')
    result = np.full_like(dy, np.nan, dtype=np.float64)
    valid = np.abs(dt) > WALD_UNDEFINED_TOLERANCE
    np.divide(dy, dt, out=result, where=valid)
    return result


def wald(arrays, backend, seed, tabpfn_version='v3'):
    x, t, z, y, xq = (np.asarray(arrays[k], float) for k in
                       ('X_context', 'treatment', 'instrument', 'outcome', 'X_query'))
    if backend == 'causalpfn':
        from ..cfm import _run_causalpfn_v2
        estimates, branches = {}, {}
        for name, target in (('delta_Y', y), ('delta_T', t)):
            branch = dict(X_context=x, treatment=z, outcome=target, X_query=xq)
            prediction, _, _, metadata = _run_causalpfn_v2(branch, 'cuda:0', 25, compute_uq=False)
            estimates[name], branches[name] = np.asarray(prediction, float), metadata
        provenance = dict(branches=branches, treatment_slot='instrument I',
                          outcome_slots=['observed Y', 'observed binary T'],
                          native_iv_interface=False, native_seed=82718)
    else:
        from ..frontdoor import _factories, _predict_chunks, _release
        classifier, regressor, provenance = (
            _factories(backend, seed, tabpfn_version)
            if backend == 'tabpfn' and tabpfn_version != 'v3'
            else _factories(backend, seed)
        )
        context = np.column_stack([x, z])
        queries = [np.column_stack([xq, np.full(len(xq), arm)]) for arm in (0, 1)]
        outcome_model = regressor(30)
        outcome_model.fit(context, y)
        dy = _predict_chunks(outcome_model, queries[1])-_predict_chunks(outcome_model, queries[0])
        _release(outcome_model)
        treatment_model = classifier(1)
        treatment_model.fit(context, t.astype(int))
        dt = (_predict_chunks(treatment_model, queries[1], probability=True)
              -_predict_chunks(treatment_model, queries[0], probability=True))
        _release(treatment_model)
        estimates = dict(delta_Y=dy, delta_T=dt)
    dy, dt = estimates['delta_Y'], estimates['delta_T']
    diagnostics = wald_diagnostics(dt)
    tabpfn_label = f"TabPFN-{str(tabpfn_version).removeprefix('v')}"
    backend_labels = {
        'tabpfn': (f'{tabpfn_label} regressor', f'{tabpfn_label} classifier', 'predict_proba'),
        'xgboost': ('XGBoost regressor', 'XGBoost classifier', 'predict_proba'),
        'causalpfn': ('CausalPFN outcome branch', 'CausalPFN first-stage branch', 'native prediction'),
    }
    outcome_backend, first_stage_backend, first_stage_output = backend_labels[backend]
    return dict(tau_hat=ratio(dy, dt), **estimates), dict(
                backend=backend, provenance=provenance,
                identification_wrapper_used=True, estimator='conditional_wald',
                formula='conditional Wald delta_Y(X)/delta_T(X)', denominator_floor=None,
                numerator='delta_y_i1_minus_i0', denominator='delta_t_i1_minus_i0',
                denominator_sign_preserved=True,
                undefined_tolerance=WALD_UNDEFINED_TOLERANCE,
                undefined_if_abs_denominator_le=WALD_UNDEFINED_TOLERANCE,
                denominator_rule='literal division; abs(delta_T)<=1e-12 is undefined',
                denominator_diagnostics=diagnostics,
                outcome_backend=outcome_backend, first_stage_backend=first_stage_backend,
                first_stage_output=first_stage_output, final_query_schema=['X'],
                estimand_name='conditional Wald',
                generic_interpretation='conditional LATE under standard IV assumptions',
                benchmark_target='CATE tau(X)',
                target_equivalence_reason=('In this SCM, Y(1)-Y(0)=tau(X) does not vary across latent '
                                           'compliance types, so the conditional Wald estimand equals tau(X).'),
                cross_fitting_folds=0, context_rows=len(x), query_columns='X only',
                tabpfn_version=(str(tabpfn_version).removeprefix('v')
                                if backend == 'tabpfn' else None))

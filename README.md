
# Well-Log ML for Subsurface Characterization

Research project exploring the integration of conventional well logs and laboratory core measurements for data-driven subsurface characterization.

## Research objectives

- Evaluate machine-learning methods for estimating rock permeability from integrated petrophysical measurements.
- Investigate model generalization under depth-blocked validation.
- Quantify prediction errors across permeability regimes.
- Explore how legacy subsurface datasets can support new geological interpretations.

## Data and methodology

The current experimental study uses 325 paired core–log observations from one anonymized well.

The workflow includes:

- Well-log features, including gamma ray (GR), sonic transit time (DT), and resistivity (RT)
- Laboratory core measurements and petrophysical attributes
- Data quality checks and feature preparation
- Regression models, including Random Forest, Extra Trees, Ridge, and histogram-based gradient boosting
- Five-fold depth-blocked outer validation and three-fold inner model selection
- Out-of-fold prediction diagnostics

## Preliminary results

In the current experiment, the Random Forest model achieved an out-of-fold mean absolute error of approximately 0.615 in log10 permeability units.

Performance varies substantially across permeability regimes. The model tends to overpredict very low permeability and underpredict high permeability.

These findings highlight the limitations of conventional supervised learning when training data are sparse or unevenly distributed.

## Limitations

- Current evaluation is based on a single well.
- Depth-blocked validation is not equivalent to independent-well validation.
- Results do not establish transferability to other reservoirs or geological settings.
- Additional independent datasets are required before operational use.

## Relevance

The project demonstrates a reproducible research approach to integrating historical subsurface measurements, evaluating machine-learning models, and interpreting their limitations.

Potential applications include petrophysical interpretation, reservoir characterization, and the assessment of legacy geoscientific datasets.

## Data availability

Original well logs, core measurements, sample identifiers, depths, and geological provenance are not publicly distributed due to confidentiality restrictions.

Public examples and reproducibility materials will be added separately after review and anonymization.

## Status

Research in progress. This repository currently documents the methodology and preliminary aggregate findings. It does not yet contain the original experimental data or a complete executable reproduction package.

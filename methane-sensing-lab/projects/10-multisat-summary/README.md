# Multi-Satellite Methane Monitoring Summary

An integrated framework for methane point-source detection and quantification using multispectral and hyperspectral satellite observations.

This repository summarises the main methodologies developed in the thesis *Detection and Quantification of Methane Point Sources Using Multispectral Observations*. The overall research investigates how multispectral observations, particularly Sentinel-2, can be combined with artificial intelligence, cross-sensor information, and meteorological data to support scalable methane point-source monitoring.

The framework progresses from methane plume detection and segmentation to spectral enhancement and emission-rate quantification.

## Overview

Satellite methane monitoring involves a trade-off between spatial resolution, spectral sensitivity, revisit frequency, and spatial coverage.

Sentinel-2 provides high spatial resolution, frequent observations, global coverage, and free access to a long-term archive. However, it was not designed specifically for methane detection and has limited spectral sampling in the methane-sensitive SWIR region.

This project summarises how these limitations can be addressed by combining physical knowledge, temporal information, artificial intelligence, hyperspectral observations, and meteorological constraints.

```text
                    Multi-Satellite Methane Monitoring
                                  │
                                  ▼
                         Sentinel-2 observations
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        Spectral + temporal                  Spatial information
           information                              │
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                       Methane plume detection
                                  │
                                  ▼
                     Automated plume segmentation
                                  │
                                  ▼
                Cross-sensor spectral enhancement
                         Sentinel-2 + EMIT
                                  │
                                  ▼
                    Methane-sensitive features
                                  │
                                  ▼
                  Emission-rate quantification
                    + wind information
                                  │
                                  ▼
                   Multi-sensor monitoring system
```

## Main Components

The thesis develops four complementary methodological components.

### 1. S2MethTracker

**S2MethTracker** establishes a physically interpretable baseline for methane point-source monitoring using Sentinel-2 imagery.

The framework combines:

* Methane-sensitive spectral information
* Bi-temporal observations
* Normalized Difference Methane Index (NDMI)
* Temporal Difference Index (TDI)
* Spatial filtering
* False-detection elimination
* Plume segmentation
* Emission-rate estimation

The method demonstrates the feasibility of using Sentinel-2 for monitoring strong methane point sources while identifying important limitations caused by surface background variability.

Large-scale and long-term experiments confirmed 106 methane emission events, with plume structures showing strong consistency with independent airborne observations. Approximate detection thresholds were around 1.5 t/h over relatively homogeneous surfaces and 3 t/h over heterogeneous surfaces. These values should be interpreted as empirical detection results rather than intrinsic sensor limits.

### 2. N-BPMSNet

**N-BPMSNet** is an NDMI-guided bi-temporal deep learning framework for automated methane plume detection and segmentation.

It addresses limitations of conventional rule-based methods, including:

* Low methane signal-to-noise ratio
* Complex surface backgrounds
* False detections
* Manually selected segmentation thresholds
* Limited scalability of conventional image processing

The model learns methane-sensitive spectral features together with temporal plume changes from Sentinel-2 observations.

On the reserved test dataset, N-BPMSNet achieved:

* **F1-score: 0.8858**
* **AUC: 0.9856**

The dataset contains 11,494 annotated plume and non-plume samples from 44 point sources across the USA, Algeria, and Turkmenistan. Additional tests were performed over previously unseen regions and challenging conditions.

### 3. MCSSR-Det

**MCSSR-Det** addresses the spectral limitations of Sentinel-2, particularly for methane detection over vegetation and other spectrally complex surfaces.

The framework combines:

* Sentinel-2 multispectral observations
* EMIT hyperspectral observations
* Cross-sensor spectral super-resolution
* Methane spectral priors
* Matched-filter responses
* Methane spectral projection
* Physics-guided plume detection

The MCSSR module reconstructs methane-sensitive EMIT-like hyperspectral representations from Sentinel-2 observations. The reconstructed spectra are then used together with methane response priors for plume detection.

The purpose is not to treat reconstructed hyperspectral data as equivalent to a real hyperspectral measurement. Instead, the method transfers methane-sensitive spectral information from hyperspectral observations to the more widely available Sentinel-2 archive.

The cross-sensor dataset includes EMIT hyperspectral observations with 285 bands and Sentinel-2 observations with 13 multispectral bands. Data cover multiple geographic regions and methane source sectors, including oil and gas, coal mining, solid waste, livestock, and other sources.

### 4. MEQNet

**MEQNet** extends the framework from plume detection to direct methane emission quantification.

The model uses:

* Bi-temporal Sentinel-2 observations
* Methane-sensitive SWIR information
* Temporal changes
* NDMI-guided features
* Methane column enhancement patterns
* 10-m wind information

The framework contains two main stages:

```text
Sentinel-2 observations
          │
          ▼
 Enhancement Map Generator
          │
          ▼
Methane column enhancement
          │
          +───────────────+
          │               │
          ▼               ▼
     Plume structure   Wind information
          │               │
          └───────┬───────┘
                  ▼
       Emission Rate Estimator
                  │
                  ▼
        Methane emission rate
```

MEQNet provides an end-to-end alternative to workflows that separately perform methane retrieval, plume segmentation, and empirical emission-rate estimation.

However, the estimated emission rates remain dependent on the training data, reference emission information, wind inputs, surface conditions, and observational conditions. The outputs should therefore be interpreted as condition-dependent estimates rather than absolute measurements.

## Multi-Sensor Monitoring Strategy

The methods developed in this thesis are complementary rather than intended to replace dedicated methane sensors.

A potential monitoring strategy is:

```text
                 Global / Regional Screening
                          │
                          ▼
                    Sentinel-2
              Frequent + Wide Coverage
                          │
                          ▼
                 Candidate Plume Events
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     Hyperspectral Data          AI Quantification
       EMIT / similar             + Wind Data
             │                         │
             └────────────┬────────────┘
                          ▼
                 Event Confirmation
                          │
                          ▼
              Airborne / Ground Validation
                          │
                          ▼
             Emission Reporting & Monitoring
```

This tiered strategy reflects the complementary strengths of different observing systems.

Sentinel-2 can provide broad and repeated screening, while hyperspectral, airborne, or ground observations can provide additional confirmation and more detailed characterisation of priority events.

## Key Scientific Contributions

The overall progression of the research can be summarised as:

| Stage                | Method                    | Main contribution                                                                                    |
| -------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| Detection            | S2MethTracker             | Physically interpretable Sentinel-2 methane plume detection                                          |
| Segmentation         | N-BPMSNet                 | Automated bi-temporal plume segmentation                                                             |
| Spectral enhancement | MCSSR-Det                 | Methane-sensitive cross-sensor spectral reconstruction                                               |
| Quantification       | MEQNet                    | End-to-end emission-rate estimation using Sentinel-2 and wind                                        |
| System integration   | Multi-satellite framework | Complementary use of multispectral, hyperspectral, meteorological, airborne, and ground observations |

Together, these methods form a progression from **interpretable screening to automated detection, spectral enhancement, and emission quantification**.

## Main Findings

The research demonstrates that multispectral satellite observations can provide useful information for methane point-source monitoring despite their limited spectral sensitivity.

The main findings are:

* Temporal information can suppress persistent background signals and enhance transient methane-related anomalies.
* Spatial plume structure provides important information for distinguishing methane plumes from background features.
* Deep learning can reduce the dependence on manually selected thresholds and improve automated plume segmentation.
* Cross-sensor learning can transfer methane-sensitive spectral information from hyperspectral observations to multispectral imagery.
* Wind information is important for interpreting plume transport and estimating emission rates.
* Combining physical constraints with data-driven learning can improve the usefulness of multispectral observations.
* No single satellite sensor is sufficient for all methane monitoring requirements.

## Limitations

Several limitations remain important for future development.

### Data availability

Confirmed methane plume observations and reliable emission-rate labels remain limited. Real plume annotations can also contain interpretation uncertainty, while paired multispectral-hyperspectral observations are relatively scarce.

### Multispectral spectral limitations

Sentinel-2 has broad spectral bands and was not designed specifically for methane monitoring. Detectability depends on emission strength, surface reflectance, atmospheric conditions, wind, cloud contamination, viewing geometry, and the availability of a suitable reference observation.

Therefore, a non-detection should not automatically be interpreted as evidence that no methane emission occurred.

### Quantification uncertainty

Uncertainty can enter the final emission-rate estimate through:

* Methane column enhancement estimation
* Plume segmentation
* Wind-field uncertainty
* Plume transport assumptions
* Model generalisation
* Reference emission data

These uncertainties should be considered together rather than treating the final emission rate as an exact measurement.

### Generalisation

AI models may experience domain shifts when applied to new:

* Geographic regions
* Surface types
* Source categories
* Atmospheric conditions
* Emission regimes
* Sensor conditions

Further validation using larger independent datasets and controlled-release observations is needed for operational deployment.

## Future Work

Future research directions include:

* Larger and more diverse real-world methane plume datasets
* Independent emission-rate validation
* Controlled-release experiments
* Uncertainty-aware methane detection and quantification
* Improved representation of local wind fields
* Physically constrained AI models
* Quality-aware cross-sensor learning
* Multi-temporal emission monitoring
* Automated satellite data ingestion and quality control
* Integration of multispectral, hyperspectral, airborne, and ground observations
* Source attribution and long-term emission tracking

The long-term goal is not to replace dedicated methane sensors with multispectral imagery, but to develop an integrated observing system in which different sensors provide complementary information.


## Citation

If you use the methods or code from this repository, please cite the corresponding thesis and relevant publications.

```text
Xu, D. Detection and Quantification of Methane Point Sources
Using Multispectral Observations. PhD Thesis,
Imperial College London.
```

Relevant methodological components include:

* S2MethTracker
* N-BPMSNet
* MCSSR-Det
* MEQNet

Please refer to the corresponding publications for detailed methodology, datasets, and experimental settings.

## Acknowledgement

This research was conducted at Imperial College London and focuses on the use of satellite remote sensing, artificial intelligence, and physical information for methane point-source monitoring.

The work demonstrates how multispectral observations can contribute to large-scale methane monitoring when combined with temporal information, spatial context, cross-sensor observations, meteorological data, and physics-guided learning.

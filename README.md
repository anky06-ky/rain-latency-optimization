
# Rainfall Forecasting under Inference Latency Constraint

## Overview

This project investigates hourly rainfall forecasting under a strict
inference latency constraint.

The goal is to identify a forecasting configuration that minimizes
prediction error while satisfying:

Inference latency <= 5 ms

## Research Question

Given historical hourly weather observations, what historical window
length and model configuration produce the lowest forecasting error
while satisfying the latency constraint?

## Optimization Problem

Decision variables:

- Historical window length L
- Model hyperparameters θ

Objective:
Minimize validation forecasting error.

Constraint:
Inference latency <= 5 ms.

## Dataset

Candidate dataset:
Meteostat hourly weather observations from Tan Son Nhat station
(Station ID: 48900).

Planned period:
2015-2025.

Target:
Hourly precipitation (`prcp`) one hour ahead.

## Current Stage

Project initialization and dataset quality assessment.

## Project Structure

- `data/` - raw and processed data
- `notebooks/` - exploratory analysis
- `src/` - reusable source code
- `configs/` - experiment configuration
- `results/` - metrics, figures and optimization logs
- `docs/` - project documentation

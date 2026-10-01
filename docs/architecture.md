# CardioTwin AI — System Architecture

## Data Flow

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│   Static / Historical    │     │   Dynamic / Wearable          │
│   EHR Data               │     │   Time-Series (14 days)       │
│   - Age, sex, BMI         │     │   - HRV (ms)                  │
│   - Blood pressure        │     │   - Resting heart rate        │
│   - Cholesterol           │     │   - Sleep hours               │
│   - Diabetes history      │     │   - Step count                │
│   - Smoking status        │     │                                │
│   - Prior cardiac events  │     │                                │
│   - Family history        │     │                                │
└───────────┬───────────────┘     └───────────────┬────────────────┘
            │                                      │
            │                          ┌───────────▼────────────┐
            │                          │  Feature Engineering    │
            │                          │  - baseline vs recent   │
            │                          │  - slope (trend)        │
            │                          │  - volatility            │
            │                          └───────────┬────────────┘
            │                                      │
            └───────────────────┬──────────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │   Feature Fusion         │
                     │   (EHR + trend vector)   │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │  Trained Model           │
                     │  Logistic Regression     │
                     │  (selected after          │
                     │   comparison vs RF, GBM)  │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │  Risk Classification     │
                     │  Low / Moderate / High    │
                     └───────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                                      │
  ┌───────────▼────────────┐          ┌──────────────▼─────────────┐
  │  Doctor Dashboard        │          │  What-If Simulation Engine  │
  │  - Risk trajectory chart │          │  - perturb wearable trend    │
  │  - Contributing factors  │◄─────────┤    features                 │
  │  - Patient profile       │          │  - re-score via SAME model   │
  └───────────────────────────┘          │    (no retraining)          │
                                          └──────────────────────────────┘
```

## Design Decisions

**Why trend features instead of a deep sequence model (LSTM)?**
Engineered trend features (baseline, recent average, slope, volatility) let a simple, interpretable model capture "trajectory" without requiring large amounts of training data or a black-box architecture. This trades some modeling sophistication for transparency, which matters for a clinical decision-support tool.

**Why is the label personalized rather than population-threshold-based?**
Two patients with the same absolute HRV can have very different clinical significance depending on their own baseline. Deriving risk from each patient's own first-3-day vs. last-3-day drift better mirrors how a digital twin is supposed to reason about *change*, not just current state.

**Why does the what-if engine reuse the trained model instead of a separate simulation model?**
This keeps the simulation honest — the "simulated twin" risk score comes from literally the same decision boundary as the "current twin" score, just with perturbed inputs. There's no separate, unvalidated simulation logic that could diverge from what the model actually learned.

**Why is the dashboard a single static HTML file?**
The trained logistic regression's coefficients, intercepts, and scaler parameters are exported as JSON and the same linear + softmax computation is reimplemented in JavaScript. This means the dashboard runs the *real* trained model in the browser with no backend server, making it trivial to demo or deploy anywhere.

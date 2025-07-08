📘 MSE (Mean Squared Error) — Regression, when large errors are very bad
	•	Use MSE when:
	•	You are doing regression (predicting continuous numbers, like price, temperature, stock values).
	•	You want to punish big mistakes a lot (because squaring errors makes big ones count more).
	•	Common in:
	•	Deep learning loss functions (e.g., for training neural networks on numeric predictions).
	•	Predicting things like house prices, weather forecasting.
	•	Fine-tuning models where large errors are dangerous.

Example: Predicting house prices — if you predict $100K instead of $1M, that’s a big problem. MSE penalizes it heavily.

⸻

📘 MAE (Mean Absolute Error) — Regression, simple average error size
	•	Use MAE when:
	•	You are doing regression, but you want equal treatment of all errors (no squaring).
	•	You want an easily interpretable error: “on average, my prediction is off by X units.”
	•	Common in:
	•	Business forecasting (sales, demand prediction).
	•	Time series forecasting where you care about average deviation.
	•	Anomaly detection (measuring average deviation from normal).

Example: Predicting daily sales numbers — being $20 off is no worse than being $40 off twice.

⸻

📘 SMAPE (Symmetric Mean Absolute Percentage Error) — Forecasting, percentage errors
	•	Use SMAPE when:
	•	You are doing time series forecasting or predicting quantities that vary a lot.
	•	You want scale-independent errors (small values matter as much as large ones).
	•	You need to report percentage-based performance.
	•	Common in:
	•	Energy usage forecasting.
	•	Stock price forecasting.
	•	Sales forecasts across products of very different volumes.

Example: Predicting electricity consumption for different cities — small cities and big cities are compared fairly via SMAPE.

⸻

![alt text](error_brief_table_image.png)
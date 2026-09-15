# Model Explainability Report

| | |
|---|---|
| **Model tested** | `Bank Marketing - Random Forest` |
| **Endpoint** | `http://0.0.0.0:8000/bank_marketing` |
| **Task type** | classification |
| **Data points analyzed** | 50 |
| **Report generated** | 2026-09-15 18:58 |

---

A sensitivity score shows how strongly each factor influences the model's output; a higher score means that factor has a bigger effect on the prediction. The scores provided are normalized to a 0-1 scale, representing each feature's share of total measured sensitivity.

![Feature sensitivity](rf_sensitivity_chart.png)

The following examples show specific real scenarios and what most influenced each prediction.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="rf_sample_case_1.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>age</strong>: 45.00</li><li><strong>balance</strong>: 12,345.00</li><li><strong>day</strong>: 15.00</li><li><strong>campaign</strong>: 3.00</li><li><strong>pdays</strong>: 20.00</li><li><strong>previous</strong>: 0.00</li><li><strong>previously_contacted</strong>: 0.00</li><li><strong>job</strong>: technician</li><li><strong>marital</strong>: married</li><li><strong>education</strong>: tertiary</li><li><strong>default</strong>: no</li><li><strong>housing</strong>: yes</li><li><strong>loan</strong>: no</li><li><strong>contact</strong>: cellular</li><li><strong>month</strong>: jun</li><li><strong>poutcome</strong>: unknown</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted probability:</strong> 0.610</div></div></td><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="rf_sample_case_2.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>age</strong>: 34.00</li><li><strong>balance</strong>: 3,025.00</li><li><strong>day</strong>: 12.00</li><li><strong>campaign</strong>: 2.00</li><li><strong>pdays</strong>: -1.00</li><li><strong>previous</strong>: 0.00</li><li><strong>previously_contacted</strong>: 0.00</li><li><strong>job</strong>: technician</li><li><strong>marital</strong>: single</li><li><strong>education</strong>: tertiary</li><li><strong>default</strong>: no</li><li><strong>housing</strong>: yes</li><li><strong>loan</strong>: no</li><li><strong>contact</strong>: cellular</li><li><strong>month</strong>: aug</li><li><strong>poutcome</strong>: unknown</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted probability:</strong> 0.620</div></div></td></tr></table>

The charts below show how the predicted output changes as each factor varies, while other factors remain typical of the dataset.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_balance.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>balance</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_campaign.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>campaign</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_pdays.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>pdays</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_age.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>age</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_poutcome.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>poutcome</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_month.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>month</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_housing.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>housing</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="rf_feature_impact_loan.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>loan</strong></div></td></tr></table>

These results should be used as a starting point for review and discussion, rather than a final verdict on the model's performance or reliability.
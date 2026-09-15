# Model Explainability Report

| | |
|---|---|
| **Model tested** | `Bank Marketing - 2-layer ANN` |
| **Endpoint** | `http://0.0.0.0:8000/bank_marketing` |
| **Task type** | classification |
| **Data points analyzed** | 50 |
| **Report generated** | 2026-09-15 19:01 |

---

A sensitivity score indicates how strongly each factor influences the model's output; a higher score means that factor has a bigger effect on the prediction. The scores provided below are normalized to a 0-1 scale, representing each feature's share of total measured sensitivity.

![Feature sensitivity](ann_sensitivity_chart.png)

The following examples illustrate specific real scenarios and highlight what most influenced each prediction made by the model.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="ann_sample_case_1.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>age</strong>: 29.00</li><li><strong>balance</strong>: 1,200.00</li><li><strong>day</strong>: 5.00</li><li><strong>campaign</strong>: 2.00</li><li><strong>pdays</strong>: 10.00</li><li><strong>previous</strong>: 1.00</li><li><strong>previously_contacted</strong>: 0.00</li><li><strong>job</strong>: blue-collar</li><li><strong>marital</strong>: single</li><li><strong>education</strong>: secondary</li><li><strong>default</strong>: no</li><li><strong>housing</strong>: yes</li><li><strong>loan</strong>: no</li><li><strong>contact</strong>: cellular</li><li><strong>month</strong>: jan</li><li><strong>poutcome</strong>: success</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted probability:</strong> 0.998</div></div></td><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="ann_sample_case_2.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>age</strong>: 47.00</li><li><strong>balance</strong>: 5,000.00</li><li><strong>day</strong>: 10.00</li><li><strong>campaign</strong>: 3.00</li><li><strong>pdays</strong>: 30.00</li><li><strong>previous</strong>: 0.00</li><li><strong>previously_contacted</strong>: 0.00</li><li><strong>job</strong>: management</li><li><strong>marital</strong>: married</li><li><strong>education</strong>: secondary</li><li><strong>default</strong>: yes</li><li><strong>housing</strong>: no</li><li><strong>loan</strong>: yes</li><li><strong>contact</strong>: telephone</li><li><strong>month</strong>: mar</li><li><strong>poutcome</strong>: failure</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted probability:</strong> 0.859</div></div></td></tr></table>

The charts below demonstrate how the predicted output changes as each factor varies, while other factors remain typical of the dataset.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_campaign.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>campaign</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_balance.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>balance</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_previously_contacted.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>previously_contacted</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_day.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>day</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_loan.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>loan</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_poutcome.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>poutcome</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_contact.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>contact</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_month.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>month</strong></div></td></tr></table>

These results should be used as a starting point for review and further analysis, rather than a final verdict on the model's performance.
# Model Explainability Report

| | |
|---|---|
| **Model tested** | `House Price - 2-layer ANN` |
| **Endpoint** | `http://0.0.0.0:8000/house_prices` |
| **Task type** | regression |
| **Data points analyzed** | 50 |
| **Report generated** | 2026-09-15 18:29 |

---

A sensitivity score shows how strongly each factor influences the model's output; a higher score means that factor has a bigger effect on the prediction. The scores are normalized to a 0-1 scale representing each feature's share of total measured sensitivity.

![Feature sensitivity](ann_sensitivity_chart.png)

The following examples show specific real scenarios and what most influenced each prediction.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="ann_sample_case_1.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>area</strong>: 3,920.00</li><li><strong>bedrooms</strong>: 3.00</li><li><strong>bathrooms</strong>: 2.00</li><li><strong>stories</strong>: 2.00</li><li><strong>parking</strong>: 1.00</li><li><strong>mainroad</strong>: yes</li><li><strong>guestroom</strong>: no</li><li><strong>basement</strong>: no</li><li><strong>hotwaterheating</strong>: yes</li><li><strong>airconditioning</strong>: no</li><li><strong>prefarea</strong>: yes</li><li><strong>furnishingstatus</strong>: furnished</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted value:</strong> 6,357,290.80</div></div></td><td style="width:50%; padding:8px; vertical-align:top;"><div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;"><img src="ann_sample_case_2.png" style="width:100%;"><div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div><ul style="font-size:12px; margin:4px 0; padding-left:18px;"><li><strong>area</strong>: 7,487.00</li><li><strong>bedrooms</strong>: 3.00</li><li><strong>bathrooms</strong>: 2.00</li><li><strong>stories</strong>: 1.00</li><li><strong>parking</strong>: 1.00</li><li><strong>mainroad</strong>: yes</li><li><strong>guestroom</strong>: no</li><li><strong>basement</strong>: yes</li><li><strong>hotwaterheating</strong>: yes</li><li><strong>airconditioning</strong>: no</li><li><strong>prefarea</strong>: yes</li><li><strong>furnishingstatus</strong>: furnished</li></ul><div style="font-size:13px; margin-top:6px;"><strong>Predicted value:</strong> 7,525,502.54</div></div></td></tr></table>

The charts below show how the predicted output changes as each factor varies, while other factors remain typical of the dataset.

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_area.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>area</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_bathrooms.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>bathrooms</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_stories.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>stories</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_parking.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>parking</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_airconditioning.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>airconditioning</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_mainroad.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>mainroad</strong></div></td></tr></table>

<table style="width:100%; border-collapse:collapse;"><tr><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_hotwaterheating.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>hotwaterheating</strong></div></td><td style="width:50%; padding:8px; vertical-align:top;"><img src="ann_feature_impact_prefarea.png" style="width:100%;"><div style="font-size:13px; text-align:center; margin-top:4px;"><strong>prefarea</strong></div></td></tr></table>

These results should be used as a starting point for review and further analysis, rather than a final verdict on the model's performance or predictions.
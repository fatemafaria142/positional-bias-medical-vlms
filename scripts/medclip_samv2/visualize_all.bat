@echo off
REM Regenerate the Original | Mask | Ground Truth | Prediction comparison images
REM for all 3 datasets from their current Results/medclip_samv2/<dataset>_results.csv.
cd /d "%~dp0..\.."
python scripts\medclip_samv2\visualize_medclip_predictions.py acdc
python scripts\medclip_samv2\visualize_medclip_predictions.py isic_task2
python scripts\medclip_samv2\visualize_medclip_predictions.py chaos

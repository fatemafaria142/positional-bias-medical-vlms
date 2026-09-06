@echo off
REM Run the MedCLIP-SAMv2 zero-shot pipeline on all 3 datasets in sequence (each resumable).
REM Safe to interrupt (Ctrl+C) and re-run later -- already-completed rows are skipped.
cd /d "%~dp0..\.."
"models\MedCLIP-SAMv2\.venv\Scripts\python.exe" scripts\medclip_samv2\run_medclip_samv2.py acdc %*
"models\MedCLIP-SAMv2\.venv\Scripts\python.exe" scripts\medclip_samv2\run_medclip_samv2.py isic_task2 %*
"models\MedCLIP-SAMv2\.venv\Scripts\python.exe" scripts\medclip_samv2\run_medclip_samv2.py chaos %*

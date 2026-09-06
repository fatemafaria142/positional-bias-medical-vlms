@echo off
REM Run the MedCLIP-SAMv2 zero-shot pipeline on the CHAOS dataset (resumable).
REM Add extra flags after the script if needed, e.g.: run_chaos.bat --batch-size 10
cd /d "%~dp0..\.."
"models\MedCLIP-SAMv2\.venv\Scripts\python.exe" scripts\medclip_samv2\run_medclip_samv2.py chaos %*

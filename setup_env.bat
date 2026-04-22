@echo off
echo ============================================
echo  Setup Environment - Poli-SEG Project
echo ============================================

:: Aktifkan virtual environment
call venv\Scripts\activate.bat

echo [1/3] Upgrade pip...
python -m pip install --upgrade pip

echo [2/3] Install PyTorch dengan CUDA 12.6...
pip install torch==2.7.0+cu126 torchaudio==2.7.0+cu126 torchvision==0.22.0+cu126 --index-url https://download.pytorch.org/whl/cu126

echo [3/3] Install package lainnya...
pip install -r requirements-no-torch.txt

echo.
echo ============================================
echo  Setup selesai! Jalankan dengan: run.bat
echo ============================================
pause

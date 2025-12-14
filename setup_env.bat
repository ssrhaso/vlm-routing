python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision transformers timm datasets wandb pytest black flake8
pip freeze > requirements.txt

.venv\Scripts\activate.ps1
echo DONE!

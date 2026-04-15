cp .env.example .env
cd models
git clone https://huggingface.co/answerdotai/ModernBERT-large
git clone https://huggingface.co/HuggingFaceTB/SmolLM2-360M
echo "Cloned models. You still need to download the SRBH dataset if you plan to evaluate it (https://doi.org/10.7910/DVN/OGOIXX)"
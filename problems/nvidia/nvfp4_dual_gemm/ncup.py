# ncup.py
from vllm_omni.entrypoints.omni import Omni
import torch

def main():
    omni = Omni(
        model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        stage_configs_path="vllm_omni/model_executor/stage_configs/qwen3_tts.yaml",
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Warmup
    omni.generate("Warmup.")

    # Profile window
    torch.cuda.cudart().cudaProfilerStart()
    omni.generate("The quick brown fox jumps over the lazy dog.")
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()
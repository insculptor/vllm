from src.models.vllm_model_manager import VLLMModelManager


class InferenceService:
    def __init__(self):
        self.model_manager = VLLMModelManager()

    def run_inference(self, input_text: str):
        result = self.model_manager.generate(input_text)
        return result

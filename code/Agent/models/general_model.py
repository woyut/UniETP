class GeneralModel:
    def __init__(
        self, model_name: str, model_config_file: str, difficulty_mode: int = 3
    ):
        self.model_name = model_name
        self.model_config = model_config_file
        self.difficulty_mode = difficulty_mode

        self.stats = {
            "calls": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_time": 0.0,
        }

    def set_difficulty_mode(self, difficulty_mode: int):
        self.difficulty_mode = difficulty_mode

    def generate_response(
        self,
        current_obs_img_path: str,
        action_history: list[str],
        feedback_history: list[str],
        action_success_history: list[bool],
        scene_info: dict[str, list],
        action_space: dict[str, list[str]],
        instruction: str = None,
    ) -> tuple[list, dict, str]:
        raise NotImplementedError

    def parse_llm_output(self, output: str) -> str:
        raise NotImplementedError

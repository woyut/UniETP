"""
Hosted and OpenAI-compatible multimodal model APIs.

Credentials are read from provider-specific environment variables. The optional
model config controls generation parameters and local OpenAI-compatible servers.
"""

from __future__ import annotations

import re
import os
import time
import json
import traceback
from typing import Any, Optional

from json_repair import repair_json

from Agent.prompt.planner_prompt import (
    SYSTEM_PROMPT_MODE1,
    SYSTEM_PROMPT_MODE2,
    SYSTEM_PROMPT_MODE3,
    USER_PROMPT,
    ActionListSchema,
    schema_to_actions_and_reasoning,
)
from Agent.models.general_model import GeneralModel

from Agent.models.model_utils import (
    build_actions_prompt,
    _image_to_data,
    _probe_image_size,
    _load_optional_json_config,
)

DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_VLLM_BASE_URL = "http://localhost:9000/v1"


class APIModel(GeneralModel):
    """Chat Completions + optional vision; compatible with OpenAI SDK (base_url + api_key)."""

    def __init__(
        self,
        model_name: str,
        model_config_file: Optional[str] = None,
        difficulty_mode: int = 3,
        use_feedback: bool = False,
    ):
        self.model_name = (model_name or "").strip() or os.environ.get(
            "OPENAI_MODEL", "gpt-5.5"
        )
        self._cfg = _load_optional_json_config(model_config_file)
        self.provider = self._infer_provider(self.model_name, self._cfg)
        self.difficulty_mode = difficulty_mode
        self.use_feedback = use_feedback
        timeout = float(self._cfg.get("timeout", 120.0))

        if self.provider == "vllm":
            from openai import OpenAI

            base_url = (
                self._cfg.get("base_url")
                or os.environ.get("VLLM_BASE_URL")
                or DEFAULT_VLLM_BASE_URL
            )
            self.client = OpenAI(api_key="dummy", base_url=base_url, timeout=1000)
        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            base_url = self._cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL")
            client_kwargs = {"api_key": api_key, "timeout": timeout}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
        elif self.provider == "google":
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            base_url = (
                self._cfg.get("base_url")
                or os.environ.get("GOOGLE_BASE_URL")
                or DEFAULT_GOOGLE_BASE_URL
            )
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    base_url=base_url.rsplit("/v1beta", 1)[0],
                    api_version="v1beta",
                ),
            )
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            base_url = (
                self._cfg.get("base_url")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or DEFAULT_ANTHROPIC_BASE_URL
            )
            self.client = Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            raise ValueError(f"Unknown model provider: {self.provider}")

        self.stats = {
            "calls": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total_time": 0.0,
        }

    def set_difficulty_mode(self, difficulty_mode: int) -> None:
        self.difficulty_mode = int(difficulty_mode)

    @staticmethod
    def _infer_provider(model_name: str, cfg: Optional[dict[str, Any]] = None) -> str:
        name = model_name or ""
        cfg = cfg or {}
        cfg_provider = str(cfg.get("provider", "")).strip().lower()
        explicit = (os.environ.get("UNIETP_MODEL_PROVIDER") or "").strip().lower()
        if explicit in {"openai", "google", "anthropic", "vllm"}:
            return explicit
        if cfg_provider in {"openai", "google", "anthropic", "vllm"}:
            return cfg_provider

        if name.startswith("gemini"):
            return "google"
        if name.startswith("claude"):
            return "anthropic"
        if name.startswith("gpt"):
            return "openai"
        if name.startswith("qwen"):
            return "openai"
        if name.startswith("kimi"):
            return "openai"
        raise ValueError(f"Unknown model name: {name}")

    def _call_vllm(
        self, system_prompt: str, user_prompt: str, current_obs_img_path: str
    ):
        json_schema = ActionListSchema.model_json_schema()

        mime, data = _image_to_data(current_obs_img_path)
        data_url = f"data:{mime};base64,{data}"

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            temperature=float(self._cfg.get("temperature", 0.2)),
            max_tokens=int(self._cfg.get("max_tokens", 2048)),
            extra_body={"structured_outputs": {"json": json_schema}},
        )

        usage = resp.usage
        self.stats["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        self.stats["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.stats["output_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

        out_details = getattr(usage, "completion_tokens_details", None)
        self.stats["thinking_tokens"] += int(
            getattr(out_details, "reasoning_tokens", 0) or 0
        )

        in_details = getattr(usage, "prompt_tokens_details", None)
        self.stats["cache_read"] += int(getattr(in_details, "cached_tokens", 0) or 0)

        if resp.choices[0].finish_reason == "length":
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                temperature=float(self._cfg.get("temperature", 0.2)),
                max_tokens=int(self._cfg.get("max_tokens", 2048)),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "structured_outputs": {"json": json_schema},
                },
            )
            usage = resp.usage
            self.stats["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
            self.stats["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.stats["output_tokens"] += int(
                getattr(usage, "completion_tokens", 0) or 0
            )

            out_details = getattr(usage, "completion_tokens_details", None)
            self.stats["thinking_tokens"] += int(
                getattr(out_details, "reasoning_tokens", 0) or 0
            )

            in_details = getattr(usage, "prompt_tokens_details", None)
            self.stats["cache_read"] += int(
                getattr(in_details, "cached_tokens", 0) or 0
            )

        return resp

    def _parse_structured_output_vllm(self, resp):
        raw = resp.choices[0].message.content or ""
        try:
            schema = ActionListSchema.model_validate_json(raw)

            return schema_to_actions_and_reasoning(schema)
        except Exception:
            pass

        try:
            cleaned = self._strip_json_markdown(raw)
            schema = ActionListSchema.model_validate_json(cleaned)
            return schema_to_actions_and_reasoning(schema)
        except Exception:
            pass

        try:
            return self._parse_structured_output_anthropic_fallback(raw)
        except Exception as e:
            raise ValueError(
                f"[vLLM] All JSON parse attempts failed.\n"
                f"Raw output: {raw!r}\n"
                f"Finish: {resp.choices[0].finish_reason}, Usage: {resp.usage}\n"
                f"{resp.choices[0].message.model_extra}"
                f"Last error: {e}"
            )

    def _call_openai(
        self, system_prompt: str, user_prompt: str, current_obs_img_path: str
    ):
        mime, data = _image_to_data(current_obs_img_path)
        data_url = f"data:{mime};base64,{data}"
        assert data_url is not None

        if "kimi" in self.model_name:
            user_content = [
                {"type": "image_url", "image_url": data_url},
                {"type": "text", "text": user_prompt},
            ]
            resp = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=float(self._cfg.get("temperature", 0.2)),
                max_tokens=int(self._cfg.get("max_tokens", 2048)),
                response_format=ActionListSchema,
            )
        else:
            user_content = [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": user_prompt},
            ]
            resp = self.client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=float(self._cfg.get("temperature", 0.2)),
                max_output_tokens=int(self._cfg.get("max_tokens", 2048)),
                text_format=ActionListSchema,
            )

        usage = resp.usage
        self.stats["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        self.stats["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        self.stats["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)

        out_details = getattr(usage, "output_tokens_details", None)
        self.stats["thinking_tokens"] += int(
            getattr(out_details, "reasoning_tokens", 0) or 0
        )

        in_details = getattr(usage, "input_tokens_details", None)
        self.stats["cache_read"] += int(getattr(in_details, "cached_tokens", 0) or 0)
        return resp

    @staticmethod
    def _extract_openai_text(raw_content) -> str:
        if hasattr(raw_content, "choices"):
            msg = raw_content.choices[0].message
            return (getattr(msg, "content", None) or "").strip()
        return (getattr(raw_content, "output_text", None) or "").strip()

    @staticmethod
    def _openai_llm_out_for_log(raw_content) -> str:
        """Return a JSON-serializable string for llm_io logging."""
        if (
            hasattr(raw_content, "output_parsed")
            and raw_content.output_parsed is not None
        ):
            parsed = raw_content.output_parsed
            if hasattr(parsed, "model_dump_json"):
                return parsed.model_dump_json()
            return json.dumps(parsed, ensure_ascii=False)
        if hasattr(raw_content, "choices"):
            parsed = raw_content.choices[0].message.parsed
            if parsed is not None:
                if hasattr(parsed, "model_dump_json"):
                    return parsed.model_dump_json()
                return json.dumps(parsed, ensure_ascii=False)
        text = APIModel._extract_openai_text(raw_content)
        return text or repr(raw_content)

    def _parse_structured_output_openai(self, raw_content):
        if "kimi" in self.model_name:
            parsed = raw_content.choices[0].message.parsed
            if parsed is not None:
                return schema_to_actions_and_reasoning(parsed)
            raw_text = self._extract_openai_text(raw_content)
        else:
            parsed = raw_content.output_parsed
            if parsed is not None:
                return schema_to_actions_and_reasoning(parsed)
            raw_text = self._extract_openai_text(raw_content)

        if not raw_text:
            raise ValueError("OpenAI response has no parsed output or text content")

        try:
            schema = ActionListSchema.model_validate_json(
                self._strip_json_markdown(raw_text)
            )
            return schema_to_actions_and_reasoning(schema)
        except Exception:
            pass

        return self._parse_structured_output_anthropic_fallback(raw_text)

    def _call_google(
        self, system_prompt: str, user_prompt: str, current_obs_img_path: str
    ) -> str:

        with open(current_obs_img_path, "rb") as f:
            image_bytes = f.read()
        if current_obs_img_path.endswith(".jpg"):
            media_type = "image/jpeg"
        elif current_obs_img_path.endswith(".png"):
            media_type = "image/png"
        else:
            raise ValueError(f"Unsupported image format: {current_obs_img_path}")
        from google.genai import types
        from google.genai.types import FinishReason

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=media_type,
        )
        text_part = types.Part.from_text(text=user_prompt)

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[image_part, text_part],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=float(self._cfg.get("temperature", 0.2)),
                max_output_tokens=int(self._cfg.get("max_tokens", 2048)),
                response_mime_type="application/json",
                response_json_schema=ActionListSchema.model_json_schema(),
            ),
        )

        if resp.candidates[0].finish_reason == FinishReason.MAX_TOKENS:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[image_part, text_part],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=float(self._cfg.get("temperature", 0.2)),
                    max_output_tokens=int(self._cfg.get("max_tokens", 2048)),
                    response_mime_type="application/json",
                    response_json_schema=ActionListSchema.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        thinking_level="low",
                    ),
                ),
            )

        m = resp.usage_metadata
        input_tokens = getattr(m, "prompt_token_count", 0) or 0
        output_tokens = getattr(m, "candidates_token_count", 0) or 0
        thinking_tokens = getattr(m, "thoughts_token_count", 0) or 0
        cache_read = getattr(m, "cached_content_token_count", 0) or 0
        total_tokens = getattr(m, "total_token_count", 0) or (
            input_tokens + output_tokens + thinking_tokens
        )
        self.stats["total_tokens"] += total_tokens
        self.stats["input_tokens"] += input_tokens
        self.stats["output_tokens"] += output_tokens
        self.stats["thinking_tokens"] += thinking_tokens
        self.stats["cache_read"] += cache_read

        return resp

    def _parse_structured_output_google(self, raw_content):
        def strip_json_markdown(text: str) -> str:
            text = text.strip()

            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return match.group(1).strip()
            return text

        text = strip_json_markdown(raw_content.text)
        try:
            schema = ActionListSchema.model_validate_json(text)
            return schema_to_actions_and_reasoning(schema)
        except Exception:
            # Gemini may return a single action object without an "actions" array.
            return self._parse_structured_output_anthropic_fallback(text)

    def _build_anthropic_user_content(
        self, user_prompt: str, current_obs_img_path: str
    ):
        mime, data = _image_to_data(current_obs_img_path)
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,  # e.g. "image/jpeg"
                    "data": data,
                },
            },
            {
                "type": "text",
                "text": user_prompt,
            },
        ]

    @staticmethod
    def _extract_anthropic_text(raw_content) -> str:
        if raw_content is None:
            return ""
        text = getattr(raw_content, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        blocks = getattr(raw_content, "content", None) or []
        parts = []
        for block in blocks:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                continue
            if getattr(block, "type", None) == "text" and isinstance(
                getattr(block, "text", None), str
            ):
                parts.append(block.text)
        return "\n".join(parts).strip()

    @staticmethod
    def _strip_json_markdown(text: str) -> str:
        text = (text or "").strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def _loads_json_loose(text: str):
        """
        Parse JSON from mixed model output.
        Supports cases like: "<reasoning text>\\n\\n{...json...}".
        """
        cleaned = APIModel._strip_json_markdown(text)
        decoder = json.JSONDecoder()
        last_error = None

        # Fast path: pure JSON
        try:
            return decoder.decode(cleaned)
        except json.JSONDecodeError as e:
            last_error = e

        # Loose path: find the first decodable JSON object/array in text
        for i, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[i:])
                return payload
            except json.JSONDecodeError as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Expecting value", cleaned, 0)

    @staticmethod
    def _loads_json_loose2(text: str):
        cleaned = APIModel._strip_json_markdown(text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for i, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                repaired = repair_json(cleaned[i:])
                return json.loads(repaired)
            except Exception:
                continue

        raise json.JSONDecodeError("Expecting value", cleaned, 0)

    def _parse_structured_output_anthropic_fallback(self, raw_text: str):

        payload = self._loads_json_loose2(raw_text)
        if isinstance(payload, dict):
            if "actions" in payload:
                payload = {
                    "reasoning": payload.get("reasoning", ""),
                    "actions": payload["actions"],
                }
            elif "action_name" in payload and "args" in payload:
                payload = {
                    "reasoning": payload.get("reasoning", ""),
                    "actions": [payload],
                }
            else:
                raise ValueError("Unexpected anthropic JSON object shape")
        elif isinstance(payload, list):
            payload = {"reasoning": "", "actions": payload}
        else:
            raise ValueError(
                f"Unexpected anthropic JSON type: {type(payload).__name__}"
            )

        schema = ActionListSchema.model_validate(payload)
        return schema_to_actions_and_reasoning(schema)

    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, current_obs_img_path: str
    ):

        user_content = self._build_anthropic_user_content(
            user_prompt, current_obs_img_path
        )
        response = self.client.messages.create(
            model=self.model_name,
            temperature=float(self._cfg.get("temperature", 0.2)),
            max_tokens=int(self._cfg.get("max_tokens", 2048)),
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content},
            ],
        )

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens
        self.stats["total_tokens"] += total_tokens
        self.stats["input_tokens"] += input_tokens
        self.stats["output_tokens"] += output_tokens

        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.stats["cache_read"] += cache_read
        self.stats["cache_write"] += cache_write

        return response

    def generate_response(
        self,
        current_obs_img_path: str,
        action_history: list[str],
        feedback_history: list[str],
        action_success_history: list[bool],
        scene_info: dict[str, Any],
        action_space: dict[str, list[str]],
        instruction: Optional[str] = None,
    ) -> list[list]:
        image_size = _probe_image_size(current_obs_img_path)

        actions_prompt = build_actions_prompt(action_space)
        if self.difficulty_mode == 1:
            if self.provider == "google":
                bbox_format = "[y1, x1, y2, x2]"
                pixel_format = "[y, x]"
                pixel_type = "normalized_1000"  # https://ai.google.dev/gemini-api/docs/image-understanding
                pixel_normalization = "All values should be normalized to an integer in [0, 1000] where 1000 = full image width/height."
                actions_prompt = build_actions_prompt(
                    action_space, pixel_type=pixel_type
                )
            elif self.provider == "anthropic":
                bbox_format = "[x1, y1, x2, y2]"
                pixel_format = "[x, y]"
                pixel_type = "absolute"  # https://platform.claude.com/docs/en/build-with-claude/vision
                pixel_normalization = "All values should be **integers** representing **absolute** pixel coordinates."
                actions_prompt = build_actions_prompt(
                    action_space, pixel_type=pixel_type
                )
            elif self.provider == "vllm" and "Qwen" in self.model_name:
                bbox_format = "[x1, y1, x2, y2]"
                pixel_format = "[x, y]"
                pixel_type = "normalized_1000"
                pixel_normalization = "All values should be normalized to an integer in [0, 1000] where 1000 = full image width/height."
                actions_prompt = build_actions_prompt(
                    action_space, pixel_type=pixel_type
                )
            else:
                bbox_format = "[x1, y1, x2, y2]"
                pixel_format = "[x, y]"
                pixel_type = "normalized_1"
                pixel_normalization = "All values should be normalized to a float in [0, 1] where 1 = full image width/height."
                actions_prompt = build_actions_prompt(
                    action_space, pixel_type=pixel_type
                )
            system_prompt = SYSTEM_PROMPT_MODE1.format(
                actions=actions_prompt,
                bbox_format=bbox_format,
                pixel_format=pixel_format,
                pixel_normalization=pixel_normalization,
            )
        elif self.difficulty_mode == 2:
            system_prompt = SYSTEM_PROMPT_MODE2.format(actions=actions_prompt)
        elif self.difficulty_mode == 3:
            system_prompt = SYSTEM_PROMPT_MODE3.format(actions=actions_prompt)
        else:
            raise ValueError(f"Unknown difficulty mode: {self.difficulty_mode}")

        info = ""
        if "scene_rooms_list" in scene_info:
            info += "Rooms in the scene:\n"
            info += str(scene_info["scene_rooms_list"]) + "\n"
        objs = scene_info.get("scene_big_objects_list", []) + scene_info.get(
            "scene_small_objects_list", []
        )
        if len(objs):
            info += "Objects in the scene:\n"
            info += str(objs) + "\n"
        if "in_view_objects_bbox" in scene_info:
            info += "Objects currently in view:\n"
            info += str(scene_info["in_view_objects_bbox"]) + "\n"

        history = ""
        assert len(action_history) == len(action_success_history)
        assert len(action_history) == len(feedback_history)
        for i in range(len(action_history)):
            history += f"Action {i + 1}: {action_history[i]}\n"
            if self.use_action_success:
                history += f"Action success: {action_success_history[i]}\n"
            if (
                self.use_feedback
                and action_success_history[i] == False
                and feedback_history[i] != ""
            ):
                history += f"Feedback: {feedback_history[i]}\n"

        user_prompt = USER_PROMPT.format(
            task_instruction=instruction,
            scene_info=info,
            action_history=history,
        )
        llm_in = {
            "system": system_prompt,
            "user": user_prompt,
            "image_path": current_obs_img_path,
        }

        t0 = time.perf_counter()

        if self.provider == "vllm":
            try:
                raw_content = self._call_vllm(
                    system_prompt, user_prompt, current_obs_img_path
                )
            except Exception as e:
                print(f"[vLLM] Failed to generate response: {e}")
                print(traceback.format_exc())
                return [], llm_in, ""
            try:
                actions, reasoning = self._parse_structured_output_vllm(raw_content)
            except Exception as e:
                print(f"[vLLM] Failed to parse JSON: {e}")
                print(traceback.format_exc())
                raw_text = (
                    (raw_content.choices[0].message.content or "")
                    if raw_content
                    else ""
                )
                return [], llm_in, raw_text

        elif self.provider == "anthropic":
            try:
                raw_content = self._call_anthropic(
                    system_prompt, user_prompt, current_obs_img_path
                )
            except Exception as e:
                print(f"[AnthropicAPIModel] Failed to generate response: {e}")
                print(traceback.format_exc())
                return [], llm_in, ""
            raw_text = self._extract_anthropic_text(raw_content)
            try:
                actions, reasoning = self._parse_structured_output_anthropic_fallback(
                    raw_text
                )
            except Exception as e:
                print(f"[AnthropicAPIModel] Failed to parse JSON: {e}")
                print(traceback.format_exc())
                print(f"[AnthropicAPIModel] Raw text: {raw_text}")
                return [], llm_in, raw_text
        elif self.provider == "openai":
            try:
                raw_content = self._call_openai(
                    system_prompt, user_prompt, current_obs_img_path
                )
            except Exception as e:
                print(f"[APIModel] Failed to generate response: {e}")
                return [], llm_in, ""
            try:
                actions, reasoning = self._parse_structured_output_openai(raw_content)
            except Exception as e:
                raw_text = self._openai_llm_out_for_log(raw_content)
                print(f"[APIModel] Failed to parse JSON: {e}")
                print(f"[APIModel] Raw content: {raw_text}")
                return [], llm_in, raw_text
        elif self.provider == "google":
            try:
                raw_content = self._call_google(
                    system_prompt, user_prompt, current_obs_img_path
                )
            except Exception as e:
                print(f"[GoogleAPIModel] Failed to generate response: {e}")
                return [], llm_in, ""
            try:
                actions, reasoning = self._parse_structured_output_google(raw_content)
            except Exception as e:
                print(f"[GoogleAPIModel] Failed to parse JSON: {e}")
                print(f"[GoogleAPIModel] Raw content: {raw_content.text}")
                return [], llm_in, raw_content.text

            #         a, b, c, d = action[1]["object_bbox"]["bbox_xyxy"]
            #         action[1]["object_bbox"]["bbox_xyxy"] = [b, a, d, c]

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.difficulty_mode == 1:
            for action in actions:
                if "bbox_xyxy" in action[1].get("object_bbox", {}):
                    a, b, c, d = action[1]["object_bbox"]["bbox_xyxy"]
                    if self.provider == "google":
                        bbox = [b / 1000.0, a / 1000.0, d / 1000.0, c / 1000.0]
                    elif self.provider == "anthropic":
                        bbox = [
                            a / image_size[0],
                            b / image_size[1],
                            c / image_size[0],
                            d / image_size[1],
                        ]
                    elif self.provider == "vllm" and "Qwen" in self.model_name:
                        bbox = [a / 1000.0, b / 1000.0, c / 1000.0, d / 1000.0]
                    else:
                        bbox = [a, b, c, d]
                    action[1]["object_bbox"]["bbox_xyxy_real"] = bbox
                if "pixel" in action[1]:
                    x, y = action[1]["pixel"]
                    if self.provider == "google":
                        pixel = [y / 1000.0, x / 1000.0]
                    elif self.provider == "anthropic":
                        pixel = [x / image_size[0], y / image_size[1]]
                    elif self.provider == "vllm" and "Qwen" in self.model_name:
                        pixel = [x / 1000.0, y / 1000.0]
                    else:
                        pixel = [x, y]
                    action[1]["pixel_real"] = pixel

        content = {"actions": actions, "reasoning": reasoning}

        elapsed = time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["total_time"] += elapsed

        return actions, llm_in, content

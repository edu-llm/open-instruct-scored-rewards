import torch

from projects.pedagogy_mech_interp.trace import tokenize_record


class Encoding(dict):
    def __getattr__(self, name):
        return self[name]


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = "|".join(f"{message['role']}:{message['content']}" for message in messages if message["role"] != "assistant")
        prefix += "|assistant:"
        assistant = next((message["content"] for message in messages if message["role"] == "assistant"), None)
        if assistant is None:
            return prefix
        return prefix + assistant + "<eot>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
        data = Encoding(
            input_ids=torch.tensor([[ord(character) for character in text]]),
            attention_mask=torch.ones(1, len(text), dtype=torch.long),
        )
        if return_offsets_mapping:
            data["offset_mapping"] = torch.tensor([[(index, index + 1) for index in range(len(text))]])
        return data


def record():
    return {
        "variant_id": "v1",
        "question": "Q",
        "student_before": "S",
        "tutor_turn": "TARGET",
    }


def test_character_offsets_locate_exact_tutor_span_after_left_truncation():
    tokenizer = CharacterTokenizer()
    encoded = tokenize_record(tokenizer, record(), max_len=20)
    content = encoded.input_ids[0, encoded.content_start : encoded.content_stop]
    assert "".join(chr(value) for value in content.tolist()) == "TARGET"


def test_partial_response_truncation_fails_closed():
    tokenizer = CharacterTokenizer()
    try:
        tokenize_record(tokenizer, record(), max_len=8)
    except ValueError as exc:
        assert "removed part" in str(exc)
    else:
        raise AssertionError("partial tutor truncation should fail")

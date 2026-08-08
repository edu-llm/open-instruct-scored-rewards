import torch
from datasets import IterableDataset
from projects.olmoe_full_finetune.midtrain import (
    PackedTokenStream,
    TokenMixtureStream,
    get_last_complete_checkpoint,
    router_z_loss,
)


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


def make_source(texts=("abc", "defgh", "ijklmnop")):
    return IterableDataset.from_generator(lambda: ({"text": text} for text in texts))


def test_packed_stream_resumes_at_exact_token():
    stream = PackedTokenStream(make_source(), TinyTokenizer(), sequence_length=4)
    iterator = iter(stream)
    first = next(iterator)
    state = stream.state_dict()
    expected_next = next(iterator)

    resumed_stream = PackedTokenStream(make_source(), TinyTokenizer(), sequence_length=4)
    resumed_stream.load_state_dict(state)
    actual_next = next(iter(resumed_stream))

    assert first["input_ids"].tolist() == [97, 98, 99, 0]
    assert actual_next["input_ids"].tolist() == expected_next["input_ids"].tolist()
    assert actual_next["labels"].tolist() == actual_next["input_ids"].tolist()


def test_router_z_loss_averages_layers_and_tokens():
    logits = (torch.zeros(2, 4), torch.zeros(3, 4))

    loss = router_z_loss(logits)

    assert torch.isclose(loss, torch.tensor(4.0).log().square())


def test_token_mixture_resumes_sources_and_rng():
    def make_mixture():
        return TokenMixtureStream(
            [
                PackedTokenStream(make_source(("aaaa", "aaaa")), TinyTokenizer(), sequence_length=2),
                PackedTokenStream(make_source(("bbbb", "bbbb")), TinyTokenizer(), sequence_length=2),
            ],
            probabilities=[0.25, 0.75],
            seed=7,
        )

    mixture = make_mixture()
    iterator = iter(mixture)
    prefix = [next(iterator)["input_ids"].tolist() for _ in range(3)]
    state = mixture.state_dict()
    expected = [next(iterator)["input_ids"].tolist() for _ in range(3)]

    resumed_mixture = make_mixture()
    resumed_mixture.load_state_dict(state)
    resumed_iterator = iter(resumed_mixture)
    actual = [next(resumed_iterator)["input_ids"].tolist() for _ in range(3)]

    assert prefix
    assert actual == expected


def test_resume_ignores_newer_incomplete_checkpoint(tmp_path):
    complete = tmp_path / "step_20"
    complete.mkdir()
    (complete / "COMPLETED").write_text("COMPLETED\n")
    (tmp_path / "step_40").mkdir()

    assert get_last_complete_checkpoint(tmp_path) == complete

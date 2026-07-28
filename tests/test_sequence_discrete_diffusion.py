from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_design_model import (  # noqa: E402
    MASK_TOKEN_ID,
    MSASequenceDiffusionModel,
    SequenceDiffusionDecoder,
    STOP_TOKEN_ID,
    TOKEN_TO_ID,
    decode_tokens_until_stop,
    encode_sequence_with_stop,
)
from scripts.train_sequence_decoder import (  # noqa: E402
    SequenceCollator,
    compatible_checkpoint_state_dict,
    curriculum_timestep_range,
)
from scripts.train_mean_start_ccdd_from_cached_msas import (  # noqa: E402
    MeanStartCCDDModel,
    target_row_residue_embeddings,
)


def test_mask_token_is_special_and_not_encoded_as_target() -> None:
    tokens, _ = encode_sequence_with_stop("ACD", max_length=6)
    assert MASK_TOKEN_ID not in tokens.tolist()
    decoded = decode_tokens_until_stop(
        [
            TOKEN_TO_ID["A"],
            MASK_TOKEN_ID,
            TOKEN_TO_ID["C"],
            STOP_TOKEN_ID,
            TOKEN_TO_ID["D"],
        ]
    )
    assert decoded == "AC"


def test_discrete_mask_corruption_uses_absorbing_mask_token() -> None:
    decoder = SequenceDiffusionDecoder(
        d_model=8,
        max_sequence_length=6,
        num_layers=1,
        num_heads=2,
        num_timesteps=4,
        dropout=0.0,
    )
    target_tokens = torch.tensor(
        [
            [TOKEN_TO_ID["A"], TOKEN_TO_ID["C"], TOKEN_TO_ID["D"], STOP_TOKEN_ID, STOP_TOKEN_ID, STOP_TOKEN_ID],
            [TOKEN_TO_ID["E"], TOKEN_TO_ID["F"], TOKEN_TO_ID["G"], TOKEN_TO_ID["H"], STOP_TOKEN_ID, STOP_TOKEN_ID],
        ],
        dtype=torch.long,
    )
    corrupted, corruption_mask = decoder.discrete_corrupt_tokens(
        target_tokens=target_tokens,
        timesteps=torch.tensor([3, 0], dtype=torch.long),
        mode="discrete_mask",
    )

    assert corruption_mask[0].all()
    assert torch.equal(corrupted[0], torch.full_like(corrupted[0], MASK_TOKEN_ID))
    assert torch.equal(corrupted[corruption_mask], torch.full_like(corrupted[corruption_mask], MASK_TOKEN_ID))
    assert torch.equal(corrupted[~corruption_mask], target_tokens[~corruption_mask])


def test_discrete_forward_weights_corrupted_positions_only() -> None:
    torch.manual_seed(5)
    decoder = SequenceDiffusionDecoder(
        d_model=8,
        max_sequence_length=12,
        num_layers=1,
        num_heads=2,
        num_timesteps=10,
        dropout=0.0,
    )
    latent_tokens = torch.randn(1, 4, 8)
    latent_mask = torch.ones(1, 4, dtype=torch.bool)
    target_tokens = torch.randint(0, MASK_TOKEN_ID, (1, 12), dtype=torch.long)
    loss_weights = torch.ones_like(target_tokens, dtype=torch.float32)
    loss_weights[:, -2:] = 0.0

    outputs = decoder(
        latent_tokens=latent_tokens,
        latent_mask=latent_mask,
        target_tokens=target_tokens,
        loss_weights=loss_weights,
        timesteps=torch.tensor([4], dtype=torch.long),
        decoder_start_mode="discrete_mask",
        discrete_loss_corrupted_only=True,
    )

    expected = loss_weights * outputs["corruption_mask"].to(dtype=loss_weights.dtype)
    if torch.sum(expected) > 0:
        assert torch.equal(outputs["token_loss_weights"], expected)
    assert outputs["token_loss"].ndim == 0
    assert outputs["full_token_accuracy"].ndim == 0


def test_noisy_mean_start_noises_mean_embedding_not_target() -> None:
    torch.manual_seed(9)
    decoder = SequenceDiffusionDecoder(
        d_model=8,
        max_sequence_length=4,
        num_layers=1,
        num_heads=2,
        num_timesteps=4,
        dropout=0.0,
    )
    target_tokens = torch.tensor(
        [[TOKEN_TO_ID["A"], TOKEN_TO_ID["C"], TOKEN_TO_ID["D"], STOP_TOKEN_ID]],
        dtype=torch.long,
    )
    clean_embeddings = decoder.token_embedding(target_tokens)
    timesteps = torch.tensor([3], dtype=torch.long)
    noise = torch.ones_like(clean_embeddings)

    state = decoder.decoder_start_state(
        clean_embeddings=clean_embeddings,
        timesteps=timesteps,
        mode="noisy_mean",
        noise=noise,
    )

    mean_embedding = decoder.token_embedding.weight.mean(dim=0).view(1, 1, -1)
    mean_embeddings = mean_embedding.expand_as(clean_embeddings)
    expected = decoder.q_sample(mean_embeddings, timesteps, noise=noise)
    target_based = decoder.q_sample(clean_embeddings, timesteps, noise=noise)
    assert torch.allclose(state["noisy_embeddings"], expected)
    assert not torch.allclose(state["noisy_embeddings"], target_based)


def test_noisy_mean_leaves_condition_memory_unchanged() -> None:
    torch.manual_seed(13)
    model = MSASequenceDiffusionModel(
        input_dim=5,
        d_model=8,
        max_sequence_length=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_timesteps=4,
        numeric_condition_fields=("kcat_1_per_s",),
        condition_layers=0,
    )
    model.eval()
    token_embeddings = torch.randn(2, 3, 5, 5)
    aa_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    target_tokens = torch.randint(0, MASK_TOKEN_ID, (2, 6), dtype=torch.long)
    loss_weights = torch.ones(2, 6)
    condition_values = torch.randn(2, 1)
    condition_mask = torch.ones(2, 1, dtype=torch.bool)
    timesteps = torch.tensor([3, 3], dtype=torch.long)

    common_inputs = {
        "token_embeddings": token_embeddings,
        "aa_mask": aa_mask,
        "target_tokens": target_tokens,
        "loss_weights": loss_weights,
        "timesteps": timesteps,
        "condition_values": condition_values,
        "condition_mask": condition_mask,
    }
    mean_outputs = model(**common_inputs, decoder_start_mode="mean")
    noisy_outputs = model(**common_inputs, decoder_start_mode="noisy_mean")

    assert torch.allclose(noisy_outputs["latent_tokens"], mean_outputs["latent_tokens"])
    assert torch.equal(noisy_outputs["latent_mask"], mean_outputs["latent_mask"])
    assert torch.allclose(noisy_outputs["condition_tokens"], mean_outputs["condition_tokens"])
    assert not torch.allclose(noisy_outputs["noisy_embeddings"], mean_outputs["noisy_embeddings"])


def test_timestep_curriculum_ramps_start_range_to_final_range() -> None:
    start_range = (0, 49)
    final_range = (200, 249)

    assert curriculum_timestep_range(1, final_range, start_range, curriculum_epochs=5) == (0, 49)
    assert curriculum_timestep_range(3, final_range, start_range, curriculum_epochs=5) == (100, 149)
    assert curriculum_timestep_range(5, final_range, start_range, curriculum_epochs=5) == final_range
    assert curriculum_timestep_range(9, final_range, start_range, curriculum_epochs=5) == final_range
    assert curriculum_timestep_range(1, final_range, start_range, curriculum_epochs=0) == final_range


def test_collator_preserves_target_continuous_before_masking_msa() -> None:
    token_embeddings = np.zeros((2, 4, 5), dtype=np.float32)
    token_embeddings[1, 0] = np.arange(5, dtype=np.float32) + 1.0
    token_embeddings[1, 1] = np.arange(5, dtype=np.float32) + 11.0
    token_embeddings[1, 2] = np.arange(5, dtype=np.float32) + 21.0
    aa_mask = np.zeros((2, 4), dtype=np.bool_)
    aa_mask[:, :3] = True
    collator = SequenceCollator(
        max_sequence_length=6,
        tail_stop_weight=0.05,
        mask_target_row_in_msa=True,
    )

    batch = collator(
        [
            {
                "token_embeddings": token_embeddings,
                "aa_mask": aa_mask,
                "target_sequence": "ACD",
                "condition_values": np.zeros((0,), dtype=np.float32),
                "condition_mask": np.zeros((0,), dtype=np.bool_),
                "categorical_condition_ids": tuple(),
                "embedding_path": "example.npz",
                "metadata_path": "",
                "row_index": 1,
            }
        ]
    )

    assert torch.all(batch["token_embeddings"][0, 1] == 0)
    assert not batch["aa_mask"][0, 1].any()
    assert torch.allclose(batch["target_continuous_embeddings"][0, 0], torch.from_numpy(token_embeddings[1, 0]))
    assert torch.allclose(batch["target_continuous_embeddings"][0, 1], torch.from_numpy(token_embeddings[1, 1]))
    assert batch["target_continuous_mask"][0, :3].all()
    assert not batch["target_continuous_mask"][0, 3:].any()


def test_mean_start_target_row_residue_embeddings_are_extracted_before_masking() -> None:
    token_embeddings = np.zeros((2, 4, 5), dtype=np.float32)
    token_embeddings[1, 0] = np.arange(5, dtype=np.float32) + 1.0
    token_embeddings[1, 2] = np.arange(5, dtype=np.float32) + 21.0
    token_embeddings[1, 3] = np.arange(5, dtype=np.float32) + 31.0
    aa_mask = np.zeros((2, 4), dtype=np.bool_)
    aa_mask[1, [0, 2, 3]] = True
    item = {
        "npz_path": "example.npz",
        "token_embeddings": token_embeddings,
        "aa_mask": aa_mask,
        "sequences": ["----", "A-CD"],
    }

    embeddings, mask = target_row_residue_embeddings(item, row_index=1, target_sequence="ACD")

    assert embeddings.shape == (3, 5)
    assert mask.tolist() == [True, True, True]
    assert torch.allclose(torch.from_numpy(embeddings[0]), torch.from_numpy(token_embeddings[1, 0]))
    assert torch.allclose(torch.from_numpy(embeddings[1]), torch.from_numpy(token_embeddings[1, 2]))


def test_mean_start_target_row_embedding_continuous_objective() -> None:
    torch.manual_seed(31)
    model = MeanStartCCDDModel(
        row_embedding_dim=5,
        d_model=8,
        layers=1,
        heads=2,
        dropout=0.0,
        max_sequence_length=6,
        diffusion_timesteps=4,
        category_buckets=32,
        memory_mode="profile_only",
        continuous_target_mode="target_row_embedding",
        target_continuous_dim=5,
    )
    profiles = torch.zeros(1, 6, 22)
    profile_mask = torch.ones(1, 6, dtype=torch.bool)
    target_tokens = torch.tensor(
        [[TOKEN_TO_ID["A"], TOKEN_TO_ID["C"], TOKEN_TO_ID["D"], STOP_TOKEN_ID, STOP_TOKEN_ID, STOP_TOKEN_ID]],
        dtype=torch.long,
    )
    loss_weights = torch.ones(1, 6)
    sequence_loss_weights = loss_weights.clone()
    target_continuous_embeddings = torch.randn(1, 6, 5)
    target_continuous_mask = torch.zeros(1, 6, dtype=torch.bool)
    target_continuous_mask[:, :3] = True

    outputs = model(
        profiles=profiles,
        profile_mask=profile_mask,
        row_embeddings=torch.zeros(1, 0, 5),
        row_mask=torch.zeros(1, 0, dtype=torch.bool),
        msa_embeddings=torch.zeros(1, 0, 6, 1),
        msa_embedding_mask=torch.zeros(1, 0, 6, dtype=torch.bool),
        numeric_values=torch.zeros(1, 5),
        numeric_mask=torch.zeros(1, 5, dtype=torch.bool),
        category_ids=torch.full((1, 4), -1, dtype=torch.long),
        category_mask=torch.zeros(1, 4, dtype=torch.bool),
        target_tokens=target_tokens,
        loss_weights=loss_weights,
        sequence_loss_weights=sequence_loss_weights,
        timesteps=torch.tensor([2], dtype=torch.long),
        decoder_start_mode="noisy_mean",
        target_continuous_embeddings=target_continuous_embeddings,
        target_continuous_mask=target_continuous_mask,
    )

    assert outputs["predicted_continuous_embeddings"].shape == target_continuous_embeddings.shape
    assert outputs["weighted_continuous_loss"].ndim == 0
    assert torch.isclose(outputs["target_continuous_mask_fraction"], torch.tensor(0.5))


def test_ccdd_continuous_stream_erases_masked_positions_before_noise() -> None:
    torch.manual_seed(29)
    decoder = SequenceDiffusionDecoder(
        d_model=8,
        max_sequence_length=4,
        num_layers=1,
        num_heads=2,
        num_timesteps=4,
        dropout=0.0,
    )
    latent_tokens = torch.randn(1, 3, 8)
    latent_mask = torch.ones(1, 3, dtype=torch.bool)
    target_tokens = torch.tensor(
        [[TOKEN_TO_ID["A"], TOKEN_TO_ID["C"], TOKEN_TO_ID["D"], STOP_TOKEN_ID]],
        dtype=torch.long,
    )
    continuous_targets = torch.randn(1, 4, 8)
    zero_noise = torch.zeros_like(continuous_targets)

    outputs = decoder(
        latent_tokens=latent_tokens,
        latent_mask=latent_mask,
        target_tokens=target_tokens,
        loss_weights=torch.ones(1, 4),
        timesteps=torch.tensor([3], dtype=torch.long),
        decoder_start_mode="discrete_mask",
        ccdd_continuous_targets=continuous_targets,
        ccdd_continuous_mask=torch.ones(1, 4, dtype=torch.bool),
        ccdd_continuous_noise=zero_noise,
        ccdd_continuous_timestep_scale=1.0,
    )

    assert outputs["corruption_mask"].all()
    assert torch.allclose(outputs["ccdd_leakage_safe_continuous_targets"], torch.zeros_like(continuous_targets))
    assert torch.allclose(outputs["ccdd_noisy_continuous"], torch.zeros_like(continuous_targets))
    assert outputs["ccdd_continuous_loss"].ndim == 0


def test_latent_codiffusion_adds_target_free_sequence_latents() -> None:
    torch.manual_seed(23)
    model = MSASequenceDiffusionModel(
        input_dim=5,
        d_model=8,
        max_sequence_length=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_timesteps=5,
        numeric_condition_fields=("kcat_1_per_s",),
        condition_layers=0,
        latent_codiffusion_tokens=3,
    )
    model.eval()
    token_embeddings = torch.randn(2, 3, 5, 5)
    aa_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    target_tokens = torch.randint(0, MASK_TOKEN_ID, (2, 8), dtype=torch.long)
    loss_weights = torch.ones(2, 8)
    condition_values = torch.randn(2, 1)
    condition_mask = torch.ones(2, 1, dtype=torch.bool)

    outputs = model(
        token_embeddings=token_embeddings,
        aa_mask=aa_mask,
        target_tokens=target_tokens,
        loss_weights=loss_weights,
        timesteps=torch.tensor([4, 4], dtype=torch.long),
        condition_values=condition_values,
        condition_mask=condition_mask,
        decoder_start_mode="noisy_mean",
    )

    assert outputs["latent_loss"].ndim == 0
    assert outputs["clean_sequence_latents"].shape == (2, 3, 8)
    assert outputs["predicted_sequence_latents"].shape == (2, 3, 8)
    assert outputs["sequence_memory_tokens"].shape[1] == outputs["latent_tokens"].shape[1] + 3
    assert torch.equal(outputs["sequence_memory_mask"][:, 3:], outputs["latent_mask"])
    assert not torch.allclose(outputs["noisy_sequence_latents"], outputs["clean_sequence_latents"])


def test_discrete_sampler_starts_masked_and_finishes_without_mask_tokens() -> None:
    torch.manual_seed(11)
    decoder = SequenceDiffusionDecoder(
        d_model=8,
        max_sequence_length=7,
        num_layers=1,
        num_heads=2,
        num_timesteps=5,
        dropout=0.0,
    )
    latent_tokens = torch.randn(2, 3, 8)
    latent_mask = torch.ones(2, 3, dtype=torch.bool)

    outputs = decoder.sample_discrete(
        latent_tokens=latent_tokens,
        latent_mask=latent_mask,
        steps=4,
        temperature=0.0,
    )

    assert outputs["tokens"].shape == (2, 7)
    assert outputs["logits"].shape == (2, 7, decoder.vocab_size)
    assert not torch.any(outputs["tokens"] == MASK_TOKEN_ID)


def test_condition_dropout_replaces_memory_with_null_token() -> None:
    torch.manual_seed(17)
    model = MSASequenceDiffusionModel(
        input_dim=5,
        d_model=8,
        max_sequence_length=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_timesteps=4,
        condition_layers=0,
    )
    model.train()
    token_embeddings = torch.randn(2, 3, 5, 5)
    aa_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    target_tokens = torch.randint(0, MASK_TOKEN_ID, (2, 6), dtype=torch.long)
    loss_weights = torch.ones(2, 6)

    outputs = model(
        token_embeddings=token_embeddings,
        aa_mask=aa_mask,
        target_tokens=target_tokens,
        loss_weights=loss_weights,
        timesteps=torch.tensor([3, 3], dtype=torch.long),
        decoder_start_mode="discrete_mask",
        condition_dropout=1.0,
    )

    assert outputs["condition_drop_mask"].all()
    assert outputs["latent_mask"][:, 0].all()
    assert not outputs["latent_mask"][:, 1:].any()


def test_old_checkpoint_vocab_heads_expand_for_mask_token() -> None:
    model = MSASequenceDiffusionModel(
        input_dim=5,
        d_model=8,
        max_sequence_length=6,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_timesteps=4,
        condition_layers=0,
    )
    old_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for key in (
        "decoder.token_embedding.weight",
        "decoder.lm_head.weight",
        "decoder.lm_head.bias",
    ):
        old_state[key] = old_state[key][:-1].clone()
    old_state.pop("null_condition_token")

    compatible_state, notes = compatible_checkpoint_state_dict(model, old_state)
    result = model.load_state_dict(compatible_state, strict=False)

    assert "null_condition_token" in result.missing_keys
    assert any("decoder.token_embedding.weight" in note for note in notes)
    assert compatible_state["decoder.token_embedding.weight"].shape[0] == MASK_TOKEN_ID + 1
    assert compatible_state["decoder.lm_head.weight"].shape[0] == MASK_TOKEN_ID + 1


if __name__ == "__main__":
    for test in (
        test_mask_token_is_special_and_not_encoded_as_target,
        test_discrete_mask_corruption_uses_absorbing_mask_token,
        test_discrete_forward_weights_corrupted_positions_only,
        test_noisy_mean_start_noises_mean_embedding_not_target,
        test_noisy_mean_leaves_condition_memory_unchanged,
        test_timestep_curriculum_ramps_start_range_to_final_range,
        test_collator_preserves_target_continuous_before_masking_msa,
        test_mean_start_target_row_residue_embeddings_are_extracted_before_masking,
        test_mean_start_target_row_embedding_continuous_objective,
        test_ccdd_continuous_stream_erases_masked_positions_before_noise,
        test_latent_codiffusion_adds_target_free_sequence_latents,
        test_discrete_sampler_starts_masked_and_finishes_without_mask_tokens,
        test_condition_dropout_replaces_memory_with_null_token,
        test_old_checkpoint_vocab_heads_expand_for_mask_token,
    ):
        test()
    print("sequence_discrete_diffusion_tests=ok")

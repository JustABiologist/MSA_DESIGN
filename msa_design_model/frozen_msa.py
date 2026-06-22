"""Frozen ESM MSA Transformer encoder wrapper.

The training script uses precomputed embeddings by default, but this wrapper keeps the
encoder machinery in-tree and makes the frozen-boundary explicit for future end-to-end
experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


CONTACT_REGRESSION_URL_TEMPLATE = (
    "https://dl.fbaipublicfiles.com/fair-esm/regression/"
    "{model_name}-contact-regression.pt"
)


def contact_regression_path(weights_path: Path) -> Path:
    """Return fair-esm's expected sidecar path for local model loading."""
    return Path(str(weights_path.with_suffix("")) + "-contact-regression.pt")


class FrozenMSATransformerEncoder:
    """Small frozen wrapper around fair-esm's ESM-MSA-1b model.

    Parameters are set to ``requires_grad=False`` and calls to :meth:`encode` run under
    ``torch.no_grad()``. The wrapper returns token representations aligned to cleaned
    MSA columns, with BOS/EOS tokens removed when fair-esm includes them.
    """

    def __init__(
        self,
        weights_path: str | Path = "weights/esm_msa1b_t12_100M_UR50S.pt",
        layer: int = 12,
        device: str = "auto",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.layer = layer
        if not self.weights_path.exists():
            raise FileNotFoundError(f"MSA Transformer weights not found: {self.weights_path}")
        regression_path = contact_regression_path(self.weights_path)
        if not regression_path.exists():
            model_name = self.weights_path.stem
            url = CONTACT_REGRESSION_URL_TEMPLATE.format(model_name=model_name)
            raise FileNotFoundError(
                "fair-esm local loading expects the contact-regression sidecar next to the model weights. "
                f"Missing: {regression_path}. Download: {url}"
            )

        try:
            import torch
            import esm
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("FrozenMSATransformerEncoder requires torch and fair-esm/esm") from exc

        self.torch = torch
        self.esm = esm
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(str(self.weights_path))
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.batch_converter = self.alphabet.get_batch_converter()

    def encode(self, msa_records: Sequence[tuple[str, str]]) -> Any:
        """Encode one aligned MSA and return ``rows x cols x hidden`` tensor on CPU.

        ``msa_records`` should be a sequence of ``(header, aligned_sequence)`` pairs.
        Sequences must already be cleaned/aligned, matching the behavior of
        ``scripts/embed_msas.py``.
        """
        if not msa_records:
            raise ValueError("msa_records must contain at least one sequence")
        cols = len(msa_records[0][1])
        if any(len(sequence) != cols for _, sequence in msa_records):
            raise ValueError("all MSA records must have the same aligned length")

        _, _, tokens = self.batch_converter([list(msa_records)])
        tokens = tokens.to(self.device)
        with self.torch.no_grad():
            results = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
            representations = results["representations"][self.layer]
            token_repr = representations[0]
            if token_repr.shape[1] == cols + 1:
                token_repr = token_repr[:, 1:, :]
            elif token_repr.shape[1] == cols + 2:
                token_repr = token_repr[:, 1:-1, :]
            elif token_repr.shape[1] != cols:
                raise RuntimeError(
                    f"cannot align representation length {token_repr.shape[1]} to MSA columns {cols}"
                )
            return token_repr.detach().cpu()

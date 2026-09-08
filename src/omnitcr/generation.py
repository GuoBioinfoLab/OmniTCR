"""SFT generation and manuscript-compatible PMI reranking."""

import torch
import torch.nn.functional as F

from .config import GenerationConfig


MANUSCRIPT_GENERATION_CONFIG = GenerationConfig()
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_tcr_sequence(sequence):
    if sequence is None:
        return ""
    sequence = str(sequence).strip().upper()
    for token in (
        "[TRB]", "<TRB>", "[BOS]", "[EOS]", "<BOS>", "<EOS>",
    ):
        sequence = sequence.replace(token, "")
    return (
        sequence.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
    )


def is_valid_sft_cdr3(sequence):
    sequence = clean_tcr_sequence(sequence)
    return (
        7 <= len(sequence) <= 24
        and sequence.startswith("C")
        and sequence.endswith(("F", "W"))
    )


def is_valid_pmi_cdr3(sequence):
    sequence = clean_tcr_sequence(sequence)
    return (
        is_valid_sft_cdr3(sequence)
        and all(amino_acid in STANDARD_AMINO_ACIDS for amino_acid in sequence)
    )


class TCRGenerator:
    """Generate CDR3beta sequences with SFT or PMI inference logic."""

    def __init__(
        self,
        model,
        tokenizer,
        input_builder,
        mode="sft",
        settings=MANUSCRIPT_GENERATION_CONFIG,
    ):
        mode = str(mode).strip().lower()
        if mode not in {"sft", "pmi"}:
            raise ValueError("Generation mode must be 'sft' or 'pmi'.")
        self.model = model
        self.tokenizer = tokenizer
        self.input_builder = input_builder
        self.mode = mode
        self.pad_token_id = tokenizer.pad_token_id
        self.settings = settings

    def input_device(self):
        return next(self.model.parameters()).device

    def _encode_ids(self, text):
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = encoding["input_ids"] if isinstance(encoding, dict) else encoding
        if torch.is_tensor(input_ids):
            input_ids = input_ids.tolist()
        return input_ids

    def _decode_generation(self, sequence_ids, prompt_length):
        generated_ids = sequence_ids[prompt_length:]
        text = self.tokenizer.decode(
            generated_ids.tolist(),
            skip_special_tokens=True,
        )
        return clean_tcr_sequence(text)

    def _beam_generate(self, input_ids, num_return_sequences):
        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.input_device(),
        )
        attention_mask = torch.ones_like(input_tensor)
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                num_beams=self.settings.num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                repetition_penalty=self.settings.repetition_penalty,
                length_penalty=self.settings.length_penalty,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def _sft_generate(self, input_ids, num_sequences):
        if num_sequences > self.settings.num_beams:
            raise ValueError(
                "SFT generation supports at most "
                f"{self.settings.num_beams} sequences."
            )
        outputs = self._beam_generate(input_ids, num_sequences)
        generated = []
        for sequence_ids in outputs:
            tcr = self._decode_generation(sequence_ids, len(input_ids))
            if is_valid_sft_cdr3(tcr):
                generated.append(tcr)
        return generated

    def _pmi_candidates(self, input_ids):
        outputs = self._beam_generate(
            input_ids, self.settings.pmi_candidates
        )
        candidates = []
        seen = set()
        for sequence_ids in outputs:
            tcr = self._decode_generation(sequence_ids, len(input_ids))
            if not is_valid_pmi_cdr3(tcr) or tcr in seen:
                continue
            seen.add(tcr)
            candidates.append(
                {"tcr": tcr, "beam_rank": len(candidates) + 1}
            )
        return candidates

    @torch.no_grad()
    def _score_tcrs_avg_logp(self, prompt, tcrs, batch_size=32):
        if not tcrs:
            return []
        if not prompt.endswith("[TRB]"):
            prompt += "[TRB]"

        prompt_ids = [self.tokenizer.bos_token_id] + self._encode_ids(prompt)
        prompt_length = len(prompt_ids)
        all_scores = []

        for start in range(0, len(tcrs), batch_size):
            batch_tcrs = tcrs[start : start + batch_size]
            full_batch = []
            label_batch = []
            mask_batch = []

            for tcr in batch_tcrs:
                continuation_ids = self._encode_ids(
                    clean_tcr_sequence(tcr) + "[TRB]"
                )
                full_ids = prompt_ids + continuation_ids
                labels = [-100] * prompt_length + continuation_ids
                full_batch.append(full_ids)
                label_batch.append(labels)
                mask_batch.append([1] * len(full_ids))

            maximum = max(len(sequence) for sequence in full_batch)
            for index in range(len(full_batch)):
                pad_length = maximum - len(full_batch[index])
                full_batch[index] += [self.pad_token_id] * pad_length
                label_batch[index] += [-100] * pad_length
                mask_batch[index] += [0] * pad_length

            input_tensor = torch.tensor(
                full_batch, dtype=torch.long, device=self.input_device()
            )
            label_tensor = torch.tensor(
                label_batch, dtype=torch.long, device=self.input_device()
            )
            attention_tensor = torch.tensor(
                mask_batch, dtype=torch.long, device=self.input_device()
            )

            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_tensor,
                return_dict=True,
            )
            shift_logits = outputs.logits.float()[:, :-1, :]
            shift_labels = label_tensor[:, 1:]
            valid_mask = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_log_probs = token_log_probs * valid_mask.float()
            lengths = valid_mask.float().sum(dim=1).clamp(min=1.0)
            scores = token_log_probs.sum(dim=1) / lengths
            all_scores.extend(scores.detach().cpu().tolist())

        return all_scores

    def _pmi_generate(self, prompt, input_ids, num_sequences):
        if num_sequences > self.settings.pmi_candidates:
            raise ValueError(
                "PMI generation supports at most "
                f"{self.settings.pmi_candidates} sequences."
            )
        candidates = self._pmi_candidates(input_ids)
        if not candidates:
            return []
        tcrs = [candidate["tcr"] for candidate in candidates]
        target_scores = self._score_tcrs_avg_logp(
            prompt,
            tcrs,
            batch_size=self.settings.pmi_scoring_batch_size,
        )
        null_scores = self._score_tcrs_avg_logp(
            self.settings.null_pmhc_prompt,
            tcrs,
            batch_size=self.settings.pmi_scoring_batch_size,
        )

        ranked = []
        for candidate, target_score, null_score in zip(
            candidates, target_scores, null_scores
        ):
            ranked.append(
                {
                    **candidate,
                    "target_avg_logp": float(target_score),
                    "pmi_score": float(
                        target_score - self.settings.pmi_alpha * null_score
                    ),
                }
            )
        ranked.sort(
            key=lambda item: (
                item["pmi_score"],
                item["target_avg_logp"],
                -item["beam_rank"],
            ),
            reverse=True,
        )
        return [item["tcr"] for item in ranked[:num_sequences]]

    def generate(self, peptide, mhc, num_sequences=100):
        if (
            isinstance(num_sequences, bool)
            or not isinstance(num_sequences, int)
            or num_sequences < 1
        ):
            raise ValueError("num_sequences must be a positive integer.")
        prompt, input_ids = self.input_builder.encode_prompt(peptide, mhc)
        maximum_positions = getattr(
            self.model.config, "max_position_embeddings", 128
        )
        if len(input_ids) + self.settings.max_new_tokens > maximum_positions:
            raise ValueError(
                "Prompt plus max_new_tokens exceeds the model's positional "
                f"limit of {maximum_positions}."
            )
        if self.mode == "sft":
            return self._sft_generate(input_ids, num_sequences)
        return self._pmi_generate(prompt, input_ids, num_sequences)

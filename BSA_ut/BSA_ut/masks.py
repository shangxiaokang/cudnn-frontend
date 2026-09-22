"""Packed text/video/audio masks with 1x8x8 video cubes and 64-token blocks.

production.json defines segment lengths and video latent shapes.
Video block selection uses Q/K block means; gather, coarse attention and
gating are outside this benchmark.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


BLOCK_ELEMENTS = 64
SPARSITY = 0.1
FIXTURE_PATH = Path(__file__).with_name("production.json")


@dataclass(frozen=True)
class Segment:
    sample: int
    kind: str
    length: int
    latent_shape: tuple[int, int, int] | None = None

    @property
    def is_video(self):
        return self.kind in ("video", "ref_video")


def compute_cube_sizes(latent_shape, device):
    """Cube order: time, height tile, width tile; valid tokens precede padding."""
    t, h, w = latent_shape
    h_sizes = torch.full(((h + 7) // 8,), 8, dtype=torch.int32, device=device)
    w_sizes = torch.full(((w + 7) // 8,), 8, dtype=torch.int32, device=device)
    h_sizes[-1].fill_((h - 1) % 8 + 1)
    w_sizes[-1].fill_((w - 1) % 8 + 1)
    return (h_sizes[:, None] * w_sizes[None, :]).flatten().repeat(t)


def build_geometry(samples, device):
    """Lay out each sample's segments and define dense/Top-K candidate ranges."""
    segments, entries, ranges = [], [], []
    groups, references = {}, {}
    cube_offset = 0
    for sample_id, sample in enumerate(samples):
        parts = [("text", sample["text_tokens"], None)]
        for kind, key in (
            ("video", "target_latent_shape"),
            ("ref_video", "reference_latent_shape"),
        ):
            shape = sample[key]
            if shape is not None:
                t, h, w = shape
                parts.append((kind, t * h * w, tuple(shape)))
        parts.append(("audio", sample["audio_tokens"], None))
        for kind, length, shape in parts:
            if length <= 0:
                raise ValueError("Every fixture segment must contain real tokens")
            segment = Segment(sample_id, kind, length, shape)
            if segment.is_video:
                sizes = compute_cube_sizes(shape, device)
            else:
                sizes = torch.full(
                    ((length + 63) // 64,), 64, dtype=torch.int32, device=device
                )
                sizes[-1].fill_((length - 1) % 64 + 1)
            end = cube_offset + sizes.numel()
            ranges.append((cube_offset, end))
            group = groups.setdefault(sample_id, {})
            group.setdefault("all", [cube_offset, end])[1] = end
            group[kind] = (cube_offset, end)
            if kind == "ref_video":
                references.setdefault(sample_id, []).append(len(segments))
            segments.append(segment)
            entries.append({"cube_sizes": sizes})
            cube_offset = end

    # Text is dense within its sample; video selects from the ranges below.
    for i, (segment, entry) in enumerate(zip(segments, entries)):
        group = groups[segment.sample]
        text = group["text"]
        candidate_ids = ()
        if segment.kind == "text":
            dense = (tuple(group["all"]),)
        elif segment.kind == "video":
            dense = (text,)
            candidate_ids = (i, *references.get(segment.sample, ()))
        elif segment.kind == "ref_video":
            dense = (text,)
            candidate_ids = (i,)
        else:  # Target audio; this fixture has no reference audio/images.
            dense = (text, group["audio"])
        entry["dense_intervals"] = dense
        entry["candidate_ranges"] = tuple(ranges[j] for j in candidate_ids)
        count = sum(hi - lo for lo, hi in entry["candidate_ranges"])
        entry["topk"] = min(count, max(1, round(SPARSITY * count)))
    return {
        "segments": tuple(segments),
        "entries": tuple(entries),
        "ranges": tuple(ranges),
        "sample_ranges": tuple(tuple(g["all"]) for g in groups.values()),
        "cube_sizes": torch.cat([entry["cube_sizes"] for entry in entries]),
        "total_cubes": cube_offset,
    }


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def valid_token_mask(block_sizes):
    positions = torch.arange(BLOCK_ELEMENTS, device=block_sizes.device)
    return (positions[None, :] < block_sizes[:, None]).flatten()


def block_means(x, block_sizes):
    """Average valid tokens in FP32, then cast back to the input dtype."""
    b, s, h, d = x.shape
    valid = valid_token_mask(block_sizes).view(1, s, 1, 1)
    blocks = x.masked_fill(~valid, 0).reshape(b, -1, BLOCK_ELEMENTS, h, d)
    means = blocks.sum(dim=2, dtype=torch.float32)
    means = means / block_sizes.view(1, -1, 1, 1)
    return means.to(x.dtype).transpose(1, 2).contiguous()


@torch.no_grad()
def build_cube_mask(q, k, geometry):
    """Score valid-token block means; break ties by smaller candidate ID."""
    m = geometry["total_cubes"]
    mask = torch.zeros(q.shape[0], q.shape[2], m, m, dtype=torch.bool, device=q.device)
    q_mean = block_means(q, geometry["cube_sizes"])
    k_mean = block_means(k, geometry["cube_sizes"])
    for entry, (lo, hi) in zip(geometry["entries"], geometry["ranges"]):
        for k0, k1 in entry["dense_intervals"]:
            mask[:, :, lo:hi, k0:k1] = True
        if not entry["candidate_ranges"]:
            continue
        candidates = torch.cat(
            [
                torch.arange(k0, k1, device=q.device)
                for k0, k1 in entry["candidate_ranges"]
            ]
        )
        scores = torch.matmul(
            q_mean[:, :, lo:hi], k_mean[:, :, candidates].transpose(-2, -1)
        ) / (q.shape[-1] ** 0.5)
        selected = scores.argsort(dim=-1, descending=True, stable=True)[
            ..., : entry["topk"]
        ]
        mask[:, :, lo:hi].scatter_(-1, candidates[selected], True)
    return mask


def build_bsa_metadata(q, k, geometry):
    cube_mask = build_cube_mask(q, k, geometry)
    # Each row keeps all M block IDs: sorted active IDs form the prefix;
    # q2k_block_nums gives its length. The remaining IDs are ignored by BSA.
    q2k_block_index = cube_mask.argsort(dim=-1, descending=True, stable=True).to(
        torch.int32
    )
    q2k_block_nums = cube_mask.sum(dim=-1, dtype=torch.int32)
    return {
        "q2k_block_index": q2k_block_index,
        "q2k_block_nums": q2k_block_nums,
        "block_sizes": geometry["cube_sizes"],
    }


def block_causal_metadata(seqlen, heads, device):
    blocks = (seqlen + 63) // 64
    ids = torch.arange(blocks, device=device, dtype=torch.int32)
    return {
        "q2k_block_index": ids.view(1, 1, 1, blocks).expand(1, heads, blocks, blocks).contiguous(),
        "q2k_block_nums": (ids + 1).view(1, 1, blocks).expand(1, heads, blocks).contiguous(),
        "block_sizes": (seqlen - ids * 64).clamp(max=64),
    }


@torch.no_grad()
def count_visible_pairs(metadata):
    """Count valid Q/K token pairs across all batches/heads, excluding padding."""
    indices = metadata["q2k_block_index"]
    counts = metadata["q2k_block_nums"]
    sizes = metadata["block_sizes"].to(torch.int64)
    slots = torch.arange(indices.shape[-1], device=indices.device)
    total = torch.zeros((), dtype=torch.int64, device=indices.device)
    # Limit temporary memory; inactive index suffixes contribute no work.
    for b in range(indices.shape[0]):
        for h in range(indices.shape[1]):
            for lo in range(0, indices.shape[2], 128):
                hi = min(lo + 128, indices.shape[2])
                keys = indices[b, h, lo:hi].long()
                active = slots[None, :] < counts[b, h, lo:hi, None]
                total += (sizes[lo:hi, None] * sizes[keys] * active).sum()
    return total.item()

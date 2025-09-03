# pytorch_all2all.py
import os
import torch
import torch.distributed as dist
import dataclasses
from task import input_t, output_t


# ---------------- MoE config ----------------
@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16


# ---------------- data per dp rank ----------------
class RankTestData:
    def __init__(self, cfg: MoEConfig, rng: torch.Generator, rank: int):
        device = torch.device(f"cuda:{rank}")
        self.num_tokens = int(
            torch.randint(
                1, cfg.max_num_tokens, [1], generator=rng, device=device
            ).item()
        )
        # token expert map
        self.indices = torch.empty(
            self.num_tokens, cfg.experts_per_token, dtype=torch.int32, device=device
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(cfg.num_experts, generator=rng, device=device)
            self.indices[i] = perm[: cfg.experts_per_token]
        # topk weights
        self.weights = torch.rand(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.float32,
            generator=rng,
            device=device,
        )
        # dp tokens, input of dispatch
        self.x = torch.randn(
            self.num_tokens,
            cfg.hidden_dim,
            dtype=cfg.in_dtype,
            generator=rng,
            device=device,
        )


# ---------------- All2All pytorch impl ----------------
class PyTorchAllToAll:
    META_DIM = 5  # global_exp, src_rank, src_token, src_k, pad

    def __init__(self, cfg: MoEConfig, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # num experts per rank
        self.num_local_experts = cfg.num_experts // world_size
        # max recv tokens per rank
        self.max_recv = cfg.max_num_tokens * world_size

    # ---------- dispatch ----------
    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg

        # ---------1. get counts of send and recv for each rank -----------
        # 1.1 token nums to send to each rank
        send_counts = [0] * self.world_size
        # 1.2 token id to send to each rank
        token_map = [[] for _ in range(self.world_size)]
        # 1.3 token meta data, need update for combine
        meta_map = [[] for _ in range(self.world_size)]
        for t, expert_list in enumerate(indices.tolist()):
            for k, e in enumerate(expert_list):
                dst_rank = e // self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend(
                    [e, self.rank, t, k, 0]
                )  # srcGobalExpert, srcRank, srcIndex, expert index

        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # 1.3 token nums to recv from each rank
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # ---------2. send and recv buffer, order by tokens on each rank ----------
        send_buf = torch.cat([dp_x[idx_list] for idx_list in token_map], dim=0)
        total_recv = int(recv_counts_t.sum().item())
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device
        )

        # 2.1 meta buf for send and recv
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # ---------3. dispatch send_buf to recv_buf by recv and send counts--------------
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )

        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1, self.META_DIM)
        # ---------4. define output tensor of dispatch ------------
        # 4.1 num tokens per expert
        expert_num_tokens = torch.zeros(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        # 4.2 token tensor on each expert
        expert_x = torch.empty(
            (self.num_local_experts, self.max_recv, cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device,
        )
        expert_meta = torch.empty(
            (self.num_local_experts, self.max_recv, self.META_DIM),
            dtype=torch.int32,
            device=device,
        )
        # ---------5. dispatch send_meta to recv_meta by recv and send counts------
        # ---------6. write tokens to each expert on each rank ------
        # 6.1 fetch the local expert id of corresponding token i
        for i in range(total_recv):
            global_eid = int(recv_meta[i, 0].item())
            local_eid = global_eid % self.num_local_experts
            # output, store token buf and token meta and token nums of each expert
            expert_x[local_eid, expert_num_tokens[local_eid]] = recv_buf[i]
            expert_meta[local_eid, expert_num_tokens[local_eid]] = recv_meta[i]
            expert_num_tokens[local_eid] += 1
        # 6.2 after dispatch, token nums and token and meta of token on expert
        return expert_num_tokens, expert_x, expert_meta

    # ---------- combine ----------
    def combine(
        self,
        out_tokens: torch.Tensor,  # output, (max num tokens, token dim)
        weights: torch.Tensor,  # topk weight
        expert_meta: torch.Tensor,  # input
        expert_y: torch.Tensor,  # input, (num_local_experts, max_num_tokens * num_dp, token_dim)
        expert_num_tokens: torch.Tensor,
    ):  # input
        device = out_tokens.device
        cfg = self.cfg

        # 1. count send-back tokens in cur rank
        send_counts = [0] * self.world_size
        # 1.1 token that will send back
        y_map = [[] for _ in range(self.world_size)]
        # 1.2 meta info of each token that send back to its src rank
        meta_map = [[] for _ in range(self.world_size)]

        # 2. traverse each token of each local expert of each rank, fill into send_counts and y_map and meta_map
        for local_eid in range(self.num_local_experts):
            cnt = int(expert_num_tokens[local_eid].item())
            for j in range(cnt):
                # meta info token j of local eid
                meta = expert_meta[local_eid, j]
                dst_rank = int(meta[1].item())
                send_counts[dst_rank] += 1
                # token j and its meta that send back to dst rank/local eid
                y_map[dst_rank].append(expert_y[local_eid, j].unsqueeze(0))
                meta_map[dst_rank].extend(meta.tolist())
        # token nums that cur rank plan to send to other ranks
        send_counts_t = torch.tensor(send_counts, dtype=torch.long, device=device)
        # token nums that will recv from other ranks
        recv_counts_t = torch.empty(self.world_size, dtype=torch.long, device=device)
        # call all2all to send send counts and recv recv_counts_t at each rank by all2all
        dist.all_to_all_single(recv_counts_t, send_counts_t)
        # 3.send buffers of each rank, that is, the tokens at its experts
        y_map_tensors = []
        for sub_list in y_map:
            if sub_list:
                y_map_tensors.append(torch.cat(sub_list, dim=0))
            else:
                y_map_tensors.append(
                    torch.empty((0, cfg.hidden_dim), dtype=cfg.out_dtype, device=device)
                )
        send_buf = torch.cat(y_map_tensors, dim=0)
        # 4. flatten send meta by tokens
        send_meta = torch.tensor(
            [v for sub in meta_map for v in sub], dtype=torch.int32, device=device
        ).view(-1, self.META_DIM)
        # 5. total recv tokens of cur rank
        total_recv = int(recv_counts_t.sum().item())
        # 6. recv buffer of cur rank
        recv_buf = torch.empty(
            total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device
        )
        recv_meta = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device
        )
        # 7. call all2all to send and recv for each rank
        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        # 8. call all2all to send meta and recv meta for each rank
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c * self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c * self.META_DIM for c in send_counts_t.tolist()],
        )
        # 9. restore recv meta
        recv_meta = recv_meta.view(-1, self.META_DIM)

        # 10. write back tokens from recv buf, per meta info, and do weighted sum
        for i in range(total_recv):
            src_token = int(recv_meta[i, 2].item())
            src_k = int(recv_meta[i, 3].item())
            src_rank = int(recv_meta[i, 1].item())
            w = weights[src_token, src_k].to(torch.float32)
            out_tokens[src_token] += recv_buf[i].to(torch.float32) * w

        return out_tokens


def generate_input(
    num_experts, experts_per_token, hidden_dim, max_num_tokens, seed, rank, world_size
):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    cfg = MoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        max_num_tokens=max_num_tokens,
        in_dtype=torch.float16,
        out_dtype=torch.float16,
    )
    rank_data = RankTestData(cfg, gen, rank)
    return cfg, rank_data, rank, world_size


def ref_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data

    ata = PyTorchAllToAll(cfg, rank, world_size)

    expert_num, expert_x, expert_meta = ata.dispatch(rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(
        cfg.max_num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )

    ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[: rank_data.num_tokens]


def check_implementation(data: input_t, output: output_t):
    expected = ref_kernel(data)
    if output.device != expected.device:
        return False, f"Output device mismatch: {output.device} != {expected.device}"
    res = torch.allclose(output, expected, rtol=1e-2, atol=5e-3)
    if not res:
        return False, f"Output values mismatch, {output} != {expected}"

    return True, ""

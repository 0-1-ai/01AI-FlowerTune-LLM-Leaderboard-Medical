"""mmfl: A Flower / FlowerTune app.

서버 전략(FedAvg)을 확장하여 통신량(바이트)을 라운드별로 추적합니다.
 - 기본 FedAvg 동작 유지
 - 각 라운드의 송수신 크기를 MB 단위로 누적/로그 출력
 - 과도한 통신량(200,000MB 초과) 경고
"""

from collections import OrderedDict
from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.

    This class behaves just like FedAvg but also tracks the communication
    costs associated with `train` over FL rounds.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training.

        다음 라운드의 훈련 메시지들을 구성하며, 동시에 송신되는 모델 크기를 기록합니다.
        """
        messages = super().configure_train(server_round, arrays, config, grid)

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages.

        클라이언트들이 반환한 메시지들의 크기를 기록하고, 기본 FedAvg 로직으로 집계합니다.
        """
        # Track communication costs
        self.comm_tracker.track(replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        return arrays, metrics


class FlexLoraStrategy(FedAvg):
    """FlexLoRA aggregation on top of the Flower App-based strategy."""

    A_SUFFIX = ".lora_A.weight"
    B_SUFFIX = ".lora_B.weight"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        messages = super().configure_train(server_round, arrays, config, grid)
        self.comm_tracker.track(messages)
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        replies = list(replies)
        if not replies:
            return None, None
        self.comm_tracker.track(replies)

        try:
            aggregated_arrays = self._aggregate_flexlora_arrays(replies)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log(
                WARN,
                "FlexLoRA aggregation failed on round %s (%s). Falling back to FedAvg.",
                server_round,
                exc,
            )
            return super().aggregate_train(server_round, replies)

        metric_record = self._aggregate_metrics(replies)
        return aggregated_arrays, metric_record

    @staticmethod
    def _get_record(msg: Message, key: str):
        try:
            return msg.content[key]
        except (KeyError, TypeError):
            return None

    def _extract_num_examples(self, msg: Message) -> int:
        metrics_record = self._get_record(msg, "metrics")
        default_examples = 1
        if metrics_record is None:
            return default_examples
        metrics = getattr(metrics_record, "metrics", {})
        for key in ("num-examples", "num_examples"):
            value = metrics.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default_examples
        return default_examples

    def _aggregate_metrics(self, replies: list[Message]) -> MetricRecord:
        total_examples = 0
        weighted_loss = 0.0
        for msg in replies:
            metrics_record = self._get_record(msg, "metrics")
            if metrics_record is None:
                continue
            metrics = getattr(metrics_record, "metrics", {})
            num_examples = self._extract_num_examples(msg)
            total_examples += num_examples
            if "train_loss" in metrics:
                weighted_loss += float(metrics["train_loss"]) * num_examples
        if total_examples > 0 and weighted_loss > 0:
            return MetricRecord({"train_loss": weighted_loss / total_examples})
        return MetricRecord()

    def _aggregate_flexlora_arrays(self, replies: list[Message]) -> ArrayRecord:
        template_state: Optional[OrderedDict[str, torch.Tensor]] = None
        module_acc: dict[str, dict] = {}
        extra_acc: dict[str, dict] = {}

        for msg in replies:
            arrays_record = self._get_record(msg, "arrays")
            if arrays_record is None:
                continue
            state_dict = arrays_record.to_torch_state_dict()
            if template_state is None:
                template_state = OrderedDict(
                    (k, v.detach().clone()) for k, v in state_dict.items()
                )
            num_examples = self._extract_num_examples(msg)
            module_entries = self._collect_module_entries(state_dict)

            for module_key, entry in module_entries.items():
                if "A" not in entry or "B" not in entry:
                    continue
                A_tensor = entry["A"].detach().float()
                B_tensor = entry["B"].detach().float()
                weighted = torch.matmul(B_tensor, A_tensor)
                acc = module_acc.setdefault(
                    module_key,
                    {
                        "sum": torch.zeros_like(weighted),
                        "weight": 0,
                        "rank": A_tensor.shape[0],
                        "A_key": entry["A_key"],
                        "B_key": entry["B_key"],
                        "A_dtype": entry["A"].dtype,
                        "B_dtype": entry["B"].dtype,
                    },
                )
                acc["sum"] += weighted * num_examples
                acc["weight"] += num_examples

            for key, tensor in state_dict.items():
                if key.endswith(self.A_SUFFIX) or key.endswith(self.B_SUFFIX):
                    continue
                tens = tensor.detach().float()
                acc = extra_acc.setdefault(
                    key,
                    {
                        "sum": torch.zeros_like(tens),
                        "weight": 0,
                        "dtype": tensor.dtype,
                    },
                )
                acc["sum"] += tens * num_examples
                acc["weight"] += num_examples

        if template_state is None or not module_acc:
            raise RuntimeError("No LoRA modules available for FlexLoRA aggregation.")

        aggregated_state: dict[str, torch.Tensor] = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for module_key, acc in module_acc.items():
            avg = acc["sum"] / max(acc["weight"], 1)
            new_A, new_B = self.distribute_weight_fast(avg, acc["rank"], device)
            aggregated_state[acc["A_key"]] = new_A.to(acc["A_dtype"])
            aggregated_state[acc["B_key"]] = new_B.to(acc["B_dtype"])

        for key, acc in extra_acc.items():
            if acc["weight"] == 0:
                continue
            aggregated_state[key] = (acc["sum"] / acc["weight"]).to(acc["dtype"])

        ordered = OrderedDict()
        for key in template_state.keys():
            if key in aggregated_state:
                ordered[key] = aggregated_state[key]
            else:
                ordered[key] = template_state[key]
        return ArrayRecord(ordered)

    def _collect_module_entries(
        self, state_dict: OrderedDict[str, torch.Tensor]
    ) -> dict[str, dict]:
        modules: dict[str, dict] = {}
        for key, tensor in state_dict.items():
            if key.endswith(self.A_SUFFIX):
                module = key[: -len(self.A_SUFFIX)]
                entry = modules.setdefault(module, {})
                entry["A"] = tensor
                entry["A_key"] = key
            elif key.endswith(self.B_SUFFIX):
                module = key[: -len(self.B_SUFFIX)]
                entry = modules.setdefault(module, {})
                entry["B"] = tensor
                entry["B_key"] = key
        return modules

    @staticmethod
    def distribute_weight_fast(
        svd_weights: torch.Tensor, max_rank: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = svd_weights.to(device)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        u = u[:, :max_rank]
        s = s[:max_rank]
        vh = vh[:max_rank, :]
        lora_B = u @ torch.diag(s)
        lora_A = vh
        return lora_A.cpu(), lora_B.cpu()


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""
    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        """주어진 메시지들의 배열 레코드 크기를 MB로 환산해 누적합니다."""
        comm_cost = (
            sum(
                record.count_bytes()
                for msg in messages
                if msg.has_content()
                for record in msg.content.array_records.values()
            )
            / 1024**2
        )

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )

"""GPU 作业抢占/迁移/暂停规划器（纯规划，不连接真实 GPU，不执行 kill 或 checkpoint）。

输入集群 GPU 拓扑（节点型号、单卡显存）、当前任务的占用与检查点成本、
配额与截止时间，输出一份分配计划：受害者顺序、释放后的卡集合、
无法满足时的原因，以及每个任务的预计恢复成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

# 迁移链的最大深度，保证抢占/迁移链必然终止。
MAX_CHAIN_DEPTH = 8


class Action(Enum):
    MIGRATE = "migrate"    # 迁移到其他有空闲显存的卡
    PAUSE = "pause"        # 检查点后挂起，释放整卡
    PREEMPT = "preempt"    # 直接抢占，丢失未检查点的工作


@dataclass(frozen=True)
class GPU:
    gpu_id: str
    node: str
    model: str
    memory_total_mb: int
    # 基线占用（系统预留 + 不属于任何受管任务的部分）。
    memory_used_mb: int = 0

    @property
    def free_memory_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


@dataclass(frozen=True)
class Job:
    job_id: str
    user: str
    gpu_ids: Tuple[str, ...]
    memory_per_gpu_mb: int
    # 预计自行释放时间（秒，相对于规划时刻）；None 表示不会自行释放。
    expected_release_s: Optional[float] = None
    # 做一次 checkpoint 的耗时（秒），迁移/暂停前必须支付。
    checkpoint_cost_s: float = 0.0
    # 被抢占时丢失的未检查点工作量（秒）。
    lost_work_s: float = 0.0
    # 公平配额权重。
    quota: int = 1
    # 是否允许被抢占（不允许时只能迁移/暂停）。
    preemptible: bool = True


@dataclass(frozen=True)
class Request:
    job_id: str
    user: str
    num_gpus: int
    memory_per_gpu_mb: int
    model: Optional[str] = None
    # 必须在此前拿到卡（秒，相对于规划时刻）；None 表示不限。
    deadline_s: Optional[float] = None
    quota: int = 1


@dataclass(frozen=True)
class Victim:
    job_id: str
    action: Action
    # 预计恢复成本（秒）：迁移/暂停为 checkpoint 耗时，抢占为丢失的工作量。
    recovery_cost_s: float
    # 该动作释放（腾空）的卡。
    freed_gpus: Tuple[str, ...]
    # 多少秒后卡可用。
    available_after_s: float


@dataclass
class Plan:
    success: bool
    victims: List[Victim] = field(default_factory=list)
    freed_gpus: Set[str] = field(default_factory=set)
    reasons: List[str] = field(default_factory=list)
    recovery_costs: Dict[str, float] = field(default_factory=dict)


class Planner:
    """对集群做一次性的分配规划。构造时校验输入，plan() 可多次调用。"""

    def __init__(self, gpus: Sequence[GPU], jobs: Sequence[Job]):
        self.gpus: Dict[str, GPU] = {}
        for gpu in gpus:
            if gpu.gpu_id in self.gpus:
                raise ValueError("重复的 GPU 键: %s" % gpu.gpu_id)
            if gpu.memory_used_mb < 0 or gpu.memory_used_mb > gpu.memory_total_mb:
                raise ValueError("GPU %s 显存占用非法" % gpu.gpu_id)
            self.gpus[gpu.gpu_id] = gpu

        self.jobs: Dict[str, Job] = {}
        for job in jobs:
            if job.job_id in self.jobs:
                raise ValueError("重复的任务键: %s" % job.job_id)
            for gid in job.gpu_ids:
                if gid not in self.gpus:
                    raise ValueError("任务 %s 引用了未知 GPU: %s" % (job.job_id, gid))
            self.jobs[job.job_id] = job

        # 校验显存账目一致：各任务在卡上的占用不能超过该卡的已用量。
        used_by_jobs: Dict[str, int] = {gid: 0 for gid in self.gpus}
        for job in self.jobs.values():
            for gid in job.gpu_ids:
                used_by_jobs[gid] += job.memory_per_gpu_mb
        for gid, used in used_by_jobs.items():
            if used > self.gpus[gid].memory_used_mb:
                raise ValueError(
                    "GPU %s 上任务占用 %dMB 超过该卡已用量 %dMB"
                    % (gid, used, self.gpus[gid].memory_used_mb)
                )

    def plan(self, request: Request) -> Plan:
        if request.num_gpus <= 0:
            raise ValueError("请求的 GPU 数量必须为正")
        if request.memory_per_gpu_mb <= 0:
            raise ValueError("请求的显存必须为正")
        return self._plan(request)

    def _plan(self, request: Request) -> Plan:
        reasons: List[str] = []
        compatible = {
            g.gpu_id
            for g in self.gpus.values()
            if request.model is None or g.model == request.model
        }
        if len(compatible) < request.num_gpus:
            return Plan(
                success=False,
                reasons=[
                    "型号 %s 的 GPU 数量不足：需要 %d，集群仅有 %d"
                    % (request.model or "任意", request.num_gpus, len(compatible))
                ],
            )

        free_map = {gid: g.free_memory_mb for gid, g in self.gpus.items()}

        # 1) 截止时间前会自然释放的任务，不需要惊动它们。
        released_jobs: Set[str] = set()
        released_gpus: Set[str] = set()
        wait_s = 0.0
        for job in self.jobs.values():
            if job.expected_release_s is None:
                continue
            if request.deadline_s is not None and job.expected_release_s > request.deadline_s:
                continue
            released_jobs.add(job.job_id)
            wait_s = max(wait_s, job.expected_release_s)
            for gid in job.gpu_ids:
                free_map[gid] += job.memory_per_gpu_mb
                released_gpus.add(gid)

        def usable_count() -> int:
            return sum(
                1 for gid in compatible if free_map[gid] >= request.memory_per_gpu_mb
            )

        if usable_count() >= request.num_gpus:
            if released_jobs:
                reasons.append("等待 %.0fs 内自然释放后即可满足，无需抢占" % wait_s)
            return Plan(success=True, freed_gpus=released_gpus, reasons=reasons)

        # 2) 选择受害者：超配额用户优先，其次丢失工作量小、检查点便宜。
        victims: List[Victim] = []
        candidates = self._rank_candidates(request, released_jobs)
        for job in candidates:
            if usable_count() >= request.num_gpus:
                break
            evicted = self._evict(job, request, compatible, free_map)
            if evicted is not None:
                victims.extend(evicted)

        if usable_count() >= request.num_gpus:
            freed = {
                gid for gid in compatible if free_map[gid] >= request.memory_per_gpu_mb
            }
            return Plan(
                success=True,
                victims=victims,
                freed_gpus=freed,
                reasons=reasons,
                recovery_costs={v.job_id: v.recovery_cost_s for v in victims},
            )

        reasons.extend(self._failure_reasons(request, compatible))
        return Plan(
            success=False,
            victims=victims,
            reasons=reasons,
            recovery_costs={v.job_id: v.recovery_cost_s for v in victims},
        )

    def _user_usage(self) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        for job in self.jobs.values():
            usage[job.user] = usage.get(job.user, 0) + len(job.gpu_ids)
        return usage

    def _user_quota(self) -> Dict[str, int]:
        quota: Dict[str, int] = {}
        for job in self.jobs.values():
            quota[job.user] = quota.get(job.user, 0) + job.quota
        return quota

    def _over_quota_users(self) -> Set[str]:
        usage = self._user_usage()
        quota = self._user_quota()
        total_quota = sum(quota.values()) or 1
        total_gpus = len(self.gpus)
        over = set()
        for user, used in usage.items():
            fair_share = total_gpus * quota.get(user, 0) / total_quota
            if used > fair_share:
                over.add(user)
        return over

    def _rank_candidates(self, request: Request, released_jobs: Set[str]) -> List[Job]:
        over = self._over_quota_users()

        def usable(job: Job) -> bool:
            if job.job_id in released_jobs:
                return False
            can_checkpoint = (
                request.deadline_s is None
                or job.checkpoint_cost_s <= request.deadline_s
            )
            return can_checkpoint or job.preemptible

        def key(job: Job):
            return (
                0 if job.user in over else 1,
                job.lost_work_s,
                job.checkpoint_cost_s,
                job.job_id,
            )

        return sorted((j for j in self.jobs.values() if usable(j)), key=key)

    def _evict(
        self,
        job: Job,
        request: Request,
        compatible: Set[str],
        free_map: Dict[str, int],
    ) -> Optional[List[Victim]]:
        can_checkpoint = (
            request.deadline_s is None or job.checkpoint_cost_s <= request.deadline_s
        )
        if can_checkpoint:
            moves = self._find_migration(job, free_map, compatible, frozenset(), 0)
            if moves is not None:
                move_list, new_map = moves
                free_map.clear()
                free_map.update(new_map)
                return [
                    Victim(
                        job_id=job_id,
                        action=Action.MIGRATE,
                        recovery_cost_s=self.jobs[job_id].checkpoint_cost_s,
                        freed_gpus=self.jobs[job_id].gpu_ids,
                        available_after_s=self.jobs[job_id].checkpoint_cost_s,
                    )
                    for job_id, _targets in move_list
                ]
            for gid in job.gpu_ids:
                free_map[gid] += job.memory_per_gpu_mb
            return [
                Victim(
                    job_id=job.job_id,
                    action=Action.PAUSE,
                    recovery_cost_s=job.checkpoint_cost_s,
                    freed_gpus=job.gpu_ids,
                    available_after_s=job.checkpoint_cost_s,
                )
            ]
        if job.preemptible:
            for gid in job.gpu_ids:
                free_map[gid] += job.memory_per_gpu_mb
            return [
                Victim(
                    job_id=job.job_id,
                    action=Action.PREEMPT,
                    recovery_cost_s=job.lost_work_s,
                    freed_gpus=job.gpu_ids,
                    available_after_s=0.0,
                )
            ]
        return None

    def _find_migration(
        self,
        job: Job,
        free_map: Dict[str, int],
        avoid: Set[str],
        visited: frozenset,
        depth: int,
    ) -> Optional[Tuple[List[Tuple[str, Tuple[str, ...]]], Dict[str, int]]]:
        """尝试把 job 迁走，必要时递归迁走挡路的任务形成迁移链。

        返回 (迁移序列, 新的 free_map)；无法迁移返回 None。
        visited 与深度上限共同保证链必然终止。
        """
        if job.job_id in visited or depth >= MAX_CHAIN_DEPTH:
            return None
        visited = visited | {job.job_id}

        direct = self._place(job, free_map, avoid)
        if direct is not None:
            targets, new_map = direct
            return ([(job.job_id, targets)], new_map)

        for blocker in self._blockers(job, visited):
            sub = self._find_migration(blocker, free_map, avoid, visited, depth + 1)
            if sub is None:
                continue
            sub_moves, sub_map = sub
            retry = self._place(job, sub_map, avoid)
            if retry is not None:
                targets, new_map = retry
                return (sub_moves + [(job.job_id, targets)], new_map)
        return None

    def _place(
        self, job: Job, free_map: Dict[str, int], avoid: Set[str]
    ) -> Optional[Tuple[Tuple[str, ...], Dict[str, int]]]:
        candidates = [
            gid
            for gid in self.gpus
            if gid not in job.gpu_ids and free_map[gid] >= job.memory_per_gpu_mb
        ]
        # 优先迁到与请求型号不兼容的卡上，把目标型号的卡腾出来。
        candidates.sort(key=lambda gid: (gid in avoid, free_map[gid], gid))
        if len(candidates) < len(job.gpu_ids):
            return None
        chosen = tuple(candidates[: len(job.gpu_ids)])
        new_map = dict(free_map)
        for gid in chosen:
            new_map[gid] -= job.memory_per_gpu_mb
        for gid in job.gpu_ids:
            new_map[gid] += job.memory_per_gpu_mb
        return chosen, new_map

    def _blockers(self, job: Job, visited: frozenset) -> List[Job]:
        blockers = [
            other
            for other in self.jobs.values()
            if other.job_id != job.job_id and other.job_id not in visited
        ]
        blockers.sort(
            key=lambda other: (
                other.lost_work_s,
                other.checkpoint_cost_s,
                other.job_id,
            )
        )
        return blockers

    def _failure_reasons(self, request: Request, compatible: Set[str]) -> List[str]:
        reasons: List[str] = []
        # 理论上限：驱逐所有可被驱逐的任务后每张卡能腾出的显存。
        evictable: Dict[str, int] = {gid: 0 for gid in self.gpus}
        blocked_by_deadline = False
        for job in self.jobs.values():
            can_checkpoint = (
                request.deadline_s is None
                or job.checkpoint_cost_s <= request.deadline_s
            )
            if can_checkpoint or job.preemptible:
                for gid in job.gpu_ids:
                    evictable[gid] += job.memory_per_gpu_mb
            else:
                blocked_by_deadline = True

        best_free = {
            gid: self.gpus[gid].free_memory_mb + evictable[gid] for gid in compatible
        }
        need_mb = request.num_gpus * request.memory_per_gpu_mb
        if sum(best_free.values()) < need_mb:
            reasons.append(
                "兼容型号显存总量不足：最多可腾出 %dMB，需要 %dMB"
                % (sum(best_free.values()), need_mb)
            )
        else:
            fit = sum(
                1 for free in best_free.values() if free >= request.memory_per_gpu_mb
            )
            if fit < request.num_gpus:
                reasons.append(
                    "显存碎片：总量足够，但单卡可用显存达到 %dMB 的卡只有 %d 张（需要 %d 张）"
                    % (request.memory_per_gpu_mb, fit, request.num_gpus)
                )
        if blocked_by_deadline and request.deadline_s is not None:
            reasons.append(
                "部分任务检查点耗时超过截止时间 %.0fs 且不可抢占，无法在截止时间前释放"
                % request.deadline_s
            )
        future = [
            job.expected_release_s
            for job in self.jobs.values()
            if job.expected_release_s is not None
            and request.deadline_s is not None
            and job.expected_release_s > request.deadline_s
        ]
        if future:
            reasons.append(
                "有任务将在 %.0fs 后自然释放，但已超过截止时间 %.0fs"
                % (min(future), request.deadline_s)
            )
        if not reasons:
            reasons.append("资源不足，无法满足请求")
        return reasons

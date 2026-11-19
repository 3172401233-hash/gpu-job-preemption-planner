import unittest

from gpu_planner import Action, GPU, Job, Planner, Request


def make_gpu(gid, model="A100", total=40000, used=0, node="n0"):
    return GPU(gpu_id=gid, node=node, model=model,
               memory_total_mb=total, memory_used_mb=used)


class TestFragmentation(unittest.TestCase):
    def test_single_gpu_fragmentation_fails(self):
        # 两张卡各剩 10GB，总量 20GB 够，但任何单卡都放不下 20GB。
        gpus = [make_gpu("g0", used=30000), make_gpu("g1", used=30000)]
        planner = Planner(gpus, [])
        plan = planner.plan(Request("r1", "u0", 1, 20000, model="A100"))
        self.assertFalse(plan.success)
        self.assertTrue(any("碎片" in r for r in plan.reasons), plan.reasons)

    def test_exact_fit_on_one_card_succeeds(self):
        gpus = [make_gpu("g0", used=30000), make_gpu("g1", used=10000)]
        planner = Planner(gpus, [])
        plan = planner.plan(Request("r1", "u0", 1, 20000, model="A100"))
        self.assertTrue(plan.success)
        self.assertEqual(plan.victims, [])


class TestCrossModel(unittest.TestCase):
    def test_model_mismatch_fails(self):
        gpus = [make_gpu("g0", model="V100"), make_gpu("g1", model="V100")]
        planner = Planner(gpus, [])
        plan = planner.plan(Request("r1", "u0", 1, 10000, model="A100"))
        self.assertFalse(plan.success)
        self.assertTrue(any("数量不足" in r for r in plan.reasons), plan.reasons)

    def test_cross_model_migration_preferred(self):
        # A100 被任务占用，V100 有空位：应把 A100 上的任务迁到 V100，
        # 而不是暂停或抢占它。
        gpus = [
            make_gpu("a0", model="A100", used=40000),
            make_gpu("v0", model="V100", used=0),
        ]
        job = Job("j1", "u1", ("a0",), 40000, checkpoint_cost_s=10)
        planner = Planner(gpus, [job])
        plan = planner.plan(Request("r1", "u0", 1, 40000, model="A100"))
        self.assertTrue(plan.success)
        self.assertEqual(len(plan.victims), 1)
        self.assertEqual(plan.victims[0].action, Action.MIGRATE)
        self.assertEqual(plan.freed_gpus, {"a0"})
        self.assertEqual(plan.recovery_costs["j1"], 10)


class TestPreemptionChain(unittest.TestCase):
    def test_migration_chain_relocates_blocker(self):
        # A 占满 g0；g1 有 30GB 空闲但被 B 占 10GB；g2 有 30GB 空闲。
        # 需要 40GB：必须先把 B 迁到 g2，再把 A 迁到 g1，链长为 2。
        gpus = [
            make_gpu("g0", used=40000),
            make_gpu("g1", used=10000),
            make_gpu("g2", used=10000),
        ]
        job_a = Job("A", "ua", ("g0",), 40000, checkpoint_cost_s=10)
        job_b = Job("B", "ub", ("g1",), 10000, checkpoint_cost_s=999)
        planner = Planner(gpus, [job_a, job_b])
        plan = planner.plan(Request("r1", "u0", 1, 40000, model="A100"))
        self.assertTrue(plan.success, plan.reasons)
        self.assertEqual([v.job_id for v in plan.victims], ["B", "A"])
        self.assertTrue(all(v.action == Action.MIGRATE for v in plan.victims))
        self.assertIn("g0", plan.freed_gpus)
        self.assertEqual(plan.recovery_costs, {"A": 10, "B": 999})

    def test_cyclic_chain_terminates_and_falls_back(self):
        # A 占 g0、B 占 g1，都占满，互为目标形成环；规划器必须终止
        # 并退化为暂停，而不是死循环。
        gpus = [make_gpu("g0", used=40000), make_gpu("g1", used=40000)]
        job_a = Job("A", "ua", ("g0",), 40000, checkpoint_cost_s=10)
        job_b = Job("B", "ub", ("g1",), 40000, checkpoint_cost_s=20)
        planner = Planner(gpus, [job_a, job_b])
        plan = planner.plan(Request("r1", "u0", 1, 40000, model="A100"))
        self.assertTrue(plan.success, plan.reasons)
        self.assertEqual(len(plan.victims), 1)
        self.assertEqual(plan.victims[0].action, Action.PAUSE)
        self.assertEqual(plan.victims[0].job_id, "A")

    def test_preempt_prefers_least_lost_work(self):
        # 截止时间太紧无法 checkpoint，只能抢占；应选丢失工作量小的。
        gpus = [make_gpu("g0", used=40000), make_gpu("g1", used=40000)]
        cheap = Job("cheap", "u1", ("g0",), 40000,
                    checkpoint_cost_s=100, lost_work_s=5)
        costly = Job("costly", "u2", ("g1",), 40000,
                     checkpoint_cost_s=100, lost_work_s=5000)
        planner = Planner(gpus, [cheap, costly])
        plan = planner.plan(
            Request("r1", "u0", 1, 40000, model="A100", deadline_s=10))
        self.assertTrue(plan.success, plan.reasons)
        self.assertEqual(len(plan.victims), 1)
        self.assertEqual(plan.victims[0].job_id, "cheap")
        self.assertEqual(plan.victims[0].action, Action.PREEMPT)
        self.assertEqual(plan.recovery_costs["cheap"], 5)


class TestFairQuota(unittest.TestCase):
    def test_over_quota_user_is_victim_first(self):
        # u1 配额 1 却占 3 卡，u2 配额 3 只占 1 卡；
        # 即使 u2 的任务检查点更便宜，也应先动超配额的 u1。
        gpus = [make_gpu("g%d" % i, used=40000) for i in range(4)]
        jobs = [
            Job("j1", "u1", ("g0",), 40000, checkpoint_cost_s=50, quota=1),
            Job("j2", "u1", ("g1",), 40000, checkpoint_cost_s=50, quota=1),
            Job("j3", "u1", ("g2",), 40000, checkpoint_cost_s=50, quota=1),
            Job("j4", "u2", ("g3",), 40000, checkpoint_cost_s=1, quota=9),
        ]
        planner = Planner(gpus, jobs)
        plan = planner.plan(Request("r1", "u0", 1, 40000, model="A100"))
        self.assertTrue(plan.success, plan.reasons)
        self.assertEqual(len(plan.victims), 1)
        self.assertIn(plan.victims[0].job_id, {"j1", "j2", "j3"})


class TestDeadline(unittest.TestCase):
    def test_deadline_already_impossible(self):
        # 唯一占用者不可抢占，checkpoint 要 100s，自然释放要 1000s，
        # 而截止时间是 5s：必然失败并说明原因。
        gpus = [make_gpu("g0", used=40000)]
        job = Job("j1", "u1", ("g0",), 40000,
                  checkpoint_cost_s=100, expected_release_s=1000,
                  preemptible=False)
        planner = Planner(gpus, [job])
        plan = planner.plan(
            Request("r1", "u0", 1, 40000, model="A100", deadline_s=5))
        self.assertFalse(plan.success)
        self.assertTrue(any("截止时间" in r for r in plan.reasons), plan.reasons)

    def test_natural_release_before_deadline_succeeds(self):
        gpus = [make_gpu("g0", used=40000)]
        job = Job("j1", "u1", ("g0",), 40000, expected_release_s=30)
        planner = Planner(gpus, [job])
        plan = planner.plan(
            Request("r1", "u0", 1, 40000, model="A100", deadline_s=60))
        self.assertTrue(plan.success, plan.reasons)
        self.assertEqual(plan.victims, [])
        self.assertEqual(plan.freed_gpus, {"g0"})


class TestDuplicateKeys(unittest.TestCase):
    def test_duplicate_gpu_id_rejected(self):
        gpus = [make_gpu("g0"), make_gpu("g0")]
        with self.assertRaises(ValueError):
            Planner(gpus, [])

    def test_duplicate_job_id_rejected(self):
        gpus = [make_gpu("g0", used=10000)]
        jobs = [
            Job("j1", "u1", ("g0",), 10000),
            Job("j1", "u2", ("g0",), 10000),
        ]
        with self.assertRaises(ValueError):
            Planner(gpus, jobs)

    def test_unknown_gpu_reference_rejected(self):
        gpus = [make_gpu("g0")]
        jobs = [Job("j1", "u1", ("ghost",), 10000)]
        with self.assertRaises(ValueError):
            Planner(gpus, jobs)


if __name__ == "__main__":
    unittest.main()

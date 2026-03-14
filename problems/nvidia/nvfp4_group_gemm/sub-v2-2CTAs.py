import torch
from task import input_t, output_t
from typing import Tuple
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16


class Sm100BlockScaledDenseGemmKernel:
    def __init__(self, mma_tiler_mn, cluster_shape_mn):
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float16
        self.sf_vec_size = 16
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.occupancy = 1
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)))

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_tiler_mn)
        mma_inst_tile_k = 4
        mk = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], mk * mma_inst_tile_k)
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape), self.mma_tiler[1], self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.mma_inst_shape_mn_sfb = (self.mma_tiler_mn[0], cute.round_up(self.mma_tiler_mn[1], 128))
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb)
        self.mma_tiler_sfb = (self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1], mk * mma_inst_tile_k)
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, False, self.c_layout, self.c_dtype)
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(tiled_mma, self.mma_tiler, self.a_dtype, self.b_dtype, self.epi_tile, self.c_dtype, self.c_layout, self.sf_dtype, self.sf_vec_size, self.smem_capacity, self.occupancy)
        self.prefetch_stage = self.num_ab_stage
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)

    @cute.jit
    def __call__(self, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, m, n, k, l):
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        self.c_dtype = c_ptr.value_type
        self.a_major_mode, self.b_major_mode, self.c_layout = (tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, utils.LayoutEnum.ROW_MAJOR)
        self._setup_attributes()

        a_tensor = cute.make_tensor(a_ptr, cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2)))
        b_tensor = cute.make_tensor(b_ptr, cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2)))
        c_tensor = cute.make_tensor(c_ptr, cute.make_ordered_layout((m, cute.assume(n, 32), l), order=(1, 0, 2)))
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_tiler_mn)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(self.ab_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb)

        cop = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_a, tma_ta = cute.nvgpu.make_tiled_tma_atom_A(cop, a_tensor, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_b, tma_tb = cute.nvgpu.make_tiled_tma_atom_B(cop, b_tensor, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        sfa_smem = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_sfa, tma_tsfa = cute.nvgpu.make_tiled_tma_atom_A(cop, sfa_tensor, sfa_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16)
        sfb_smem = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        tma_sfb, tma_tsfb = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_tensor, sfb_smem, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.num_tma_load_bytes = (cute.size_in_bytes(self.ab_dtype, a_smem) + cute.size_in_bytes(self.ab_dtype, b_smem) + cute.size_in_bytes(self.sf_dtype, sfa_smem) + cute.size_in_bytes(self.sf_dtype, sfb_smem)) * atom_thr_size

        epi_smem = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_c, tma_tc = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem, self.epi_tile)

        grid = self._compute_grid(c_tensor, self.cta_tile_shape_mnk, self.cluster_shape_mn)

        self.buffer_align_bytes = 1024
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)], self.buffer_align_bytes]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)], self.buffer_align_bytes]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], self.buffer_align_bytes]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage
        self.kernel(tiled_mma, tiled_mma_sfb, tma_a, tma_ta, tma_b, tma_tb, tma_sfa, tma_tsfa, tma_sfb, tma_tsfb, tma_c, tma_tc, self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk, self.a_smem_layout_staged, self.b_smem_layout_staged, self.sfa_smem_layout_staged, self.sfb_smem_layout_staged, self.c_smem_layout_staged, self.epi_tile).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), smem=self.shared_storage.size_in_bytes())

    @cute.kernel
    def kernel(self, tiled_mma, tiled_mma_sfb, tma_atom_a, mA, tma_atom_b, mB, tma_atom_sfa, mSFA, tma_atom_sfb, mSFB, tma_atom_c, mC, cluster_layout_vmnk, cluster_layout_sfb_vmnk, a_sl, b_sl, sfa_sl, sfb_sl, c_sl, epi_tile):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a); cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa); cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        bcc = cluster_layout_vmnk.get_flat_coord(cta_rank)
        bcc_sfb = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)
        mma_coord = (bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ab_pipe = pipeline.PipelineTmaAsync.create(barrier_storage=storage.ab_full_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1), tx_count=self.num_tma_load_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        acc_pipe = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.acc_full_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.epilog_warp_id)), cta_layout_vmnk=cluster_layout_vmnk)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=self.epilog_warp_id[0])

        cute.arch.cluster_arrive_relaxed()
        sC = storage.sC.get_tensor(c_sl.outer, swizzle=c_sl.inner)
        sA = storage.sA.get_tensor(a_sl.outer, swizzle=a_sl.inner)
        sB = storage.sB.get_tensor(b_sl.outer, swizzle=b_sl.inner)
        sSFA = storage.sSFA.get_tensor(sfa_sl); sSFB = storage.sSFB.get_tensor(sfb_sl)

        amm = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, bcc, mcast_mode=2)
        bmm = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, bcc, mcast_mode=1)
        sfamm = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, bcc, mcast_mode=2)
        sfbmm = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, bcc_sfb, mcast_mode=1)

        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gC = cute.local_tile(mC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_cnt = cute.size(gA, mode=[3])

        thr = tiled_mma.get_slice(mma_tile_coord_v)
        thr_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr.partition_A(gA); tCgB = thr.partition_B(gB)
        tCgSFA = thr.partition_A(gSFA); tCgSFB = thr_sfb.partition_B(gSFB)
        tCgC = thr.partition_C(gC)

        acl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, bcc[2], acl, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        bcl = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, bcc[1], bcl, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, bcc[2], acl, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA); tAgSFA = cute.filter_zeros(tAgSFA)
        sfbcl = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, bcc_sfb[1], sfbcl, cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB); tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA); tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        cute.arch.cluster_wait()

        if warp_idx == self.tma_warp_id:
            ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            gA_s = tAgA[(None, mma_coord[0], None, mma_coord[2])]
            gB_s = tBgB[(None, mma_coord[1], None, mma_coord[2])]
            gSFA_s = tAgSFA[(None, mma_coord[0], None, mma_coord[2])]
            sn = mma_coord[1]
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                sn = mma_coord[1] // 2
            gSFB_s = tBgSFB[(None, sn, None, mma_coord[2])]
            for pf in cutlass.range(0, self.prefetch_stage, unroll=1):
                cute.prefetch(tma_atom_a, gA_s[(None, pf)]); cute.prefetch(tma_atom_b, gB_s[(None, pf)])
                cute.prefetch(tma_atom_sfa, gSFA_s[(None, pf)]); cute.prefetch(tma_atom_sfb, gSFB_s[(None, pf)])
            peek = ab_pipe.producer_try_acquire(ps)
            for ki in cutlass.range(0, k_cnt, 1, unroll=1):
                ab_pipe.producer_acquire(ps, peek)
                bar = ab_pipe.producer_get_barrier(ps)
                cute.copy(tma_atom_a, gA_s[(None, ps.count)], tAsA[(None, ps.index)], tma_bar_ptr=bar, mcast_mask=amm)
                cute.copy(tma_atom_b, gB_s[(None, ps.count)], tBsB[(None, ps.index)], tma_bar_ptr=bar, mcast_mask=bmm)
                cute.copy(tma_atom_sfa, gSFA_s[(None, ps.count)], tAsSFA[(None, ps.index)], tma_bar_ptr=bar, mcast_mask=sfamm)
                cute.copy(tma_atom_sfb, gSFB_s[(None, ps.count)], tBsSFB[(None, ps.index)], tma_bar_ptr=bar, mcast_mask=sfbmm)
                if ki < k_cnt - self.prefetch_stage:
                    nk = ps.count + self.prefetch_stage
                    cute.prefetch(tma_atom_a, gA_s[(None, nk)]); cute.prefetch(tma_atom_b, gB_s[(None, nk)])
                    cute.prefetch(tma_atom_sfa, gSFA_s[(None, nk)]); cute.prefetch(tma_atom_sfb, gSFB_s[(None, nk)])
                ps.advance()
                if ps.count < k_cnt:
                    peek = ab_pipe.producer_try_acquire(ps)
            ab_pipe.producer_tail(ps)

        elif warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            atp = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(atp, tCtAcc_fake.layout)
            sfa_tp = cute.recast_ptr(atp + tcgen05.find_tmem_tensor_col_offset(tCtAcc), dtype=self.sf_dtype)
            tCtSFA_l = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_sl, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tp, tCtSFA_l)
            sfb_tp = cute.recast_ptr(atp + tcgen05.find_tmem_tensor_col_offset(tCtAcc) + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=self.sf_dtype)
            tCtSFB_l = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_sl, (None, None, None, 0)))
            tCtSFB = cute.make_tensor(sfb_tp, tCtSFB_l)
            s2t_sfa, csSFA, ctSFA = self._s2t(sSFA, tCtSFA)
            s2t_sfb, csSFB, ctSFB = self._s2t(sSFB, tCtSFB)
            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            aps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            peek = ab_pipe.consumer_try_wait(cs)
            tCtSFB_mma = tCtSFB
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                off = cutlass.Int32((mma_coord[1] % 2) * 2)
                tCtSFB_mma = cute.make_tensor(cute.recast_ptr(atp + tcgen05.find_tmem_tensor_col_offset(tCtAcc) + tcgen05.find_tmem_tensor_col_offset(tCtSFA) + off, dtype=self.sf_dtype), tCtSFB_l)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _ in range(k_cnt):
                ab_pipe.consumer_wait(cs, peek)
                sc = (None, None, None, None, cs.index)
                cute.copy(s2t_sfa, csSFA[sc], ctSFA); cute.copy(s2t_sfb, csSFB[sc], ctSFB)
                for kp in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                    kc = (None, None, kp, cs.index)
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[(None, None, kp)].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipe.consumer_release(cs); cs.advance()
                if cs.count < k_cnt: peek = ab_pipe.consumer_try_wait(cs)
            acc_pipe.producer_commit(aps)

        elif warp_idx in self.epilog_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            atp = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(atp, tCtAcc_fake.layout)
            t2r_op = sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, False)
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            t2r = tcgen05.make_tmem_copy(t2r_op, tAcc_epi[(None, None, 0, 0)])
            t2r_thr = t2r.get_slice(tidx)
            tTR_tAcc = t2r_thr.partition_S(tAcc_epi)
            gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = t2r_thr.partition_D(gC_epi)
            tTR_rAcc = cute.make_fragment(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            r2s_op = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, t2r)
            r2s = cute.make_tiled_copy_D(r2s_op, t2r)
            r2s_thr = r2s.get_slice(tidx)
            tRS_sC = r2s_thr.partition_D(sC); tRS_rC = r2s.retile(tTR_rC)
            sC2 = cute.group_modes(sC, 0, 2); gC2 = cute.group_modes(gC_epi, 0, 2)
            bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC2, gC2)
            bSG_gC = bSG_gC[(None, None, None, *mma_coord)]
            acs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            acc_pipe.consumer_wait(acs)
            tTR_tAcc2 = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            bSG_gC2 = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
            for si in range(cute.size(tTR_tAcc2.shape, mode=[3])):
                cute.copy(t2r, tTR_tAcc2[(None, None, None, si)], tTR_rAcc)
                tRS_rC.store(tTR_rAcc.load().to(self.c_dtype))
                cute.copy(r2s, tRS_rC, tRS_sC[(None, None, None, si)])
                cute.arch.fence_view_async_shared()
                if warp_idx == self.epilog_warp_id[0]:
                    cute.copy(tma_atom_c, bSG_sC[(None, si)], bSG_gC2[(None, si)])
            tmem.relinquish_alloc_permit(); tmem.free(atp)

    def _s2t(self, sSF, tSF):
        csSF = cute.filter_zeros(sSF); ctSF = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
        tc = tcgen05.make_s2t_copy(atom, ctSF); thr = tc.get_slice(0)
        return tc, tcgen05.get_s2t_smem_desc_tensor(tc, thr.partition_S(csSF)), thr.partition_D(ctSF)

    @staticmethod
    def _compute_stages(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, sf_dtype, sf_vec_size, smem_capacity, occupancy):
        num_acc_stage = 1; num_c_stage = 2
        ab_bytes = cute.size_in_bytes(a_dtype, sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)) + cute.size_in_bytes(b_dtype, sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)) + cute.size_in_bytes(sf_dtype, blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)) + cute.size_in_bytes(sf_dtype, blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1))
        c_bytes = cute.size_in_bytes(c_dtype, sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1))
        num_ab_stage = (smem_capacity - (1024 + c_bytes * num_c_stage)) // ab_bytes
        num_c_stage += (smem_capacity - ab_bytes * num_ab_stage - (1024 + c_bytes * num_c_stage)) // c_bytes
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn):
        return (cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]), cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]), c.layout.shape[2])


_compiled_kernel_cache = None

def compile_kernel():
    global _compiled_kernel_cache
    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    # Use (1,1) cluster to avoid compilation timeout
    my_kernel = Sm100BlockScaledDenseGemmKernel((128, 128), (1, 1))
    _compiled_kernel_cache = cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, 0, 0, 0, 0)
    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    compiled_func = compile_kernel()
    results = []
    for (a, b, c), (sfa_p, sfb_p), (m, n, k, l) in zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes):
        # problem_sizes contains full K (e.g. 7168), NOT K//2
        # CuTe handles FP4 packing internally given ab_dtype=Float4E2M1FN
        compiled_func(
            make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(sf_dtype, sfa_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(sf_dtype, sfb_p.data_ptr(), cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            m, n, k, l,
        )
        results.append(c)
    return results


'''
g: 8; k: [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168]; m: [80, 176, 128, 72, 64, 248, 96, 160]; n: [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]; seed: 1111
 ⏱ 115 ± 0.0 µs
 ⚡ 115 µs 🐌 115 µs

g: 8; k: [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]; m: [40, 76, 168, 72, 164, 148, 196, 160]; n: [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168]; seed: 1111
 ⏱ 73.4 ± 0.04 µs
 ⚡ 73.4 µs 🐌 73.5 µs

g: 2; k: [4096, 4096]; m: [192, 320]; n: [3072, 3072]; seed: 1111
 ⏱ 20.7 ± 0.01 µs
 ⚡ 20.7 µs 🐌 20.7 µs

g: 2; k: [1536, 1536]; m: [128, 384]; n: [4096, 4096]; seed: 1111
 ⏱ 12.5 ± 0.01 µs
 ⚡ 12.5 µs 🐌 12.8 µs
 '''